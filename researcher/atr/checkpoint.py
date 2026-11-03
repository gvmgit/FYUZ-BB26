"""The checkpoint: one policy decision for every model, tool and agent-to-agent call, with a decision record.

The policies live in policies/atr.cedar, outside the agent code. In AWS mode Amazon Verified Permissions evaluates them
(AvpPdp); offline, LocalPdp evaluates the same file with Cedar semantics (default deny, a forbid wins over a permit).
Whether a model or tool is callable at all comes from the registry: nothing is callable until it is approved, and
withdrawing it (the pause) refuses every call to it.

Every decision is written to the audit sink with the run id and trace id, and printed as one JSON line, which lands in
CloudWatch Logs when this runs in AgentCore Runtime or Lambda.
"""
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config

POLICY_FILE = Path(__file__).resolve().parents[1] / "policies" / "atr.cedar"
NS = "Atr"

# which entity type an action's resource is; anything else is a tool named after the action
RESOURCE_TYPE = {"InvokeModel": "Model", "ApproveChange": "Card", "ApplyNetworkChange": "Card"}


class PolicyViolation(Exception):
    pass


def resource_for(action: str, resource: Optional[str] = None) -> Tuple[str, str]:
    """(entity type, id) of what an action acts on: ask_<agent> -> that Agent, InvokeModel -> the Model, else the Tool."""
    if action.startswith("ask_"):
        return "Agent", action[4:]
    kind = RESOURCE_TYPE.get(action, "Tool")
    return kind, resource or action


# ---------------------------------------------------------------- policies
@dataclass
class Policy:
    id: str
    effect: str                                  # permit | forbid
    principal: Tuple[str, Optional[str]]         # ("any"|"eq"|"is", "Agent::scientist" | "Agent")
    actions: Optional[set]                       # None = any action
    resource: Tuple[str, Optional[str]]
    when: Optional[Tuple[str, str]] = None       # (attribute, required value) on the resource
    text: str = ""


_STMT = re.compile(r"(?:^|\n)((?://[^\n]*\n)*)\s*(permit|forbid)\s*\((.*?)\)\s*(?:when\s*\{(.*?)\})?\s*;", re.S)
_ENT = re.compile(rf'{NS}::(\w+)::"([^"]+)"')


def _scope(part: str, word: str) -> Tuple[str, Optional[str]]:
    part = part.strip()
    if part == word:
        return "any", None
    m = re.match(rf'{word}\s*==\s*{NS}::(\w+)::"([^"]+)"$', part)
    if m:
        return "eq", f"{m.group(1)}::{m.group(2)}"
    m = re.match(rf"{word}\s+is\s+{NS}::(\w+)$", part)
    if m:
        return "is", m.group(1)
    raise ValueError(f"unsupported {word} scope: {part}")


def parse_policies(text: str) -> List[Policy]:
    """Parse the subset of Cedar this project writes (scopes ==, is, in [...]; a `resource.<attr> == "<v>"` condition)."""
    out = []
    for comments, effect, scope, cond in _STMT.findall(text):
        m = re.search(r"//\s*id:\s*(\S+)", comments)
        pid = m.group(1) if m else f"policy-{len(out) + 1}"
        parts = [p.strip() for p in re.split(r",(?![^\[]*\])", scope)]
        principal, action, resource = parts
        if action == "action":
            actions = None
        else:
            actions = {a for _, a in _ENT.findall(action)}
        when = None
        if cond.strip():
            c = re.match(r'\s*resource\.(\w+)\s*==\s*"([^"]+)"\s*$', cond)
            if not c:
                raise ValueError(f"unsupported condition in {pid}: {cond}")
            when = (c.group(1), c.group(2))
        stmt = f"{effect} ({scope})" + (f"\nwhen {{{cond}}}" if cond.strip() else "") + ";"
        out.append(Policy(pid, effect, _scope(principal, "principal"), actions, _scope(resource, "resource"), when, stmt))
    if not out:
        raise ValueError("no policies found")
    return out


def load_policies(path: Path = POLICY_FILE) -> List[Policy]:
    return parse_policies(Path(path).read_text(encoding="utf-8"))


def _matches(scope, etype: str, eid: str) -> bool:
    kind, value = scope
    return kind == "any" or (kind == "is" and value == etype) or (kind == "eq" and value == f"{etype}::{eid}")


class LocalPdp:
    """Cedar semantics over the parsed policies: default deny; any matching forbid wins over every permit."""

    name = "local-cedar"

    def __init__(self, policies: Optional[List[Policy]] = None):
        self.policies = policies or load_policies()

    def decide(self, principal: Tuple[str, str], action: str, resource: Tuple[str, str], attrs: Dict[str, str]):
        permits, forbids = [], []
        for p in self.policies:
            if not (_matches(p.principal, *principal) and _matches(p.resource, *resource)):
                continue
            if p.actions is not None and action not in p.actions:
                continue
            if p.when and attrs.get(p.when[0]) != p.when[1]:
                continue
            (forbids if p.effect == "forbid" else permits).append(p.id)
        if forbids:
            return "DENY", forbids
        return ("ALLOW", permits) if permits else ("DENY", [])


class AvpPdp:
    """Amazon Verified Permissions: the policy store holds policies/atr.cedar (deploy/sync_policies.py)."""

    name = "verified-permissions"

    def __init__(self, store_id: Optional[str] = None, client=None, region: Optional[str] = None):
        if client is None:
            import boto3
            client = boto3.client("verifiedpermissions", region_name=region)
        self.client, self.store = client, store_id or os.environ["ATR_POLICY_STORE_ID"]
        self._names: Dict[str, str] = {}

    def _name(self, policy_id: str) -> str:
        """Our policy name ('// id:' line, stored as the AVP description) for an AVP policy id."""
        if policy_id not in self._names:
            try:
                d = self.client.get_policy(policyStoreId=self.store, policyId=policy_id)
                self._names[policy_id] = d["definition"]["static"].get("description") or policy_id
            except Exception:  # noqa: BLE001  (the id is still a usable record)
                self._names[policy_id] = policy_id
        return self._names[policy_id]

    def decide(self, principal, action, resource, attrs):
        def ent(t, i):
            return {"entityType": f"{NS}::{t}", "entityId": i}
        entities = [{"identifier": ent(*resource), "attributes": {k: {"string": v} for k, v in attrs.items()}},
                    {"identifier": ent(*principal), "attributes": {}}]
        r = self.client.is_authorized(policyStoreId=self.store, principal=ent(*principal),
                                      action={"actionType": f"{NS}::Action", "actionId": action},
                                      resource=ent(*resource), entities={"entityList": entities})
        return r["decision"], [self._name(p["policyId"]) for p in r.get("determiningPolicies", [])]


# ---------------------------------------------------------------- registry
DEFAULT_MODELS = ("playbook",)          # the offline model; AWS mode registers the Bedrock model id at deploy time


class LocalRegistry:
    """Registered models and tools with a status: approved, withdrawn (paused) or absent (not callable)."""

    def __init__(self, path: Optional[Path] = None, approve_all_tools: bool = True):
        self.path = Path(path or config.RUNS / "registry.json")
        self.items: Dict[str, dict] = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.approve_all_tools = approve_all_tools

    def status(self, kind: str, name: str) -> str:
        item = self.items.get(f"{kind}#{name}")
        if item:
            return item["status"]
        if kind in ("Tool", "Agent") and self.approve_all_tools:
            return "approved"
        if kind == "Model" and name in DEFAULT_MODELS:
            return "approved"
        return "unregistered"

    def set(self, kind: str, name: str, status: str, by: str = "operator") -> dict:
        item = {"id": f"{kind}#{name}", "kind": kind, "name": name, "status": status, "by": by, "ts": time.time()}
        self.items[item["id"]] = item
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, indent=2))
        return item

    def list(self) -> List[dict]:
        return sorted(self.items.values(), key=lambda i: i["id"])


class DynamoRegistry(LocalRegistry):
    """The same registry in DynamoDB (table ATR_REGISTRY_TABLE). Tools and models must be registered explicitly."""

    def __init__(self, table=None, table_name: Optional[str] = None, region: Optional[str] = None):
        if table is None:
            import boto3
            table = boto3.resource("dynamodb", region_name=region).Table(table_name or os.environ["ATR_REGISTRY_TABLE"])
        self.table, self.approve_all_tools, self._cache = table, False, {}

    def status(self, kind, name):
        key = f"{kind}#{name}"
        hit = self._cache.get(key)
        if hit and time.time() - hit[1] < 5:      # re-read every 5 s, so a withdrawal takes effect mid-run
            return hit[0]
        item = self.table.get_item(Key={"id": key}).get("Item")
        st = item["status"] if item else ("approved" if kind == "Agent" else "unregistered")
        self._cache[key] = (st, time.time())
        return st

    def set(self, kind, name, status, by="operator"):
        item = {"id": f"{kind}#{name}", "kind": kind, "name": name, "status": status, "by": by,
                "ts": str(round(time.time(), 3))}
        self.table.put_item(Item=item)
        self._cache.pop(item["id"], None)
        return item

    def list(self):
        return sorted(self.table.scan().get("Items", []), key=lambda i: i["id"])


# ---------------------------------------------------------------- the checkpoint
def new_run_id(unit: str) -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime()) + f"-{unit}-{uuid.uuid4().hex[:4]}"


@dataclass
class Checkpoint:
    pdp: object
    registry: object
    audit: object = None
    run_id: str = field(default_factory=lambda: new_run_id("run"))
    trace_id: Optional[str] = None
    echo: bool = True
    seq: int = 0

    def __post_init__(self):
        self.trace_id = self.trace_id or self.run_id

    def decide(self, principal_type: str, principal: str, action: str, resource: Optional[str] = None,
               context: Optional[dict] = None) -> dict:
        rtype, rid = resource_for(action, resource)
        status = self.registry.status(rtype, rid)
        decision, policies = self.pdp.decide((principal_type, principal), action, (rtype, rid), {"status": status})
        self.seq += 1
        rec = {"kind": "decision", "run_id": self.run_id, "trace_id": self.trace_id, "seq": self.seq,
               "principal": f"{principal_type}::{principal}", "action": action, "resource": f"{rtype}::{rid}",
               "resource_status": status, "decision": decision, "policies": policies, "pdp": self.pdp.name,
               **({"context": context} if context else {})}
        self.record(rec)
        return rec

    def enforce(self, principal_type: str, principal: str, action: str, resource: Optional[str] = None,
                context: Optional[dict] = None) -> dict:
        rec = self.decide(principal_type, principal, action, resource, context)
        if rec["decision"] != "ALLOW":
            why = (f"policy {', '.join(rec['policies'])}" if rec["policies"]
                   else f"no policy permits it ({rec['resource']} is {rec['resource_status']})")
            raise PolicyViolation(f"{rec['principal']} may not {action} on {rec['resource']}: {why}")
        return rec

    def record(self, rec: dict) -> None:
        rec.setdefault("run_id", self.run_id)
        rec.setdefault("trace_id", self.trace_id)
        rec.setdefault("epoch", round(time.time(), 3))
        if self.audit is not None:
            self.audit.write(rec)
        if self.echo:
            print("ATR_AUDIT " + json.dumps(rec, default=str), flush=True)


def guarded(model, role: str, checkpoint: Checkpoint, model_id: str):
    """Put the checkpoint in front of a Strands model: every call asks InvokeModel on Model::<model_id> first, so a
    withdrawn model is refused before any request reaches it."""
    inner = model.stream

    async def stream(*a, **kw):
        checkpoint.enforce("Agent", role, "InvokeModel", model_id)
        async for event in inner(*a, **kw):
            yield event

    model.stream = stream
    return model


def local_checkpoint(audit=None, run_id: Optional[str] = None, registry=None, echo: bool = False) -> Checkpoint:
    return Checkpoint(LocalPdp(), registry or LocalRegistry(), audit, run_id or new_run_id("local"), echo=echo)
