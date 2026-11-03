"""The checkpoint: Cedar policies outside the agents, allow / deny / pause, human approval, one trace id per run.

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_checkpoint.py
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atr.anf import AgentMesh  # noqa: E402
from atr.audit import FileAudit  # noqa: E402
from atr.checkpoint import (AvpPdp, Checkpoint, LocalPdp, LocalRegistry, PolicyViolation, guarded,  # noqa: E402
                            load_policies, parse_policies)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail and not ok else ""))


TMP = Path(tempfile.mkdtemp())
policies = load_policies()
ids = {p.id for p in policies}
check("policies: every statement in policies/atr.cedar parses, each with an id", len(policies) >= 10 and len(ids) == len(policies))
check("policies: no policy text lives in the agent mesh", "DEFAULT_POLICY" not in Path("atr/anf.py").read_text())

reg = LocalRegistry(TMP / "registry.json")
reg.set("Model", "claude-test", "approved")
audit = FileAudit(TMP / "audit.jsonl")
cp = Checkpoint(LocalPdp(policies), reg, audit, run_id="2026-11-02T00-00-00-check-0001", echo=False)

allow = cp.decide("Agent", "experimenter", "run_experiment")
check("allow: the Experimenter may run an experiment", allow["decision"] == "ALLOW" and "experimenter-tools" in allow["policies"])
deny = cp.decide("Agent", "critic", "run_experiment")
check("deny: the Critic may not run experiments (default deny, no permit)", deny["decision"] == "DENY" and deny["policies"] == [])
a2a = cp.decide("Agent", "scientist", "ask_critic")
check("deny: agent-to-agent - the Scientist may not task the Critic", a2a["decision"] == "DENY")
check("allow: agent-to-agent - the Orchestrator may task the Critic",
      cp.decide("Agent", "orchestrator", "ask_critic")["decision"] == "ALLOW")
check("model: an approved model is callable", cp.decide("Agent", "scientist", "InvokeModel", "claude-test")["decision"] == "ALLOW")
check("model: an unregistered model is not", cp.decide("Agent", "scientist", "InvokeModel", "other-model")["decision"] == "DENY")

reg.set("Model", "claude-test", "withdrawn")
paused = cp.decide("Agent", "scientist", "InvokeModel", "claude-test")
check("pause: a withdrawn model is refused by the forbid policy", paused["decision"] == "DENY" and "withdrawn-is-refused" in paused["policies"])
reg.set("Tool", "run_experiment", "withdrawn")
check("pause: a withdrawn tool is refused even for the Experimenter",
      cp.decide("Agent", "experimenter", "run_experiment")["decision"] == "DENY")
reg.set("Tool", "run_experiment", "approved")
reg.set("Model", "claude-test", "approved")

check("approval: a person may approve a change", cp.decide("Human", "engineer-1", "ApproveChange", "card-1")["decision"] == "ALLOW")
agent_ok = cp.decide("Agent", "orchestrator", "ApproveChange", "card-1")
check("approval: no agent may approve a change", agent_ok["decision"] == "DENY" and "agents-never-change-state" in agent_ok["policies"])
check("approval: no agent may apply a network change",
      cp.decide("Agent", "experimenter", "ApplyNetworkChange", "card-1")["decision"] == "DENY")

try:
    cp.enforce("Agent", "critic", "run_experiment")
    raised = False
except PolicyViolation as e:
    raised = "critic" in str(e)
check("enforce: a deny raises PolicyViolation naming the caller", raised)

recs = [__import__("json").loads(line) for line in (TMP / "audit.jsonl").read_text().splitlines()]
check("trace: every decision record carries the run id and trace id, numbered in order",
      recs and all(r["run_id"] == cp.run_id and r["trace_id"] == cp.trace_id for r in recs)
      and [r["seq"] for r in recs] == sorted(r["seq"] for r in recs))

mesh = AgentMesh(audit=audit, checkpoint=cp)
try:
    mesh.authorize("scientist", "publish_findings")
    ok = False
except PolicyViolation:
    ok = True
check("mesh: delegates to the checkpoint (the Scientist may not publish)", ok)


class Echo:                                         # a stand-in Strands model
    async def stream(self, *a, **kw):
        yield {"ok": True}


async def drain(model):
    return [e async for e in model.stream([])]


m = guarded(Echo(), "scientist", cp, "claude-test")
check("model guard: an approved model streams", asyncio.run(drain(m)) == [{"ok": True}])
reg.set("Model", "claude-test", "withdrawn")
try:
    asyncio.run(drain(m))
    ok = False
except PolicyViolation:
    ok = True
check("model guard: a withdrawn model is refused before any request is sent", ok)
reg.set("Model", "claude-test", "approved")


class FakeAvp:
    def __init__(self):
        self.calls = []

    def is_authorized(self, **kw):
        self.calls.append(kw)
        return {"decision": "ALLOW", "determiningPolicies": [{"policyId": "p-123"}]}

    def get_policy(self, policyStoreId, policyId):
        return {"definition": {"static": {"description": "experimenter-tools"}}}


fa = FakeAvp()
d, names = AvpPdp("ps-1", client=fa).decide(("Agent", "experimenter"), "run_experiment", ("Tool", "run_experiment"),
                                             {"status": "approved"})
c = fa.calls[0]
check("Verified Permissions: namespaced principal/action/resource, registry status as an entity attribute",
      c["principal"] == {"entityType": "Atr::Agent", "entityId": "experimenter"}
      and c["action"] == {"actionType": "Atr::Action", "actionId": "run_experiment"}
      and c["entities"]["entityList"][0]["attributes"] == {"status": {"string": "approved"}})
check("Verified Permissions: decision records name policies by their id, not the AVP id", d == "ALLOW" and names == ["experimenter-tools"])

try:
    parse_policies('permit (principal, action, resource) when { context.x == "y" };')
    ok = False
except ValueError:
    ok = True
check("parser: refuses a condition it cannot evaluate locally (no silent allow)", ok)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
