"""The experiment protocol as a state machine (Amazon States Language), and a local interpreter for it.

The same ASL document is what the CDK stack deploys to AWS Step Functions (deploy/stepfunctions/protocol.asl.json is written
from `definition()`); every Task invokes one Lambda with {"task": name, "input": state}. Locally `LocalStateMachine` runs the
document, calling the same task functions in-process (research_cycle.ResearchCycle.tasks). The interpreter supports the subset
this protocol uses: Task, Choice, Map, Pass, Succeed, Fail; InputPath/ResultPath/ItemsPath with simple paths ($, $.a.b);
ItemSelector with $$.Map.Item.Value; Choice rules BooleanEquals, NumericGreaterThan(Equals), IsPresent; Retry and Catch on
Task with ErrorEquals ["States.ALL"].
"""
import copy
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

FN = "${ProtocolTaskFunction}"   # substituted by the CDK stack


def _task(name: str, next_: str = None, result_path: str = None, end: bool = False, catch_to: str = "Failed") -> dict:
    s = {"Type": "Task", "Resource": "arn:aws:states:::lambda:invoke",
         "Parameters": {"FunctionName": FN, "Payload": {"task": name, "input.$": "$"}},
         "ResultSelector": {"result.$": "$.Payload"}, "ResultPath": result_path or f"$.{name}",
         "Retry": [{"ErrorEquals": ["Lambda.TooManyRequestsException", "Lambda.ServiceException"], "IntervalSeconds": 2,
                    "MaxAttempts": 3, "BackoffRate": 2.0}],
         "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": catch_to}]}
    if end:
        s["End"] = True
    else:
        s["Next"] = next_
    return s


def definition() -> dict:
    return {
        "Comment": "Autonomous Telecom Researcher: one experiment protocol run (observe, hypothesise, intervene, replicate, "
                   "analyse, falsify, publish, learn)",
        "StartAt": "RunLabCampaign",
        "States": {
            "RunLabCampaign": _task("lab_campaign", "Ingest"),
            "Ingest": _task("ingest", "Observe"),
            "Observe": _task("observe", "Investigate"),
            "Investigate": _task("investigate", "CrossCheckModalities"),
            "CrossCheckModalities": _task("cross_check", "ClaimSurvived"),
            "ClaimSurvived": {"Type": "Choice", "Choices": [
                {"Variable": "$.cross_check.result.headline_found", "BooleanEquals": True, "Next": "PreRegisterReplication"}],
                "Default": "PublishWithoutReplication"},
            "PreRegisterReplication": _task("preregister_replication", "ReplicateAcrossProfiles"),
            "ReplicateAcrossProfiles": {
                "Type": "Map", "ItemsPath": "$.preregister_replication.result.profiles", "MaxConcurrency": 2,
                "ItemSelector": {"profile.$": "$$.Map.Item.Value", "run_id.$": "$.run_id"},
                "ItemProcessor": {"ProcessorConfig": {"Mode": "INLINE"}, "StartAt": "LabReplicate",
                                  "States": {"LabReplicate": _task("lab_replicate", result_path="$.replicate", end=True,
                                                                   catch_to="ReplicationFailed"),
                                             "ReplicationFailed": {"Type": "Fail", "Error": "ReplicationFailed"}}},
                "ResultPath": "$.replications", "Next": "Analyse",
                "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "Failed"}]},
            "Analyse": _task("analyse", "Falsify"),
            "Falsify": _task("falsify", "Publish"),
            "PublishWithoutReplication": _task("publish", "UpdateFindingGraph", result_path="$.publish"),
            "Publish": _task("publish", "UpdateFindingGraph"),
            "UpdateFindingGraph": _task("update_graph", "DeriveArtefacts"),
            "DeriveArtefacts": _task("derive_artefacts", "QueueRetraining"),
            "QueueRetraining": _task("queue_retraining", "Done"),
            "Done": {"Type": "Succeed"},
            "Failed": {"Type": "Fail", "Error": "ProtocolFailed", "Cause": "a task raised; see the execution history"},
        },
    }


def write_definition(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(definition(), indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------- interpreter
def _get(doc, path: str):
    if path in (None, "$"):
        return doc
    cur = doc
    for part in path[2:].split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def _set(doc: dict, path: str, value) -> dict:
    if path is None:
        return doc
    if path == "$":
        return value
    out = copy.deepcopy(doc)
    cur = out
    parts = path[2:].split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return out


def _select(selector: dict, state, item=None) -> dict:
    out = {}
    for k, v in selector.items():
        if k.endswith(".$"):
            out[k[:-2]] = item if v == "$$.Map.Item.Value" else _get(state, v)
        elif isinstance(v, dict):
            out[k] = _select(v, state, item)
        else:
            out[k] = v
    return out


def _rule(rule: dict, state) -> bool:
    try:
        val = _get(state, rule["Variable"])
    except KeyError:
        return rule.get("IsPresent") is False
    if "IsPresent" in rule:
        return rule["IsPresent"]
    if "BooleanEquals" in rule:
        return val is rule["BooleanEquals"]
    if "NumericGreaterThan" in rule:
        return val > rule["NumericGreaterThan"]
    if "NumericGreaterThanEquals" in rule:
        return val >= rule["NumericGreaterThanEquals"]
    raise ValueError(f"unsupported choice rule {rule}")


class ExecutionFailed(Exception):
    pass


class LocalStateMachine:
    def __init__(self, doc: dict, tasks: Dict[str, Callable[[dict], dict]], emit=print):
        self.doc, self.tasks, self.emit = doc, tasks, emit
        self.history: List[dict] = []

    def run(self, payload: dict) -> dict:
        return self._run(self.doc, payload)

    def _run(self, machine: dict, state: dict) -> dict:
        name = machine["StartAt"]
        while True:
            s = machine["States"][name]
            t0 = time.time()
            typ = s["Type"]
            if typ == "Succeed":
                self._log(name, typ, "succeeded", t0)
                return state
            if typ == "Fail":
                self._log(name, typ, s.get("Error", "failed"), t0)
                raise ExecutionFailed(f"{s.get('Error')}: {state.get('error', s.get('Cause', ''))}")
            if typ == "Pass":
                state = _set(state, s.get("ResultPath", "$"), s.get("Result", _get(state, s.get("InputPath", "$"))))
            elif typ == "Choice":
                nxt = next((r["Next"] for r in s["Choices"] if _rule(r, state)), s.get("Default"))
                self._log(name, typ, f"-> {nxt}", t0)
                name = nxt
                continue
            elif typ == "Task":
                task = s["Parameters"]["Payload"]["task"]
                try:
                    result = self.tasks[task](_get(state, s.get("InputPath", "$")))
                except Exception as e:  # noqa: BLE001 - routed through Catch like Step Functions
                    catch = next((c for c in s.get("Catch", []) if "States.ALL" in c["ErrorEquals"]), None)
                    self._log(name, typ, f"error: {e}", t0)
                    if catch is None:
                        raise
                    state = _set(state, catch.get("ResultPath", "$.error"), {"Error": type(e).__name__, "Cause": str(e)})
                    name = catch["Next"]
                    continue
                state = _set(state, s.get("ResultPath", "$"), {"result": result})
            elif typ == "Map":
                items = _get(state, s["ItemsPath"])
                proc = s.get("ItemProcessor") or s.get("Iterator")
                outs = []
                for it in items:
                    sub = _select(s["ItemSelector"], state, it) if "ItemSelector" in s else it
                    outs.append(self._run(proc, sub))
                state = _set(state, s.get("ResultPath", "$"), outs)
            else:
                raise ValueError(f"unsupported state type {typ}")
            self._log(name, typ, "ok", t0)
            if s.get("End"):
                return state
            name = s["Next"]

    def _log(self, name, typ, status, t0):
        rec = {"state": name, "type": typ, "status": status, "seconds": round(time.time() - t0, 2)}
        self.history.append(rec)
        if self.emit:
            self.emit(f"[protocol] {name:<26} {status}  ({rec['seconds']}s)")
