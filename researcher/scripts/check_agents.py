"""Checks the guarantees the agents must keep, whatever the model decides.

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_agents.py
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from atr import runtime  # noqa: E402
from atr.agents import Review  # noqa: E402
from atr.anf import PolicyViolation  # noqa: E402
from atr.gateway import ToolError  # noqa: E402
from atr.store import SQLiteStore  # noqa: E402
from atr.strands_agents import build_investigation_agent, run_strands_investigation  # noqa: E402
from atr.world import Fault  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


quiet = lambda m: None  # noqa: E731
fault = Fault("tilt", 3, 16)

# 1. Full Strands run finds the planted fault, rules out rivals, and escalates to a human
rt = runtime.build("local", fault, emit=quiet)
out = run_strands_investigation(rt.researcher, rt.mesh, rt.model_for)
head = out["headline"]
check("Strands agents locate the planted fault",
      head and head["claim"]["kind"] == "tilt" and head["claim"]["cell"] == 3 and head["claim"]["value"] == 16,
      head["title"] if head else "no finding")
rivals = {c["claim"]["kind"]: c["verdict"] for c in out["cards"] if c["claim"]["cell"] == 3 and not c["headline"]}
check("every rival explanation for the same cell is eliminated (rejected, or ruled out head-to-head); none left ambiguous",
      not head["alternatives_not_excluded"] and rivals and all(v in ("refuted", "ruled out (head-to-head)") for v in rivals.values()),
      str(rivals))
check("every card has a human-in-the-loop status and no live change is made",
      all(c["escalation"]["status"] in ("awaiting_human_approval", "needs_human_review") for c in out["cards"]))
check("the confirmed finding carries control and holdout evidence",
      head["control"] is not None and head["holdout"] is not None)
rt.close()

# 2. Publish guard: refused until every claim has a final Critic verdict (regardless of what the model does)
rt = runtime.build("local", fault, emit=quiet)
_, inv = build_investigation_agent(rt.researcher, rt.mesh, rt.model_for)
publish = inv.tools_for("orchestrator")[0]
try:
    publish()
    check("publish is refused before any investigation", False)
except Exception as e:  # noqa: BLE001
    check("publish is refused before any investigation", "no investigation has been run" in str(e), str(e)[:70])
observe, propose = inv.tools_for("scientist")
observe(), propose()
test, control, null, holdout, replication = inv.tools_for("experimenter")
review = inv.tools_for("critic")[0]
first = next(iter(inv.claims))
test(first)
status = json.loads(review(first))["status"]
check("the Critic demands a control run first", status == "needs_control", status)
try:
    publish()
    check("publish is refused while a claim is still awaiting control/null/holdout", False)
except Exception as e:  # noqa: BLE001
    check("publish is refused while a claim is still awaiting control/null/holdout", "without a final" in str(e))
control(first)
status = json.loads(review(first))["status"]
check("then the Critic demands the twin-bias null test", status == "needs_null", status)
null(first)
status = json.loads(review(first))["status"]
check("then a held-out replication", status == "needs_holdout", status)
holdout(first)
status = json.loads(review(first))["status"]
check("then, for a tilt claim, the second instrument", status == "needs_replication", status)
replication(first)
status = json.loads(review(first))["status"]
check("and only then is the claim accepted", status == "accepted", status)
# defence in depth: even if a verdict said 'accepted', publishing needs control, null and holdout on record
for cid in inv.claims:
    inv.reviews[cid] = Review("accepted", [], "forced for the test")
inv.nulls.clear()
try:
    publish()
    check("publish refuses an accepted claim with no null test on record", False)
except Exception as e:  # noqa: BLE001
    check("publish refuses an accepted claim with no null test on record", "null test" in str(e), str(e)[:80])
dummy = object()
for cid in inv.claims:
    inv.controls[cid] = inv.nulls[cid] = inv.holdouts[cid] = dummy
inv.replications.clear()
try:
    publish()
    check("publish refuses an accepted tilt claim with no second-instrument result", False)
except Exception as e:  # noqa: BLE001
    check("publish refuses an accepted tilt claim with no second-instrument result", "second-instrument" in str(e),
          str(e)[:80])
rt.close()

# 3. Access control: agents can only call what the policy gives them
rt = runtime.build("local", fault, emit=quiet)
for agent, tool in [("critic", "run_experiment"), ("scientist", "test_claim"), ("experimenter", "publish_findings")]:
    try:
        rt.mesh.authorize(agent, tool)
        check(f"{agent} is denied {tool}", False)
    except PolicyViolation:
        check(f"{agent} is denied {tool}", True)
try:
    rt.researcher.gw.invoke("run_experiment", {"config": {"cells": {}}, "seeds": [1]}, agent="critic")
    check("the Critic cannot run experiments through the gateway", False)
except ToolError:
    check("the Critic cannot run experiments through the gateway", True)
rt.close()

# 4. The plain pipeline and the Strands pipeline agree
rt = runtime.build("local", fault, emit=quiet)
classic = rt.researcher.investigate()
rt.close()
check("classic pipeline and Strands pipeline reach the same finding",
      classic["headline"]["claim"]["estimate"] == head["claim"]["estimate"] and classic["experiments"] == out["experiments"],
      f"{classic['headline']['claim']['estimate']} / {out['experiments']} experiments")

# 5. No fault in the network: the researcher must not invent one
rt = runtime.build("local", None, emit=quiet)
none_out = run_strands_investigation(rt.researcher, rt.mesh, rt.model_for)
check("with no hidden fault, no claim is confirmed", none_out["headline"] is None,
      "confirmed " + str(none_out["headline"]["title"]) if none_out["headline"] else "")
rt.close()

# 6. A mismatched twin with no fault is not mistaken for a fault (Strands path)
rt = runtime.build("local", None, emit=quiet, twin_bias={1: -3.0, 4: 2.5}, field_seed=100)
mis = run_strands_investigation(rt.researcher, rt.mesh, rt.model_for)
check("twin mismatch and no fault: nothing is confirmed, and a calibration alert goes to a human",
      mis["headline"] is None and any(c.get("twin_calibration_alert") and c["escalation"]["status"] == "needs_human_review"
                                      for c in mis["cards"]))
rt.close()

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
