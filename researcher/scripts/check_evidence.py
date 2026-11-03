"""The evidence chain and the grader, offline: declare first, items keyed by one run id, re-grading, tamper detection.

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_evidence.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atr import grader  # noqa: E402
from atr.evidence import Evidence  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail and not ok else ""))


# ---- grader rules
t = {"kind": "tilt", "cell": 3, "value": 16}
check("grader: right kind and cell is located; estimate within 2 deg", grader.diagnosis_item(t, {"kind": "tilt", "cell": 3, "estimate": "15"}) ==
      {"located": True, "estimate_ok": True, "why": "kind and cell match"})
check("grader: wrong cell is a miss", not grader.diagnosis_item(t, {"kind": "tilt", "cell": 2, "estimate": 16})["located"])
check("grader: no finding is a miss", not grader.diagnosis_item(t, None)["located"])
check("grader: azimuth error wraps around 360", grader.diagnosis_item({"kind": "azimuth", "cell": 4, "value": 350},
                                                                      {"kind": "azimuth", "cell": 4, "estimate": "10"})["estimate_ok"])
g = grader.grade_diagnosis([{"truth": t, "predicted": {"kind": "tilt", "cell": 3, "estimate": 16}},
                            {"truth": t, "predicted": None}], {"value": 0.8})
check("grader: accuracy against the declared target (1 of 2 < 0.8 fails)", g["value"] == 0.5 and not g["pass"])
check("grader: no model in the grader", "bedrock" not in (ROOT / "atr" / "grader.py").read_text().lower().split('"""')[2])
rec = {"run_id": "r", "decision": "DENY", "policies": []}
check("grader: decisions must match and carry a record", grader.grade_decisions([{"expected": "DENY", "record": rec}], {"value": 1.0})["pass"]
      and not grader.grade_decisions([{"expected": "ALLOW", "record": rec}], {"value": 1.0})["pass"])
pause_items = [{"phase": "paused", "record": {"decision": "DENY", "policies": ["withdrawn-is-refused"]}},
               {"phase": "paused", "headline": None, "record": {"decision": "DENY", "policies": ["withdrawn-is-refused"]}},
               {"phase": "resumed", "record": {"decision": "ALLOW", "policies": ["agents-use-approved-models"]}}]
check("grader: pause = refused while withdrawn, allowed after", grader.grade_pause(pause_items, {"value": 1.0})["pass"])
trace = ([{"kind": "request", "trace_id": "x", "epoch": 1}] +
         [{"kind": "decision", "trace_id": "x", "epoch": 2 + i, "principal": f"Agent::{a}", "action": "InvokeModel"}
          for i, a in enumerate(("orchestrator", "scientist", "experimenter", "critic"))] +
         [{"kind": "published", "trace_id": "x", "epoch": 9}, {"kind": "human_approval", "trace_id": "x", "epoch": 10}])
check("grader: a complete trace passes", grader.grade_trace(trace, {"value": 1.0})["pass"])
check("grader: a trace without the human approval fails", not grader.grade_trace(trace[:-1], {"value": 1.0})["pass"])

# ---- evidence chain on a scratch copy, then the real tools against it
tmp = Path(tempfile.mkdtemp())
ev = Evidence(root=tmp / "evidence", bucket="")
(ev.root / "frozen").mkdir(parents=True)
shutil.copy(ROOT / "evidence" / "frozen" / "diagnosis_v1.json", ev.root / "frozen")
d = ev.declare("2026-11-02T00-00-00-diagnosis-test", "diagnosis", {"metric": "located_accuracy", "value": 0.5}, grader.NAME,
               "model-x", test_set="diagnosis_v1.json")
check("declare: frozen test set hash in the declaration", len(d["test_set_sha256"]) == 64)
try:
    ev.declare("2026-11-02T00-00-00-diagnosis-test", "diagnosis", {"value": 0}, grader.NAME)
    ok = False
except FileExistsError:
    ok = True
check("declare: a run cannot be re-declared after the fact", ok)
items = ev.items(d["run_id"], [{"truth": t, "predicted": {"kind": "tilt", "cell": 3, "estimate": 16}}])
r = grader.grade_diagnosis(items, d["target"])
ev.record(d["run_id"], {"unit": "diagnosis", "target": d["target"], "grader": grader.NAME, "result": r, "pass": r["pass"]})
ev.results({"units": {"diagnosis": {"run_id": d["run_id"], "metric": r["metric"], "value": r["value"], "target": 0.5, "pass": r["pass"]}}})
ev.manifest()
check("items: every row keyed by the run id and numbered", items[0]["run_id"] == d["run_id"] and items[0]["item"] == 0)
check("manifest: clean evidence verifies", ev.verify_manifest() == [])


def rebuild_rc(evidence_root: Path) -> int:
    code = ("import sys; sys.argv=['rebuild']; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s');"
            "import atr.evidence as E; E.ROOT = __import__('pathlib').Path(r'%s');"
            "import rebuild; sys.exit(rebuild.main())") % (ROOT / "tools", ROOT, evidence_root)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).returncode


check("rebuild: reproduces the results with no AWS (exit 0)", rebuild_rc(ev.root) == 0)
p = ev.root / "items" / f"{d['run_id']}.jsonl"
p.write_text(p.read_text().replace('"cell": 3, "estimate"', '"cell": 2, "estimate"'))
check("rebuild: an edited item is caught (exit 1)", rebuild_rc(ev.root) == 1)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
