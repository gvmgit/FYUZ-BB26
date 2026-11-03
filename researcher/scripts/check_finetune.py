"""Checks for the fine-tuning work: twin fine-tuning, the Critic's twin-bias null test, the dataset, the evaluation
harness and the training-job request.

NOT checked: the LoRA training itself (needs a GPU, a base model and SageMaker) and any call to a real endpoint.

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_finetune.py
(needs finetune/data from `python -m atr.cli export-dataset` for the dataset checks)
"""
import io
import json
import py_compile
import sys
import tarfile
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from atr import finetune_data as fd, finetune_eval as fe, finetune_twin as ft, runtime  # noqa: E402
from atr.audit import FileAudit  # noqa: E402
from atr.faults import Fault  # noqa: E402
from atr.topology import default_inventory  # noqa: E402
from atr.twin import RFTwin  # noqa: E402
from atr.world import World  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


quiet = lambda m: None  # noqa: E731
inv = default_inventory()
BIAS = {1: -3.0, 4: 2.5}
base = RFTwin.load_or_build()

# ---- A. twin fine-tuning ---------------------------------------------------------------------------------------------
obs = World(base, inv, None, BIAS).field_observation(1200, 555)
dry = ft.finetune(base.clone(), obs, inv, approved_by="checker", source="check", dry_run=True)
gains = {c: v.get("gain_db") for c, v in dry["cells"].items()}
check("fine-tuning updates exactly the cells whose twin is wrong (guard rejects the rest)",
      dry["accepted_cells"] == [1, 4], f"accepted {dry['accepted_cells']}")
check("accepted cells improve by about 1 dB or more on held-out field data",
      all(gains[c] >= 0.8 for c in (1, 4)), {c: round(g, 2) for c, g in gains.items() if g is not None})
check("a dry run changes nothing", dry["applied"] is False and dry["model_file"] is None)
try:
    ft.finetune(base.clone(), obs, inv, approved_by="", source="check")
    check("fine-tuning refuses without a named approver", False)
except PermissionError:
    check("fine-tuning refuses without a named approver", True)

tmp = Path(tempfile.mkdtemp())
audit = FileAudit(tmp / "audit.jsonl")
twin = base.clone()
before_c1 = twin._cell_grid(1, 10, inv[1]["azimuth"])[0].copy()
before_c2 = twin._cell_grid(2, 10, inv[2]["azimuth"])[0].copy()
rep = ft.finetune(twin, obs, inv, approved_by="checker", source="check", registry=tmp / "registry.json", audit=audit)
check("applied update writes a versioned model file and a registry entry naming the approver",
      Path(rep["model_file"]).exists() and json.loads((tmp / "registry.json").read_text())[0]["approved_by"] == "checker")
check("the update is audited", any('"twin_finetune"' in line for line in open(tmp / "audit.jsonl", encoding="utf-8")))
check("updated cell moved, untouched cell did not",
      np.abs(twin._cell_grid(1, 10, inv[1]["azimuth"])[0] - before_c1).mean() > 1.0
      and np.abs(twin._cell_grid(2, 10, inv[2]["azimuth"])[0] - before_c2).max() == 0.0)
reloaded = RFTwin.load_or_build(model_file=rep["model_file"])
check("the saved version reloads and predicts like the in-memory tuned twin",
      np.allclose(reloaded._cell_grid(1, 10, inv[1]["azimuth"])[0], twin._cell_grid(1, 10, inv[1]["azimuth"])[0], atol=1e-3))
check("the base twin on disk is untouched (rollback = pick the earlier file)",
      np.allclose(RFTwin.load_or_build()._cell_grid(1, 10, inv[1]["azimuth"])[0], before_c1, atol=1e-6))
fresh = World(base, inv, None, BIAS).field_observation(1200, 9091)     # a survey the update never saw
rows = np.ones(len(fresh), bool)
fb, fa = ft.field_fit(base, fresh, inv, rows), ft.field_fit(twin, fresh, inv, rows)
check("the gain generalises to a fresh survey (cells 1 and 4 improve, others unchanged)",
      fb[1] - fa[1] > 0.8 and fb[4] - fa[4] > 0.8 and all(abs(fb[c] - fa[c]) < 0.01 for c in (2, 3, 5)),
      {c: (round(fb[c], 2), round(fa[c], 2)) for c in fb})
w_before = World(base, inv, Fault("tilt", 3, 16), BIAS).reveal_rsrp()
w_after = World(base, inv, Fault("tilt", 3, 16), BIAS).reveal_rsrp()
check("simulated reality is built from the fixed base twin, so tuning the twin under test cannot move it",
      np.array_equal(w_before, w_after))

card = {"id": "c1", "claim": {"kind": "tilt", "cell": 3, "value": 16.0}, "escalation": {"status": "awaiting_human_approval"}}
try:
    ft.apply_finding(base.clone(), card, obs, inv)
    check("an unapproved finding cannot feed the twin", False)
except PermissionError:
    check("an unapproved finding cannot feed the twin", True)
try:
    ft.corrected_settings(inv, {"claim": {"kind": "cell_off", "cell": 3, "value": 0}})
    check("a 'cell off' finding is refused as training data", False)
except ValueError:
    check("a 'cell off' finding is refused as training data", True)

# ---- B. the Critic's twin-bias null hypothesis -------------------------------------------------------------------------
rt = runtime.build("local", None, 1200, quiet, twin_bias=BIAS, field_seed=100, twin=base)
out = rt.researcher.investigate()
alerts = [c for c in out["cards"] if c.get("twin_calibration_alert")]
check("no fault but a mismatched twin: no finding is confirmed", out["headline"] is None)
check("...and the Critic raises a twin-calibration alert that goes to a human",
      alerts and alerts[0]["escalation"]["status"] == "needs_human_review"
      and "calibration survey" in alerts[0]["escalation"]["proposed_action"],
      alerts[0]["twin_calibration_alert"]["message"][:60] if alerts else "no alert")
check("every accepted-so-far claim ran the null test (it is in the trace)",
      any("null hypothesis" in line for line in out["cards"][0]["trace"]))
rt = runtime.build("local", Fault("azimuth", 5, 210), 1200, quiet, twin_bias=BIAS, twin=base)
head = rt.researcher.investigate()["headline"]
check("a real fault is still found despite the mismatch",
      head and head["claim"]["kind"] == "azimuth" and head["claim"]["cell"] == 5, head["title"] if head else "none")
check("the finding records that it beat the null hypothesis",
      any("beats 'the twin is just biased" in k["name"] and k["passed"] for k in head["checks"]))

# ---- C. dataset -------------------------------------------------------------------------------------------------------
sets = {s: fd.build_cases(s, base.cells) for s in fd.SPLITS}
tilts = {s: {c["fault"].value for c in cs if c["fault"] and c["fault"].kind == "tilt"} for s, cs in sets.items()}
azs = {s: {c["fault"].value for c in cs if c["fault"] and c["fault"].kind == "azimuth"} for s, cs in sets.items()}
check("train/val/test use disjoint planted tilt values and azimuth shifts",
      not (tilts["train"] & tilts["val"]) and not (tilts["train"] & tilts["test"]) and not (tilts["val"] & tilts["test"])
      and not (azs["train"] & azs["val"]) and not (azs["train"] & azs["test"]) and not (azs["val"] & azs["test"]),
      f"tilts {tilts}")
seeds = [c["field_seed"] for cs in sets.values() for c in cs]
check("every case has its own survey seed (none shared across or within splits)", len(seeds) == len(set(seeds)))
labels = {fd.truth_label(c)["kind"] for cs in sets.values() for c in cs}
check("labels cover every fault kind plus twin_miscalibration and none",
      labels == {"tilt", "azimuth", "cell_off", "twin_miscalibration", "none"}, str(sorted(labels)))

rt = runtime.build("local", Fault("tilt", 2, 6), 1200, quiet, twin_bias=BIAS, field_seed=4, twin=base)
out = rt.researcher.investigate()
case = {"fault": Fault("tilt", 2, 6), "bias": BIAS, "split": "val", "case_id": "val-x"}
recs = fd.records_for(case, rt.researcher, out, inv)
rank = recs[0]
check("one ranking record plus one triage record per claim", rank["task"] == "suspect_ranking"
      and len(recs) - 1 == len(out["cards"]) and all(r["task"] == "claim_triage" for r in recs[1:]))
check("the ranking prompt shows raw diagnostics but not the rule-based suspicion scores",
      "suspicion" not in rank["messages"][1]["content"] and "obs_heard" in rank["messages"][1]["content"])
check("targets are valid JSON with the planted label",
      json.loads(rank["messages"][2]["content"]) == {"kind": "tilt", "cell": 2})
check("meta keeps the rule pipeline's own answer for offline baselines",
      rank["meta"]["system_outcome"]["kind"] in ("tilt", "none", "twin_miscalibration") and rank["meta"]["experiments"] > 0)
check("triage reasons come from a known set",
      {r["meta"]["label"]["reason"] for r in recs[1:]} <= {"confirmed", "outcompeted", "no_gain", "not_specific",
                                                          "twin_bias", "no_replication", "other"})

data = ROOT / "finetune" / "data"
if (data / "test.jsonl").exists():
    tr, va, te = (fd.load(data / f"{s}.jsonl") for s in ("train", "val", "test"))
    ids = [{r["meta"]["case_id"] for r in x} for x in (tr, va, te)]
    check("exported splits share no cases", not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]),
          f"{len(tr)}/{len(va)}/{len(te)} records")
    check("exported splits contain every label class",
          {r["meta"]["truth"]["kind"] for r in tr if r["task"] == "suspect_ranking"} ==
          {"tilt", "azimuth", "cell_off", "twin_miscalibration", "none"})
    manifest = json.loads((data / "manifest.json").read_text())
    check("manifest records the split rules", "rules" in manifest and set(manifest["splits"]) == {"train", "val", "test"})
else:
    check("exported dataset found (run `python -m atr.cli export-dataset`)", False)
    tr = te = []

# ---- D. evaluation harness --------------------------------------------------------------------------------------------
perfect = lambda rec: rec["meta"]["truth"]  # noqa: E731
synthetic = [{"task": "suspect_ranking", "meta": {"truth": {"kind": k, "cell": c}, "system_outcome": {"kind": k, "cell": c},
                                                  "rule_top_suspect_cell": c, "experiments": 70}}
             for k, c in [("tilt", 1), ("azimuth", 2), ("none", None)]]
m = fe.evaluate_ranking(synthetic, perfect)
check("evaluate_ranking scores a perfect predictor at 100%", m["joint_acc"] == 1.0 and m["n"] == 3)
m = fe.evaluate_ranking(synthetic, lambda r: {"kind": "tilt", "cell": 1})
check("evaluate_ranking scores a constant guess correctly", abs(m["joint_acc"] - 1 / 3) < 1e-9 and m["by_class"]["none"] == "0/1")
tri = [{"task": "claim_triage", "meta": {"case_id": "a", "claim_order": i, "label": {"survives_critic": i == 2}}}
       for i in range(4)]
check("triage MRR of the Scientist's order is 1/3 when the winner is third",
      abs(fe.evaluate_triage(tri, fe.scientist_order_score)["mrr"] - 1 / 3) < 1e-9)
check("triage MRR of an oracle is 1.0", fe.evaluate_triage(tri, lambda r: float(r["meta"]["label"]["survives_critic"]))["mrr"] == 1.0)
check("parse_json_answer survives chatter and rejects garbage",
      fe.parse_json_answer('Sure! {"kind":"tilt","cell":3} done') == {"kind": "tilt", "cell": 3}
      and fe.parse_json_answer("no json here") == {})
ep = fe.EndpointPredictor("ep", client=SimpleNamespace(invoke_endpoint=lambda **kw: {
    "Body": io.BytesIO(json.dumps([{"generated_text": 'Answer: {"kind":"tilt","cell":3}'}]).encode())}))
check("EndpointPredictor turns generated text into a ranking answer",
      ep({"task": "suspect_ranking", "messages": [{"content": "s"}, {"content": "u"}]}) == {"kind": "tilt", "cell": 3})
if te:
    rep_ = fe.baseline_report(tr, te)
    r = rep_["ranking"]
    check("baseline report covers majority, rule top suspect and the full pipeline",
          len(r) == 3 and r["full rule-based pipeline (runs experiments)"]["n"] > 0)
    check("the full pipeline beats the majority class on the test split",
          r["full rule-based pipeline (runs experiments)"]["joint_acc"] > r["majority class"]["joint_acc"])

# ---- E. training artifacts ---------------------------------------------------------------------------------------------
import boto3  # noqa: E402
from botocore.stub import Stubber  # noqa: E402
import data_utils  # noqa: E402
import launch_sagemaker as ls  # noqa: E402

good = [{"task": "suspect_ranking", "meta": {"case_id": "x"}, "messages": [
    {"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "{\"kind\":\"tilt\"}"}]}]
ex = data_utils.to_chat_examples(good)
check("training examples contain only the chat messages (no ground-truth meta)", ex == [{"messages": good[0]["messages"]}])
bad = [{**good[0], "messages": good[0]["messages"][:2]}]
try:
    data_utils.to_chat_examples(bad)
    check("a record without an assistant answer is rejected", False)
except ValueError:
    check("a record without an assistant answer is rejected", True)
req = ls.build_request("atr-lora-test", "arn:aws:iam::123456789012:role/r", "123456789012.dkr.ecr.us-west-2.amazonaws.com/hf:1",
                       "bucket", "some/model")
client = boto3.client("sagemaker", region_name="us-west-2", aws_access_key_id="x", aws_secret_access_key="y")
with Stubber(client) as stub:
    stub.add_response("create_training_job", {"TrainingJobArn": "arn:aws:sagemaker:us-west-2:123456789012:training-job/x"})
    try:
        client.create_training_job(**req)
        check("the SageMaker create_training_job request is valid for the installed SDK", True)
    except Exception as e:  # noqa: BLE001
        check("the SageMaker create_training_job request is valid for the installed SDK", False, str(e)[:150])
pkg = ls.package_code(Path(tmp) / "src.tar.gz")
with tarfile.open(pkg) as t:
    check("the code package contains the training script, its helpers and requirements",
          set(t.getnames()) == {"train_lora.py", "data_utils.py", "requirements.txt"})
try:
    py_compile.compile(str(ROOT / "finetune" / "train_lora.py"), doraise=True, cfile=str(tmp / "t.pyc"))
    check("train_lora.py compiles", True)
except py_compile.PyCompileError as e:
    check("train_lora.py compiles", False, str(e)[:100])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
