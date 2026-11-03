"""Checks for detector rules: the features, calibration, the deployment gate, provenance, and false alarms.

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_detector.py
"""
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from atr import detector as D  # noqa: E402
from atr.faults import Fault  # noqa: E402
from atr.store import SQLiteStore  # noqa: E402
from atr.topology import default_inventory  # noqa: E402
from atr.twin import RFTwin  # noqa: E402
from atr.world import World  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


twin, inv = RFTwin.load_or_build(), default_inventory()


def survey(fault=None, bias=None, seed=9100):
    return World(twin, inv, fault, bias).field_observation(1200, seed)


def card(cell, kind, value, status="accept", verdict="confirmed", ctype="fault_finding", cid=None):
    return {"id": cid or f"card-{cell}-{kind}-{value}-{status}", "type": ctype, "verdict": verdict,
            "engineer_status": status, "engineer_verdicts": [{"by": "Dr Rao"}] if status else [],
            "claim": {"kind": kind, "cell": cell, "value": value, "estimate": str(value)}}


# ---- features -----------------------------------------------------------------------------------------------------------
base = D.features(twin, inv, survey())
check("a healthy survey sits at the measurement-noise floor (about 2 dB, tiny offsets and gaps)",
      all(1.5 < f["shape_rmse"] < 2.6 and f["offset_abs"] < 0.6 and f["heard_gap"] < 0.1 for f in base.values()),
      f"cell 1: {({k: round(v, 2) for k, v in base[1].items()})}")
biased = D.features(twin, inv, survey(bias={4: 4.0}))
check("a constant twin offset on a cell shows up as offset, not as pattern",
      abs(biased[4]["shape_rmse"] - base[4]["shape_rmse"]) < 0.15 and 3.4 < biased[4]["offset_abs"] < 4.6,
      f"shape {base[4]['shape_rmse']:.2f} -> {biased[4]['shape_rmse']:.2f}, offset {biased[4]['offset_abs']:.2f}")
off = D.features(twin, inv, survey(Fault("cell_off", 3)))
check("a dead cell is not heard", off[3]["heard_gap"] > 0.2, f"heard_gap {off[3]['heard_gap']:.2f}")
az = D.features(twin, inv, survey(Fault("azimuth", 5, (inv[5]["azimuth"] + 120) % 360)))
check("a turned cell has a far bigger pattern residual than a healthy one", az[5]["shape_rmse"] > 3 * base[5]["shape_rmse"],
      f"{az[5]['shape_rmse']:.1f} vs {base[5]['shape_rmse']:.1f}")

# ---- calibration and the firing rule -------------------------------------------------------------------------------------
bank = [{"shape_rmse": 2.0, "heard_gap": 0.02, "share_gap": 0.03, "offset_abs": 0.2},
        {"shape_rmse": 2.1, "heard_gap": 0.04, "share_gap": 0.05, "offset_abs": 0.4},
        {"shape_rmse": 2.05, "heard_gap": 0.5, "share_gap": 0.6, "offset_abs": 3.0}]     # a biased cell, outside the guard
thr = D.calibrate(bank, margin=1.0)
check("thresholds are the largest healthy value; heard/share ignore cells already read as biased",
      thr["shape_rmse"] == 2.1 and thr["heard_gap"] == 0.04 and thr["share_gap"] == 0.05)
quiet = {"shape_rmse": 2.0, "heard_gap": 0.0, "share_gap": 0.0, "offset_abs": 0.1}
check("a pattern anomaly always fires", D.fires(thr, {**quiet, "shape_rmse": 9.0, "offset_abs": 4.0}) == ["shape_rmse"])
check("a heard-less cell fires while its offset is small", D.fires(thr, {**quiet, "heard_gap": 0.3}) == ["heard_gap"])
check("...but not when a large offset says the twin is just biased on it",
      D.fires(thr, {**quiet, "heard_gap": 0.3, "offset_abs": 3.0}) == [])
check("a large offset with an ordinary pattern is flagged as a likely twin-calibration matter",
      D.twin_offset_suspected(thr, {**quiet, "offset_abs": 3.0}) and not D.twin_offset_suspected(thr, quiet))
check("Wilson interval is sane", D.wilson(0, 100)[0] == 0.0 and 0.03 < D.wilson(0, 100)[1] < 0.04
      and D.wilson(30, 30)[1] == 1.0 and 0.88 < D.wilson(30, 30)[0] < 0.89)

# ---- building rules: provenance, gate, no leakage --------------------------------------------------------------------------
good = D.build_rule(twin, inv, card(4, "tilt", 16))
v = good["validation"]
check("a rule from an accepted tilt finding passes the gate", good["status"] == "deployable"
      and v["false_alarm"]["rate"] <= D.MAX_FALSE_ALARM and v["detection"]["rate"] >= D.MIN_DETECTION,
      f"false alarms {v['false_alarm']['fired']}/{v['false_alarm']['of']}, detected {v['detection']['fired']}/{v['detection']['of']}")
check("the rule records its source finding, its approver, thresholds and Wilson intervals",
      good["resembles"]["finding"] == "card-4-tilt-16-accept" and good["approved_by"] == "Dr Rao"
      and set(D.FEATURES) <= set(good["thresholds"]) and len(v["detection"]["ci95"]) == 2)
check("the rule says it is a trigger, not a diagnosis", "does not diagnose" in good["what_it_is"])
check("calibration, validation and fault surveys use disjoint seed ranges",
      D.SEEDS["calibration"] + D.N_CALIBRATION <= D.SEEDS["validation_healthy"]
      and D.SEEDS["validation_healthy"] + D.N_VALIDATION_HEALTHY <= D.SEEDS["validation_fault"])
c1 = D.healthy_cases(twin.cells, 30, 0, "cal-4")
c2 = D.healthy_cases(twin.cells, 30, 0, "val-h-4")
check("calibration and validation draw different twin-mismatch configurations", [c["bias"] for c in c1] != [c["bias"] for c in c2])
rejected = D.build_rule(twin, inv, card(4, "tilt", 13))
check("a fault too close to a healthy cell is rejected by the gate, with the reason, not shipped",
      rejected["status"] == "rejected" and "too close to a healthy cell" in rejected["gate_reason"],
      rejected["gate_reason"][:70])
check("rejected rules are not run by scan", D.scan([rejected], twin, inv, survey()) == [])

try:
    D.build_rule(twin, inv, card(4, "tilt", 16, status=None))
    check("a finding no engineer has accepted cannot become a rule", False)
except PermissionError:
    check("a finding no engineer has accepted cannot become a rule", True)
check("...unless the demo flag is given, which leaves no approver on the rule",
      D.build_rule(twin, inv, card(3, "cell_off", 0, status=None), allow_unreviewed=True)["approved_by"] is None)
for bad_card, why in [(card(4, "tilt", 16, verdict="refuted"), "a refuted claim"),
                      (card(4, "tilt", 16, ctype="network_insight"), "an exploration insight")]:
    try:
        D.build_rule(twin, inv, bad_card)
        check(f"{why} cannot become a rule", False)
    except ValueError:
        check(f"{why} cannot become a rule", True)

store = SQLiteStore(Path(tempfile.mkdtemp()) / "s.sqlite")
for c in (card(2, "azimuth", 160.0, cid="a1"), card(2, "azimuth", 160.0, cid="a2"),
          card(1, "tilt", 13.0, status=None, cid="u1"), card(1, "tilt", 13.0, status="reject", cid="r1"),
          card(5, "azimuth", 270.0, ctype="network_insight", cid="i1")):
    store.put({**c, "created": 1.0})
from_store = D.rules_from_store(store, twin, inv)
check("rules are built only from engineer-accepted, confirmed fault findings, one per distinct fault",
      len(from_store) == 1 and from_store[0]["cell"] == 2, f"{len(from_store)} rule(s)")

# ---- scanning surveys ----------------------------------------------------------------------------------------------------
rules = [good]
healthy = D.scan(rules, twin, inv, survey(seed=9200))[0]
check("a healthy survey is quiet", not healthy["fired"])
own = D.scan(rules, twin, inv, survey(bias={4: -4.0}, seed=9201))[0]
check("a twin that is wrong by 4 dB on the rule's own cell does NOT fire it, and is flagged as a calibration matter",
      not own["fired"] and own["twin_offset_suspected"], str(own["values"]))
elsewhere = D.scan(rules, twin, inv, survey(bias={1: -3.0, 2: 3.0}, seed=9202))[0]
check("a twin mismatch on other cells is quiet", not elsewhere["fired"])
hit = D.scan(rules, twin, inv, survey(Fault("tilt", 4, 16), seed=9203))[0]
check("the source fault fires, and says which finding it resembles",
      hit["fired"] and hit["resembles"]["finding"] == "card-4-tilt-16-accept")
other = D.scan(rules, twin, inv, survey(Fault("cell_off", 4), seed=9204))[0]
check("the rule is a screen for the cell, not for one fault: a dead cell 4 also fires it", other["fired"])
check("a fault on a cell that does not interact with it stays quiet (cell 3 dead, cell 4 rule)",
      not any(D.scan(rules, twin, inv, survey(Fault("cell_off", 3), seed=s))[0]["fired"] for s in (9205, 9206, 9207)))
# Documented, not hidden: a fault on a NEIGHBOUR can fire a rule, because it changes who serves the UEs nearby.
# So a rule says "something is wrong near here", not "this cell".

tmp = Path(tempfile.mkdtemp()) / "rules.json"
D.save_rules([good, rejected], tmp)
check("rules survive a save and load unchanged, and scan identically",
      D.load_rules(tmp) == [good, rejected]
      and D.scan(D.load_rules(tmp), twin, inv, survey(Fault("tilt", 4, 16), seed=9203)) == [hit])

rep = D.network_report([good, rejected], twin, inv, n=40, seed0=8500)
check("the network report counts only deployable rules and reports an interval",
      rep["rules"] == 1 and rep["surveys"] == 40 and len(rep["ci95"]) == 2)
check("running the rules on healthy surveys rarely triggers an investigation", rep["network_false_alarm_rate"] <= 0.10,
      f"{rep['surveys_that_triggered_any_rule']}/40")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
