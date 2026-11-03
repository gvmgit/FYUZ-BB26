"""Detector rules: a cheap early-warning check derived from an accepted finding.

The researcher spends ~90 experiments to explain a field log. A detector rule is what is left once a finding is accepted: a
screen that looks at a field survey (measurement reports) and says "cell N looks like it did when finding X happened",
without running any experiment. It is a TRIGGER for an investigation, not a diagnosis: it says a cell is anomalous and which
finding it resembles, it does not prove the cause.

Features, per cell, comparing the survey with what the twin predicts for the documented (inventory) settings:
  shape_rmse  RMS of (field - twin) RSRP after removing the cell's constant offset. A twin that is simply biased on a cell
              leaves a constant offset; a tilt or azimuth change changes the pattern. This is the detector's version of the
              Critic's "twin is just biased" null test.
  heard_gap   |share of UEs that hear the cell - share the twin predicts|  (a dead cell is not heard)
  share_gap   |share of UEs the cell serves - share the twin predicts|
A rule fires on its cell when ANY feature exceeds its threshold.

How a rule earns deployment (three separate sets of simulated surveys, so nothing is judged on data it was tuned on):
  calibration  healthy surveys (some with twin mismatch, some on the rule's own cell) set the thresholds: the largest healthy
               value seen x a margin;
  validation   FRESH healthy surveys measure the false-alarm rate, and FRESH surveys with the finding's fault measure the
               detection rate, both with Wilson 95% intervals;
  gate         deployable only if the false-alarm rate is at most 5% and detection at least 90%. Otherwise the rule is kept
               as 'rejected' with the reason, never silently shipped.
"""
import json
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config

FEATURES = ("shape_rmse", "heard_gap", "share_gap")
RULES_FILE = config.RUNS / "detector_rules.json"

MARGIN = 1.15              # threshold = largest healthy value seen x this
OFFSET_GUARD_DB = 1.5      # a constant offset at or above this is read as twin mis-calibration, not a fault
N_CALIBRATION = 200        # a threshold set at the largest of n healthy values is exceeded by a new one about 1 time in n+1
N_VALIDATION_HEALTHY = 100
N_VALIDATION_FAULT = 30
MAX_FALSE_ALARM = 0.05
MIN_DETECTION = 0.90
SEEDS = {"calibration": 4000, "validation_healthy": 5000, "validation_fault": 6000}


def features(twin, inventory: dict, obs) -> Dict[int, Dict[str, float]]:
    """Per-cell features of one field survey against the twin under inventory settings."""
    mean, _ = twin.rsrp_grid(inventory)
    pm = mean[obs.grid_idx.values]
    pred_srv = pm.argmax(1)
    obs_srv = obs.serving_cell.map({c: j for j, c in enumerate(twin.cells)}).values
    out = {}
    for j, cid in enumerate(twin.cells):
        col = obs[f"rsrp_c{cid}"].values
        heard_obs, heard_pred = ~np.isnan(col), pm[:, j] >= config.DETECT_FLOOR_DBM
        both = heard_obs & heard_pred
        if both.sum() >= 10:
            diff = col[both] - pm[both, j]
            shape = float(np.sqrt(np.mean((diff - diff.mean()) ** 2)))
            offset = float(abs(diff.mean()))
        else:
            shape, offset = 0.0, 0.0   # too few joint measurements to judge a pattern; the heard gap covers a missing cell
        out[int(cid)] = {"shape_rmse": shape,
                         "heard_gap": float(abs(heard_obs.mean() - heard_pred.mean())),
                         "share_gap": float(abs(np.mean(obs_srv == j) - np.mean(pred_srv == j))),
                         "offset_abs": offset}
    return out


def calibrate(healthy: List[Dict[str, float]], margin: float = MARGIN) -> Dict[str, float]:
    """Thresholds from healthy surveys only: the largest value seen x margin. The heard/share thresholds come only from
    healthy surveys where the cell's constant offset is below the guard, because that is the only place they are applied."""
    calm = [h for h in healthy if h["offset_abs"] < OFFSET_GUARD_DB] or healthy
    thr = {"shape_rmse": round(max(h["shape_rmse"] for h in healthy) * margin, 4),
           "heard_gap": round(max(h["heard_gap"] for h in calm) * margin, 4),
           "share_gap": round(max(h["share_gap"] for h in calm) * margin, 4), "offset_guard_db": OFFSET_GUARD_DB}
    return thr


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def healthy_cases(cells, n: int, seed0: int, stream: str, share_none: float = 0.4) -> List[dict]:
    """Healthy networks: no fault; ~60% have a twin mismatch (a constant dB offset) on one or two cells, possibly the
    cell being judged. `stream` seeds the mismatch draws so calibration and validation never share configurations."""
    rng = random.Random(stream)
    out = []
    for i in range(n):
        bias = None if rng.random() < share_none else {
            c: round(rng.choice([-1, 1]) * rng.uniform(2.0, 4.5), 1) for c in rng.sample(list(cells), rng.choice([1, 2]))}
        out.append({"fault": None, "bias": bias, "seed": seed0 + i})
    return out


def fault_cases(cells, fault, n: int, seed0: int, stream: str) -> List[dict]:
    """The finding's fault, on fresh surveys, half with a twin mismatch somewhere."""
    return [{**c, "fault": fault} for c in healthy_cases(cells, n, seed0, stream, share_none=0.5)]


def survey_features(twin, inventory: dict, cases: List[dict], cell: int, n_ue: int = 1200) -> List[Dict[str, float]]:
    from .world import World
    return [features(twin, inventory, World(twin, inventory, c.get("fault"), c.get("bias")).field_observation(n_ue, c["seed"]))[cell]
            for c in cases]


def fires(thresholds: Dict[str, float], feat: Dict[str, float]) -> List[str]:
    """Names of the features over their threshold (empty = quiet). The pattern feature always counts. The 'heard less /
    serves less' features only count while the cell's constant offset is small: a large offset with an ordinary pattern is
    what a mis-calibrated twin looks like, and that is a calibration matter, not a fault."""
    over = ["shape_rmse"] if feat["shape_rmse"] > thresholds["shape_rmse"] else []
    if feat["offset_abs"] < thresholds["offset_guard_db"]:
        over += [f for f in ("heard_gap", "share_gap") if feat[f] > thresholds[f]]
    return over


def twin_offset_suspected(thresholds: Dict[str, float], feat: Dict[str, float]) -> bool:
    """Large constant offset, ordinary pattern: calibrate the twin before suspecting the network."""
    return feat["offset_abs"] >= thresholds["offset_guard_db"] and feat["shape_rmse"] <= thresholds["shape_rmse"]


def fault_from_card(card: dict):
    from .faults import Fault
    c = card["claim"]
    return Fault(c["kind"], c["cell"], float(c["value"]) if c["kind"] != "cell_off" else 0.0)


def build_rule(twin, inventory: dict, card: dict, allow_unreviewed: bool = False, margin: float = MARGIN,
               n_ue: int = 1200) -> dict:
    """Derive, validate and gate a detector rule from one finding card."""
    if card.get("type", "fault_finding") != "fault_finding" or card.get("verdict") != "confirmed":
        raise ValueError("detector rules come from confirmed fault findings only")
    approved = card.get("engineer_status") == "accept"
    if not approved and not allow_unreviewed:
        raise PermissionError(f"finding {card['id']} has not been accepted by an engineer "
                              "(pass allow_unreviewed to build a rule from a confirmed but unreviewed finding)")
    fault, cell, cells = fault_from_card(card), card["claim"]["cell"], twin.cells
    cal = survey_features(twin, inventory, healthy_cases(cells, N_CALIBRATION, SEEDS["calibration"], f"cal-{cell}"), cell, n_ue)
    thresholds = calibrate(cal, margin)

    val_h = survey_features(twin, inventory, healthy_cases(cells, N_VALIDATION_HEALTHY, SEEDS["validation_healthy"],
                                                           f"val-h-{cell}"), cell, n_ue)
    val_f = survey_features(twin, inventory, fault_cases(cells, fault, N_VALIDATION_FAULT, SEEDS["validation_fault"],
                                                         f"val-f-{cell}"), cell, n_ue)
    fa = sum(bool(fires(thresholds, f)) for f in val_h)
    hit = sum(bool(fires(thresholds, f)) for f in val_f)
    far, det = fa / len(val_h), hit / len(val_f)
    ok = far <= MAX_FALSE_ALARM and det >= MIN_DETECTION
    far_hi = wilson(fa, len(val_h))[1]
    why = (("meets the gate" + (f" (borderline: the false-alarm rate could be as high as {far_hi:.0%})"
                                if far_hi > 0.10 else "")) if ok else
           f"false-alarm rate {far:.0%} (limit {MAX_FALSE_ALARM:.0%})" if far > MAX_FALSE_ALARM else
           f"detection {det:.0%} (needs {MIN_DETECTION:.0%}): this fault is too close to a healthy cell to screen for")
    return {
        "id": f"detector-{cell}-{fault.kind}-{card['id'][-10:]}", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cell": cell, "resembles": {"finding": card["id"], "kind": fault.kind, "estimate": card["claim"]["estimate"]},
        "approved_by": (card.get("engineer_verdicts") or [{}])[-1].get("by") if approved else None,
        "rule": (f"fires when shape_rmse > {thresholds['shape_rmse']}, or (while the cell's constant offset is below "
                 f"{thresholds['offset_guard_db']} dB) heard_gap > {thresholds['heard_gap']} or "
                 f"share_gap > {thresholds['share_gap']}"),
        "thresholds": thresholds, "status": "deployable" if ok else "rejected", "gate_reason": why,
        "borderline": bool(ok and far_hi > 0.10),
        "calibration": {"n_healthy_surveys": len(cal), "margin": margin, "seeds_from": SEEDS["calibration"],
                        "largest_healthy": {f: round(max(c[f] for c in cal), 4) for f in FEATURES},
                        "offset_guard_db": OFFSET_GUARD_DB},
        "validation": {
            "false_alarm": {"fired": fa, "of": len(val_h), "rate": far, "ci95": [round(x, 4) for x in wilson(fa, len(val_h))],
                            "seeds_from": SEEDS["validation_healthy"]},
            "detection": {"fired": hit, "of": len(val_f), "rate": det, "ci95": [round(x, 4) for x in wilson(hit, len(val_f))],
                          "seeds_from": SEEDS["validation_fault"]},
            "gate": {"max_false_alarm": MAX_FALSE_ALARM, "min_detection": MIN_DETECTION}},
        "what_it_is": ("an early-warning trigger: it says this cell is anomalous and resembles the source finding; "
                       "it does not diagnose. Run an investigation to find the cause."),
        "limits": "uses the twin as reference (needs the twin and the documented inventory); validated on simulated "
                  "surveys only",
    }


def rules_from_store(store, twin, inventory: dict, allow_unreviewed: bool = False) -> List[dict]:
    """One rule per distinct (cell, kind, value) among engineer-accepted confirmed fault findings."""
    seen, out = set(), []
    for card in store.list():
        if card.get("type", "fault_finding") != "fault_finding" or card.get("verdict") != "confirmed":
            continue
        if card.get("engineer_status") != "accept" and not allow_unreviewed:
            continue
        key = (card["claim"]["cell"], card["claim"]["kind"], card["claim"]["value"])
        if key in seen:
            continue
        seen.add(key)
        out.append(build_rule(twin, inventory, card, allow_unreviewed))
    return out


def scan(rules: List[dict], twin, inventory: dict, obs) -> List[dict]:
    """Run deployable rules on one survey. Each result says whether the rule fired and why."""
    feats = features(twin, inventory, obs)
    out = []
    for r in rules:
        if r["status"] != "deployable":
            continue
        f = feats[r["cell"]]
        over = fires(r["thresholds"], f)
        out.append({"rule": r["id"], "cell": r["cell"], "fired": bool(over), "features_over_threshold": over,
                    "twin_offset_suspected": twin_offset_suspected(r["thresholds"], f) and not over,
                    "values": {k: round(f[k], 3) for k in FEATURES + ("offset_abs",)}, "resembles": r["resembles"]})
    return out


def network_report(rules: List[dict], twin, inventory: dict, n: int = 200, seed0: int = 8000, n_ue: int = 1200) -> dict:
    """How often a healthy network triggers ANY deployable rule, on surveys no rule saw: the cost of running them all."""
    from .world import World
    live = [r for r in rules if r["status"] == "deployable"]
    cases = healthy_cases(twin.cells, n, seed0, "network-report-v2")
    triggered = 0
    per_rule = {r["id"]: 0 for r in live}
    for c in cases:
        obs = World(twin, inventory, None, c["bias"]).field_observation(n_ue, c["seed"])
        feats = features(twin, inventory, obs)
        fired = [r["id"] for r in live if fires(r["thresholds"], feats[r["cell"]])]
        triggered += bool(fired)
        for i in fired:
            per_rule[i] += 1
    return {"surveys": n, "rules": len(live), "surveys_that_triggered_any_rule": triggered,
            "network_false_alarm_rate": triggered / n, "ci95": [round(x, 4) for x in wilson(triggered, n)],
            "per_rule_false_alarms": per_rule}


def save_rules(rules: List[dict], path=RULES_FILE) -> None:
    path.write_text(json.dumps(rules, indent=2))


def load_rules(path=RULES_FILE) -> List[dict]:
    return json.loads(path.read_text()) if path.exists() else []
