"""Reusable artefacts a finding yields: something deployable, not just a report.

  early_warning_indicator  one telemetry feature on the faulted cell with a threshold, validated on fresh lab windows
                           (false alarms on healthy windows, detection on windows with the fault re-injected), gated
  test_strategy            the recommended test order for the fault class, read from the finding graph
  fine_tuning_sample       an (observation -> finding) training example for the research-tuned language model
  detector_rule            survey-based screen (detector.py), built once an engineer accepts the finding
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config
from .faults import Fault

CORPUS = config.ROOT / "finetune" / "data" / "findings_corpus.jsonl"
EWI_MAX_FALSE_ALARM = 0.05
EWI_MIN_DETECTION = 0.80
FLOOR = {"ue_share": 0.01, "mean_rsrp_dbm": 0.5, "mean_sinr_db": 0.5, "prb_util_pct": 1.0}

SYSTEM = ("You are a RAN research assistant. From a cross-modal observation of a mobile network (KPI windows, decoded O-RAN "
          "interface events, alarms and a drive-test survey), name the most likely misconfigured cell and fault, and the "
          "evidence that supports it. Answer with JSON only.")


def early_warning_indicator(card: dict, observation: dict, replicator, windows: int = 24, calib: int = 12) -> dict:
    """Calibrate on one set of fresh lab windows, validate on another. Calibration: for each telemetry feature of the cell,
    the threshold is the most extreme healthy value seen (baseline + healthy calibration windows) in the direction the
    fault moves it; the feature that then catches the most calibration fault windows is chosen. Validation runs on
    different seeds, so the reported rates are not the ones the threshold was fitted on."""
    c = card["claim"]
    cell = int(c["cell"])
    j = replicator.cells.index(cell)
    lab = replicator.lab_exec.lab
    fault = Fault(c["kind"], cell, float(c.get("value") or 0.0))
    faulted = fault.apply(replicator.inventory)

    def table(settings, seed0, n):
        return replicator._tables(lab.run_campaign(n, seed0, f"ewi-{seed0}", settings=settings, profile="busy_hour"))

    def col(t, f):
        return np.where(np.isnan(t[f][:, j]), -1e9, t[f][:, j])   # no UE served: a level feature is as low as it gets

    base, h_cal, f_cal = replicator.baseline("busy_hour"), table(replicator.inventory, 8100, calib), table(faulted, 7100, calib)
    best = None
    for f in base:
        healthy = np.concatenate([col(base, f), col(h_cal, f)])
        fv = col(f_cal, f)
        direction = 1 if fv.mean() > healthy.mean() else -1
        thr = float(healthy.max() if direction > 0 else healthy.min())
        if (direction > 0 and fv.min() > thr) or (direction < 0 and fv.max() < thr):   # fully separable: split the gap
            thr = (thr + float(fv.min() if direction > 0 else fv.max())) / 2
        det = float(np.mean(fv > thr) if direction > 0 else np.mean(fv < thr))
        if best is None or det > best[3]:
            best = (f, direction, thr, det)
    feature, direction, thr, _ = best

    def fires(settings, seed0) -> List[bool]:
        v = col(table(settings, seed0, windows), feature)
        return list((v > thr) if direction > 0 else (v < thr))

    fa = fires(replicator.inventory, 8000)
    det = fires(faulted, 7000)
    fa_rate, det_rate = float(np.mean(fa)), float(np.mean(det))
    ok = fa_rate <= EWI_MAX_FALSE_ALARM and det_rate >= EWI_MIN_DETECTION
    return {"id": f"{card['id']}/early-warning", "type": "early_warning_indicator", "cell": cell, "feature": feature,
            "rule": f"cell {cell} {feature} {'>' if direction > 0 else '<'} {thr:.3g} in a one-minute window",
            "threshold": round(thr, 4), "direction": "above" if direction > 0 else "below",
            "validation": {"false_alarm": {"fired": int(sum(fa)), "of": len(fa)}, "detection": {"fired": int(sum(det)), "of": len(det)}},
            "status": "deployable" if ok else "rejected",
            "gate": f"false alarms <= {EWI_MAX_FALSE_ALARM:.0%} and detection >= {EWI_MIN_DETECTION:.0%} on fresh lab windows"}


def fine_tuning_sample(card: dict, observation: dict, suspicion: Dict[int, float], symptom: str) -> dict:
    from .observer import Observer
    c = card["claim"]
    user = (f"Cross-modal observation: {Observer.describe(observation)}. Survey symptom: {symptom}. "
            f"Per-cell survey suspicion: {json.dumps({int(k): round(float(v), 2) for k, v in suspicion.items()})}. "
            "Which cell and fault?")
    target = {"cell": int(c["cell"]), "fault_kind": c["kind"], "estimate": c.get("estimate"),
              "evidence_modalities": next((p["flagged"] for p in observation["patterns"] if p["cell"] == c["cell"]), []),
              "tests_passed": [k["name"] for k in card.get("checks", []) if k["passed"]]}
    return {"id": f"{card['id']}/sample", "type": "fine_tuning_sample", "card_id": card["id"], "task": "finding_explanation",
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                         {"role": "assistant", "content": json.dumps(target)}]}


def append_to_corpus(sample: dict, path: Path = None) -> int:
    path = Path(path or CORPUS)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []
    existing = [e for e in existing if e["id"] != sample["id"]] + [sample]
    path.write_text("".join(json.dumps(e) + "\n" for e in existing), encoding="utf-8")
    return len(existing)


def load_corpus(path: Path = None) -> List[dict]:
    path = Path(path or CORPUS)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []
