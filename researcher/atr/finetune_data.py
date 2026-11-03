"""Training data for a language model, generated from blind-test investigations.

Every investigation has a ground-truth label (the fault the teammate planted, or the twin mismatch, or nothing), so
examples are verifiable, not opinion. Two tasks, both as chat-format records:

  suspect_ranking  input: what the Scientist sees (symptom, inventory, per-cell diagnostics)
                   output: {"kind": tilt|azimuth|cell_off|twin_miscalibration|none, "cell": n|null}
  claim_triage     input: the same diagnostics plus one claim
                   output: {"survives_critic": bool, "reason": <why a claim fails, or "confirmed">}

Splits are by planted VALUE and survey seed, never by random row, so a test example is not a near copy of a training
example: train tilts {4,7,13,16} / azimuth shifts {+-60,+-120}; validation tilts {6,15} / {180}; test tilts {5,14} /
{+-90}. Cell-held-out splits are not used: the label is a cell id, so a model could not name a cell it never saw.
"""
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from .faults import Fault

SYSTEM = ("You are the Scientist in a telecom network research team. You receive field-measurement diagnostics for a "
          "mobile network and decide what is wrong. Answer with JSON only.")

SPLITS = {
    "train": {"tilts": (4, 7, 13, 16), "az": (-120, -60, 60, 120), "seed0": 1000},
    "val": {"tilts": (6, 15), "az": (180,), "seed0": 2000},
    "test": {"tilts": (5, 14), "az": (-90, 90), "seed0": 3000},
}
REASON_CLASSES = (("no material improvement", "no_gain"), ("control cell improves", "not_specific"),
                  ("twin offset", "twin_bias"), ("did not replicate", "no_replication"))


def build_cases(split: str, cells, repeats: int = 1, mismatch_share: float = 0.5, seed: int = 0) -> List[dict]:
    """Planted-fault cases for a split, plus 'no fault' and 'twin mismatch only' cases."""
    spec = SPLITS[split]
    rng = random.Random(f"{split}-{seed}")
    out = []

    def bias_for(exclude=None):
        if rng.random() > mismatch_share:
            return None
        pool = [c for c in cells if c != exclude]
        return {c: round(rng.choice([-1, 1]) * rng.uniform(2.0, 4.0), 1) for c in rng.sample(pool, rng.choice([1, 2]))}

    for _ in range(repeats):
        for c in cells:
            faults = [Fault("tilt", c, t) for t in spec["tilts"]] + [Fault("cell_off", c)]
            faults += [Fault("azimuth", c, o) for o in spec["az"]]      # value = shift from inventory; resolved later
            for f in faults:
                out.append({"fault": f, "bias": bias_for(exclude=c)})
    n_none = max(2, len(cells) * repeats)
    for i in range(n_none):
        out.append({"fault": None, "bias": None})
        out.append({"fault": None, "bias": {c: round(rng.choice([-1, 1]) * rng.uniform(2.5, 4.5), 1)
                                            for c in rng.sample(list(cells), rng.choice([1, 2]))}})
    for i, case in enumerate(out):
        case.update(split=split, field_seed=spec["seed0"] + i, case_id=f"{split}-{i:04d}")
    return out


def truth_label(case: dict) -> dict:
    f, bias = case["fault"], case["bias"]
    if f is not None:
        return {"kind": f.kind, "cell": f.cell}
    if bias:
        return {"kind": "twin_miscalibration", "cell": max(bias, key=lambda c: abs(bias[c]))}
    return {"kind": "none", "cell": None}


def diagnostics_view(symptom: dict, inventory: dict) -> dict:
    """What the Scientist sees: raw diagnostics, deliberately without the rule-based suspicion scores."""
    d = symptom["diagnostics"]
    return {"symptom": symptom["symptom_text"],
            "inventory": {str(c): {"tilt": s["tilt"], "azimuth": s["azimuth"]} for c, s in inventory.items()},
            "cells": {str(c): v for c, v in d["cells"].items()},
            "serving_confusion": d["confusion"],
            "weak_coverage_by_zone": d["weak_coverage_by_zone"]}


def reason_class(card: dict) -> str:
    if card["verdict"] == "confirmed":
        return "confirmed"
    if card["verdict"].startswith("ruled out") or card["verdict"].startswith("alternative"):
        return "outcompeted"
    reason = (card.get("review") or {}).get("reason", "")
    return next((cls for key, cls in REASON_CLASSES if key in reason), "other")


def system_outcome(out: dict) -> dict:
    head = out["headline"]
    if head:
        return {"kind": head["claim"]["kind"], "cell": head["claim"]["cell"]}
    alerts = [c["twin_calibration_alert"] for c in out["cards"] if c.get("twin_calibration_alert")]
    if alerts:
        return {"kind": "twin_miscalibration", "cell": alerts[0]["cell"]}
    return {"kind": "none", "cell": None}


def _chat(user: str, assistant: dict) -> list:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant, separators=(",", ":"))}]


def records_for(case: dict, researcher, out: dict, inventory: dict) -> List[dict]:
    """Ranking record + one triage record per claim, with the rule-based pipeline's own answer kept in meta so the
    baseline can be scored offline."""
    view = diagnostics_view(researcher.last_symptom, inventory)
    label = truth_label(case)
    susp = researcher.last_symptom["suspicion"]
    meta = {"case_id": case["case_id"], "split": case["split"], "truth": label,
            "planted_fault": None if case["fault"] is None else
            {"kind": case["fault"].kind, "cell": case["fault"].cell, "value": case["fault"].value},
            "twin_mismatch": case["bias"], "system_outcome": system_outcome(out),
            "experiments": out["experiments"], "rule_top_suspect_cell": max(susp, key=susp.get)}
    recs = [{"task": "suspect_ranking", "messages": _chat(json.dumps(view, separators=(",", ":")), label),
             "meta": meta}]
    for order, card in enumerate(c for c in out["cards"]):
        claim = {"kind": card["claim"]["kind"], "cell": card["claim"]["cell"]}
        target = {"survives_critic": card["verdict"] == "confirmed", "reason": reason_class(card)}
        recs.append({"task": "claim_triage",
                     "messages": _chat(json.dumps({"diagnostics": view, "claim": claim}, separators=(",", ":")), target),
                     "meta": {**meta, "claim_order": order, "claim": claim, "label": target}})
    return recs


def records_from_verdicts(store) -> List[dict]:
    """Training records from real findings an engineer accepted or rejected: the engineer's call is the label. These are
    the 'accepted findings corpus' the proposal fine-tunes on; unlike the generated cases they need no planted truth."""
    out = []
    for card in store.list():
        status = card.get("engineer_status")
        if status not in ("accept", "reject") or not card.get("diagnostics") or card.get("type") == "network_insight":
            continue
        claim = {"kind": card["claim"]["kind"], "cell": card["claim"]["cell"]}
        target = {"survives_critic": status == "accept",
                  "reason": "engineer_accepted" if status == "accept" else "engineer_rejected"}
        last = card["engineer_verdicts"][-1]
        out.append({"task": "claim_triage",
                    "messages": _chat(json.dumps({"diagnostics": card["diagnostics"], "claim": claim},
                                                 separators=(",", ":")), target),
                    "meta": {"case_id": card["id"], "split": "engineer", "source": "engineer", "claim": claim,
                             "label": target, "by": last["by"], "comment": last.get("comment", ""),
                             "critic_verdict": card["verdict"]}})
    return out


def generate(out_dir, splits=("train", "val", "test"), repeats: int = 1, emit=print, twin=None) -> dict:
    """Run the researcher on every case and write <split>.jsonl plus manifest.json."""
    from . import runtime
    from .topology import default_inventory
    from .twin import RFTwin
    twin = twin or RFTwin.load_or_build()
    inventory = default_inventory()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        cases = build_cases(split, twin.cells, repeats if split == "train" else 1)
        n_rec = 0
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for i, case in enumerate(cases, 1):
                fault = case["fault"]
                if fault is not None and fault.kind == "azimuth":      # stored value is the shift; make it absolute
                    fault = Fault("azimuth", fault.cell, (inventory[fault.cell]["azimuth"] + fault.value) % 360)
                    case = {**case, "fault": fault}
                rt = runtime.build("local", fault, 1200, lambda m: None, twin_bias=case["bias"],
                                   field_seed=case["field_seed"], twin=twin)
                out = rt.researcher.investigate()
                for rec in records_for(case, rt.researcher, out, inventory):
                    f.write(json.dumps(rec) + "\n")
                    n_rec += 1
                if i % 10 == 0:
                    emit(f"  {split}: {i}/{len(cases)} cases")
        emit(f"{split}: {len(cases)} cases, {n_rec} records")
    return write_manifest(out_dir, repeats)


def write_manifest(out_dir, repeats: int = 1) -> dict:
    """Summarise whatever split files exist (so one split can be regenerated without redoing the others)."""
    out_dir = Path(out_dir)
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_repeats": repeats, "splits": {}, "rules": SPLITS}
    for split in SPLITS:
        path = out_dir / f"{split}.jsonl"
        if path.exists():
            recs = load(path)
            manifest["splits"][split] = {"cases": len({r["meta"]["case_id"] for r in recs}), "records": len(recs)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load(path) -> List[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
