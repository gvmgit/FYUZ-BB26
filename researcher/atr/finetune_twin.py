"""Fine-tune the digital twin on field data from approved findings.

Why: a twin is only as good as its fit to the real network. After the researcher explains a field log (the finding
says which settings the network really had), that log becomes trustworthy training data: model inputs are known,
measured RSRP is the label. MAVERIC's per-cell models take it incrementally (`update_trained_gpmodel`).

Safeguards, all enforced in code:
  - only a finding a person approved (`atr.cli approve`) can feed the twin;
  - each cell's update is judged on field rows held out from the update; a cell that does not improve by
    `min_gain_db`, or whose predictions move by more than `max_mean_shift_db`, keeps its old model;
  - every accepted update is a new versioned model file plus a registry entry (who approved, what data, before/after);
    the base twin is never overwritten, so rollback is choosing an earlier file.
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import config
from .audit import FileAudit
from .twin import RFTwin

REGISTRY = config.MODELS / "twin_registry.json"
MIN_ROWS = 30


def corrected_settings(inventory: dict, card: dict) -> dict:
    """The settings the finding says the network really had when the field log was taken."""
    c = card["claim"]
    if c["kind"] == "cell_off":
        raise ValueError("a 'cell off' finding has no signal-strength data from that cell to learn from")
    out = {k: dict(v) for k, v in inventory.items()}
    out[c["cell"]]["azimuth" if c["kind"] == "azimuth" else "tilt"] = float(c["value"])
    return out


def field_fit(twin: RFTwin, obs, settings: dict, rows) -> Dict[int, Optional[float]]:
    """RMSE (dB) of the twin's mean vs the measured RSRP of each cell, over `rows` where the cell was heard."""
    out = {}
    for cid in twin.cells:
        col = obs[f"rsrp_c{cid}"].values[rows]
        idx = obs.grid_idx.values[rows]
        heard = ~np.isnan(col)
        if heard.sum() < MIN_ROWS // 2:
            out[cid] = None
            continue
        mean, _ = twin._cell_grid(cid, settings[cid]["tilt"], settings[cid]["azimuth"])
        out[cid] = float(np.sqrt(np.mean((mean[idx[heard]] - col[heard]) ** 2)))
    return out


def _versions(registry) -> list:
    return json.loads(registry.read_text()) if registry.exists() else []


def latest_model_file(registry=REGISTRY) -> Optional[str]:
    v = _versions(registry)
    return v[-1]["model_file"] if v else None


def finetune(twin: RFTwin, obs, settings: dict, *, approved_by: str, source: str, holdout_frac: float = 0.3,
             seed: int = 0, min_gain_db: float = 0.3, max_mean_shift_db: float = 6.0, dry_run: bool = False,
             registry=REGISTRY, audit=None) -> dict:
    """Update `twin` in place for the cells that pass the guard. Returns a per-cell report."""
    if not approved_by:
        raise PermissionError("twin fine-tuning needs the name of the person who approved the data")
    obs = obs.reset_index(drop=True)
    n = len(obs)
    perm = np.random.default_rng(seed).permutation(n)
    train = np.zeros(n, bool)
    train[perm[: int(n * (1 - holdout_frac))]] = True
    hold = ~train

    before = field_fit(twin, obs, settings, hold)
    cand = twin.clone()
    cells = {}
    for cid in twin.cells:
        s = settings[cid]
        col = obs[f"rsrp_c{cid}"].values
        rows = train & ~np.isnan(col)
        if rows.sum() < MIN_ROWS:
            cells[cid] = {"accepted": False, "reason": f"only {int(rows.sum())} measurements (need {MIN_ROWS})"}
            continue
        df = cand.features(cid, obs.grid_idx.values[rows], s["tilt"], s["azimuth"])
        df["avg_rsrp"] = col[rows]
        cand.update_cell(cid, df)
        cells[cid] = {"train_rows": int(rows.sum())}
    after = field_fit(cand, obs, settings, hold)

    for cid, c in cells.items():
        if "reason" in c:
            continue
        s = settings[cid]
        shift = float(np.mean(np.abs(cand._cell_grid(cid, s["tilt"], s["azimuth"])[0]
                                     - twin._cell_grid(cid, s["tilt"], s["azimuth"])[0])))
        b, a = before[cid], after[cid]
        c.update({"before_db": b, "after_db": a, "gain_db": None if b is None or a is None else b - a,
                  "mean_shift_db": shift})
        if b is None or a is None:
            c.update(accepted=False, reason="cell barely heard in the held-out rows")
        elif b - a < min_gain_db:
            c.update(accepted=False, reason=f"held-out error did not improve by {min_gain_db} dB")
        elif shift > max_mean_shift_db:
            c.update(accepted=False, reason=f"update moves predictions {shift:.1f} dB on average (limit {max_mean_shift_db})")
        else:
            c.update(accepted=True, reason="held-out error improved")

    accepted = [cid for cid, c in cells.items() if c.get("accepted")]
    report = {"approved_by": approved_by, "source": source, "n_ue": n, "holdout_ue": int(hold.sum()),
              "cells": cells, "accepted_cells": accepted, "applied": False, "model_file": None}
    if accepted and not dry_run:
        for cid in accepted:
            twin.models[cid] = cand.models[cid]
        twin._cache.clear()
        versions = _versions(registry)
        path = Path(registry).parent / f"twin_v{len(versions) + 1}.pickle"
        twin.save(path)
        report.update(applied=True, model_file=str(path))
        versions.append({"version": len(versions) + 1, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "model_file": str(path), "approved_by": approved_by, "source": source,
                         "accepted_cells": accepted, "cells": {str(k): v for k, v in cells.items()}})
        registry.write_text(json.dumps(versions, indent=2, default=float))
        (audit or FileAudit()).write({"kind": "twin_finetune", "approved_by": approved_by, "source": source,
                                      "accepted_cells": accepted, "model_file": str(path)})
    return report


def apply_finding(twin: RFTwin, card: dict, obs, inventory: dict, **kw) -> dict:
    """Fine-tune on the field log a finding explained. Refused unless a person approved the finding."""
    esc = card.get("escalation") or {}
    if esc.get("status") != "approved":
        raise PermissionError(f"finding {card['id']} is '{esc.get('status', 'unreviewed')}', not approved by a person")
    return finetune(twin, obs, corrected_settings(inventory, card), approved_by=esc["approved_by"],
                    source=f"finding {card['id']}", **kw)


def benchmark(twin: RFTwin, faults, bias: dict, field_seed: int, reality: RFTwin, n_ue: int = 1200) -> dict:
    """Run the researcher on each planted fault with this twin; fresh field survey (`field_seed`). The real network
    is built from `reality` (the original twin, never the one under test) plus the bias."""
    from . import runtime
    hits, floors = 0, []
    for f in faults:
        rt = runtime.build("local", f, n_ue, lambda m: None, twin_bias=bias, field_seed=field_seed, twin=twin,
                           reality_twin=reality)
        out = rt.researcher.investigate()
        head = out["headline"]
        hits += bool(head) and head["claim"]["kind"] == f.kind and head["claim"]["cell"] == f.cell
        floors.append(out["cards"][0]["symptom"]["baseline_score"])
    return {"faults": len(faults), "located": int(hits), "mean_baseline_score": float(np.mean(floors))}


def before_after(fault, bias: dict, approved_by: str, n_ue: int = 1200, bench_faults=None, bench_seed: int = 777,
                 emit=print) -> dict:
    """The whole loop: investigate -> a person approves -> fine-tune -> re-test on a fresh survey."""
    from . import runtime
    from .topology import default_inventory
    quiet = lambda m: None  # noqa: E731
    rt = runtime.build("local", fault, n_ue, quiet, twin_bias=bias)
    out = rt.researcher.investigate()
    head = out["headline"]
    if not head:
        raise RuntimeError("no confirmed finding to learn from")
    emit(f"finding: {head['title']}   (twin mismatch in the field: {bias})")
    approve_card(rt.store, head, approved_by)
    emit(f"approved by {approved_by}")
    twin = rt.world.twin
    base_twin = twin.clone()
    report = apply_finding(twin, head, rt.researcher.runner.obs, default_inventory())
    emit("\nper-cell fit to held-out field measurements (RMSE dB, lower is better)")
    for cid, c in report["cells"].items():
        if "before_db" in c:
            emit(f"  cell {cid}: {c['before_db']:.2f} -> {c['after_db']:.2f}   "
                 f"{'UPDATED' if c['accepted'] else 'kept old model'}  ({c['reason']})")
    emit(f"saved: {report['model_file']}")
    if bench_faults:
        emit(f"\nre-test on a fresh field survey, {len(bench_faults)} planted faults:")
        before = benchmark(base_twin, bench_faults, bias, bench_seed, base_twin, n_ue)
        after = benchmark(twin, bench_faults, bias, bench_seed, base_twin, n_ue)
        emit(f"  base twin : located {before['located']}/{before['faults']}, mean baseline discrepancy {before['mean_baseline_score']:.2f}")
        emit(f"  tuned twin: located {after['located']}/{after['faults']}, mean baseline discrepancy {after['mean_baseline_score']:.2f}")
        report["benchmark"] = {"base": before, "tuned": after}
    return report


def approve_card(store, card: dict, by: str, audit=None) -> dict:
    """The human step (same effect as `atr.cli approve`)."""
    card["escalation"].update({"status": "approved", "approved_by": by, "approved_at": time.time()})
    store.put(card)
    (audit or FileAudit()).write({"kind": "human_approval", "card": card["id"], "by": by})
    return card
