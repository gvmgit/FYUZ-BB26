"""Scientist, Experimenter and Critic.

Each agent has one job and talks to MAVERIC only through the gateway's `run_experiment` tool:
  Scientist    - reads the symptom, ranks suspects, writes testable claims.
  Experimenter - tests a claim by changing exactly one setting at a time.
  Critic       - tries to disprove the result; nothing is accepted without a control run and a held-out check.
On AWS each becomes a Strands agent on AgentCore with these same methods exposed as its tools.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config, prereg, stats
from .reasoner import Reasoner, TemplateReasoner

MIN_IMPROVEMENT = 0.5   # score units; CI lower bound must clear this
MIN_WIN_RATE = 0.8
# How much the claim must beat the "twin is just biased here" explanation by (CI lower bound, score units). Chosen from
# 180 mismatch-only cases (largest margin seen 0.061) and 12 real tilt faults; see docs/FINE_TUNING.md. Far below
# MIN_IMPROVEMENT because a constant offset already explains most of a small setting change.
NULL_MIN_GAP = 0.10
# The claim must also hold on the second instrument (the simulator's own ground-truth tables): CI lower bound, score units.
REPLICATION_MIN_GAP = 0.25
TILT_COARSE = (4, 6, 8, 12, 14, 16)
AZIMUTH_COARSE_OFFSETS = tuple(o for o in range(-150, 181, 30) if o != 0)
OFFSET_COARSE = (-6, -4, -2, 2, 4, 6)   # dB, for the twin-bias null hypothesis


@dataclass
class Claim:
    id: str
    kind: str          # tilt | azimuth | cell_off
    cell: int
    text: str
    registration: Optional[dict] = None   # pre-registration (prereg.register), sealed before any experiment


@dataclass
class Evidence:
    claim: Claim
    sweep: List[dict]                  # [{label, value, per_seed, mean, run_id}]
    baseline_per_seed: List[float]
    best: dict = None
    effect: dict = None

    def finalize(self):
        self.best = min(self.sweep, key=lambda s: s["mean"])
        self.effect = stats.paired_effect(self.baseline_per_seed, self.best["per_seed"])
        return self


@dataclass
class Review:
    status: str                        # needs_control | needs_null | needs_holdout | needs_replication | accepted | rejected
    checks: List[dict] = field(default_factory=list)
    reason: str = ""


def material(effect: dict) -> bool:
    return effect["ci95"][0] > MIN_IMPROVEMENT and effect["win_rate"] >= MIN_WIN_RATE


def _score_list(result: dict) -> List[float]:
    return [p["score"] for p in result["per_seed"]]


class Scientist:
    name = "scientist"

    def __init__(self, gateway, inventory: dict, reasoner: Reasoner = None, audit=None, priors=None):
        self.gw, self.inventory = gateway, inventory
        self.reasoner = reasoner or TemplateReasoner()
        self.audit = audit          # pre-registrations are written here
        self.priors = priors        # engineer-verdict priors (verdicts.Priors); reorder only, never skip
        self.suspicion: Dict[int, float] = {}

    def observe(self, seeds: List[int]) -> dict:
        base = self.gw.invoke("run_experiment", {
            "config": {"name": "baseline (inventory settings)", "cells": {}, "diagnostics": True, "render": True},
            "seeds": seeds}, agent=self.name)
        diag = base["diagnostics"]
        n_cells = len(diag["cells"])
        involvement = {c: 0.0 for c in diag["cells"]}
        for pair, n in diag["confusion"].items():
            a, b = pair.replace("obs", "").split("->pred")
            for c in (int(a), int(b)):
                involvement[c] += n
        total = sum(v for v in involvement.values()) or 1.0
        for c, d in diag["cells"].items():
            share_gap = abs(d["obs_share"] - d["pred_share"])
            heard_gap = abs(d["obs_heard"] - d["pred_heard"])
            resid = d["mean_abs_resid_db"] or 0.0
            self.suspicion[c] = round(resid + 5 * heard_gap + 5 * share_gap + 3 * involvement[c] / total, 3)
        zones = diag["weak_coverage_by_zone"]
        zname, z = max(zones.items(), key=lambda kv: abs(kv[1]["field_mean_rsrp"] - kv[1]["expected_mean_rsrp"]))
        symptom_text = (f"zone {zname}: field best-server RSRP averages {z['field_mean_rsrp']} dBm where the "
                        f"inventory predicts {z['expected_mean_rsrp']} dBm; {z['field_weak_pct']}% of UEs have weak "
                        f"coverage (< {config.WEAK_COVERAGE_DBM:.0f} dBm) vs {z['expected_weak_pct']}% expected")
        return {"baseline": base, "diagnostics": diag, "suspicion": self.suspicion, "n_cells": n_cells,
                "worst_zone": zname, "symptom_text": symptom_text}

    def propose(self, symptom: dict, top_cells: int = 2) -> List[Claim]:
        ranked = sorted(self.suspicion, key=self.suspicion.get, reverse=True)[:top_cells]
        claims = []
        for cell in ranked:
            d = symptom["diagnostics"]["cells"][cell]
            why = (f"UEs hear it {d['obs_heard']:.0%} of the time vs {d['pred_heard']:.0%} predicted from inventory, "
                   f"its RSRP is off by {d['mean_abs_resid_db'] or 0:.1f} dB on average, and it serves "
                   f"{d['obs_share']:.0%} vs {d['pred_share']:.0%} (suspicion {self.suspicion[cell]})")
            kinds = ["tilt", "azimuth", "cell_off"]
            if d["pred_heard"] > 0.1 and d["obs_heard"] < 0.3 * d["pred_heard"]:
                kinds = ["cell_off", "tilt", "azimuth"]   # barely heard at all: check it is down first
            what = {"tilt": f"actual tilt differs from inventory ({self.inventory[cell]['tilt']:.0f} deg)",
                    "azimuth": f"actual azimuth differs from inventory ({self.inventory[cell]['azimuth']:.0f} deg)",
                    "cell_off": "cell is not radiating although the inventory says it is on"}
            if self.priors is not None:   # engineers' past verdicts reorder the kinds; every kind is still tested
                kinds = self.priors.order_kinds(kinds)
            for kind in kinds:
                claims.append(self.register(Claim(
                    f"{kind}-{cell}", kind, cell,
                    self.reasoner.explain("claim", {"cell": cell, "what": what[kind], "why": why})),
                    symptom["baseline"]["mean"]["score"]))
        return claims

    def register(self, claim: Claim, baseline_score: float) -> Claim:
        """Seal the hypothesis, its prediction and its null outcome before any experiment on it."""
        controls = ["control: same sweep on an unsuspected cell", "null: a constant twin offset on this cell",
                    "replication on fresh seeds"]
        controls.append("replication on a second instrument (simulator ground truth)" if claim.kind == "tilt"
                        else "second instrument: not available for this kind (simulator tables cover tilt only)")
        claim.registration = prereg.register({
            "id": claim.id, "kind": claim.kind, "cell": claim.cell, "hypothesis": claim.text,
            "design": {"one_setting_at_a_time": True, "controls": controls},
            "predicted_effect": {
                "metric": "field-vs-twin discrepancy score (lower is closer to the field)",
                "baseline_score": round(baseline_score, 2),
                "min_reduction": MIN_IMPROVEMENT, "beats_twin_offset_by": NULL_MIN_GAP,
                "second_instrument_gap": REPLICATION_MIN_GAP, "min_win_rate": MIN_WIN_RATE},
            "null_outcome": ("no one-setting change on this cell lowers the score by the minimum, or a constant dB "
                             "offset explains the field as well (then the twin, not the network, is wrong here)")},
            self.audit)
        return claim

    def control_cell(self, claim: Claim) -> int:
        others = [c for c in self.suspicion if c != claim.cell]
        return min(others, key=self.suspicion.get)


class Experimenter:
    name = "experimenter"

    def __init__(self, gateway, inventory: dict):
        self.gw, self.inventory = gateway, inventory
        self.experiments = 0

    def _run(self, label: str, cell: int, change: dict, seeds: List[int], render: bool = False) -> dict:
        res = self.gw.invoke("run_experiment", {
            "config": {"name": label, "cells": {str(cell): change}, "render": render}, "seeds": seeds},
            agent=self.name)
        self.experiments += 1
        return res

    def _candidates(self, kind: str, cell: int, stage: str, best_value: Optional[float] = None):
        inv = self.inventory[cell]
        if kind == "tilt":
            if stage == "coarse":
                vals = [v for v in TILT_COARSE if v != inv["tilt"]]
            else:
                vals = [v for v in (best_value - 1, best_value + 1) if 4 <= v <= 16 and v != inv["tilt"]]
            return [(f"cell {cell} tilt {v:g}", v, {"tilt": float(v)}) for v in vals]
        if kind == "azimuth":
            if stage == "coarse":
                vals = [(inv["azimuth"] + o) % 360 for o in AZIMUTH_COARSE_OFFSETS]
            else:
                vals = [(best_value + o) % 360 for o in (-20, -10, 10, 20)]
            return [(f"cell {cell} azimuth {v:g}", v, {"azimuth": float(v)}) for v in vals]
        if kind == "offset":   # null hypothesis: the twin is biased on this cell by a constant number of dB
            if stage == "coarse":
                vals = OFFSET_COARSE
            else:
                vals = [v for v in (best_value - 1, best_value + 1) if -8 <= v <= 8 and v != 0]
            return [(f"cell {cell} twin offset {v:+g} dB", v, {"offset_db": float(v)}) for v in vals]
        return [(f"cell {cell} switched off", 0.0, {"enabled": False})]

    def test(self, claim: Claim, seeds: List[int], baseline_per_seed: List[float], emit=lambda *_: None) -> Evidence:
        if not prereg.verify(claim.registration):
            raise ValueError(f"claim {claim.id} is not pre-registered (or its registration was altered): "
                             "register the hypothesis before running any experiment on it")
        sweep = []

        def run_all(stage, best_value=None):
            for label, value, change in self._candidates(claim.kind, claim.cell, stage, best_value):
                r = self._run(label, claim.cell, change, seeds)
                sweep.append({"label": label, "value": value, "per_seed": _score_list(r),
                              "mean": r["mean"]["score"], "run_id": r["run_id"]})
                emit(f"    {label:<28} score {r['mean']['score']:.2f}")

        run_all("coarse")
        if claim.kind in ("tilt", "azimuth", "offset"):
            coarse_best = min(sweep, key=lambda s: s["mean"])
            emit(f"    refining around {coarse_best['value']:g}")
            run_all("fine", coarse_best["value"])
        return Evidence(claim, sweep, baseline_per_seed).finalize()

    def control(self, claim: Claim, control_cell: int, seeds: List[int], baseline_per_seed: List[float],
                emit=lambda *_: None) -> Evidence:
        """Same one-setting sweep on a cell the Scientist does not suspect (coarse only)."""
        sweep = []
        for label, value, change in self._candidates(claim.kind, control_cell, "coarse"):
            r = self._run("CONTROL " + label, control_cell, change, seeds)
            sweep.append({"label": label, "value": value, "per_seed": _score_list(r),
                          "mean": r["mean"]["score"], "run_id": r["run_id"]})
        ev = Evidence(Claim(claim.id + "-control", claim.kind, control_cell, "control"), sweep,
                      baseline_per_seed).finalize()
        emit(f"    control cell {control_cell}: best {ev.best['label']} score {ev.best['mean']:.2f}")
        return ev

    def null(self, claim: Claim, seeds: List[int], baseline_per_seed: List[float],
             emit=lambda *_: None) -> Evidence:
        """Null hypothesis: nothing is wrong with the network, the twin is simply biased on this cell. Sweep a
        constant dB offset on the claimed cell; the claim must beat the best offset to be accepted."""
        null_claim = Claim(claim.id + "-null", "offset", claim.cell, "the twin is biased on this cell",
                           registration=claim.registration)     # the null is part of the parent's registered design
        ev = self.test(null_claim, seeds, baseline_per_seed, emit)
        emit(f"    null hypothesis (twin biased on cell {claim.cell}): best {ev.best['label']} score {ev.best['mean']:.2f}")
        return ev

    def replicate(self, claim: Claim, best: dict, seeds: List[int], emit=lambda *_: None) -> Optional[dict]:
        """Replicate on a second instrument: MAVERIC's simulator ground-truth tables instead of the twin. Tilt only;
        returns None for other kinds. Same field log, same seeds as a paired baseline vs claim comparison."""
        if claim.kind != "tilt":
            return None
        base = self.gw.invoke("run_experiment", {"config": {"name": "second-instrument baseline", "cells": {},
                                                            "instrument": "simulator"}, "seeds": seeds}, agent=self.name)
        fix = self.gw.invoke("run_experiment", {"config": {
            "name": f"second instrument: {best['label']}", "instrument": "simulator",
            "cells": {str(claim.cell): {"tilt": float(best["value"])}}}, "seeds": seeds}, agent=self.name)
        self.experiments += 2
        effect = stats.paired_effect(_score_list(base), _score_list(fix))
        emit(f"    second instrument (simulator tables): improvement {effect['mean_improvement']:.2f} "
             f"(CI {effect['ci95'][0]:.2f}..{effect['ci95'][1]:.2f})")
        return effect

    def holdout(self, claim: Claim, best: dict, seeds: List[int], emit=lambda *_: None) -> dict:
        """Re-test the winning setting against inventory on fresh seeds."""
        base = self.gw.invoke("run_experiment", {"config": {"name": "holdout baseline", "cells": {}},
                                                 "seeds": seeds}, agent=self.name)
        self.experiments += 1
        change = self._candidates(claim.kind, claim.cell, "coarse")[0][2]
        change = {k: (best["value"] if k in ("tilt", "azimuth") else v) for k, v in change.items()}
        fix = self._run(f"HOLDOUT {best['label']}", claim.cell, change, seeds)
        effect = stats.paired_effect(_score_list(base), _score_list(fix))
        emit(f"    holdout: improvement {effect['mean_improvement']:.2f} (CI {effect['ci95'][0]:.2f}.."
             f"{effect['ci95'][1]:.2f})")
        return effect


class Critic:
    name = "critic"

    def review(self, ev: Evidence, control: Optional[Evidence] = None, holdout: Optional[dict] = None,
               null: Optional[Evidence] = None, replication: Optional[dict] = None) -> Review:
        checks = [self._check("improves on inventory", ev.effect,
                              f"best '{ev.best['label']}' cuts score by {ev.effect['mean_improvement']:.2f} "
                              f"(CI {ev.effect['ci95'][0]:.2f}..{ev.effect['ci95'][1]:.2f}, "
                              f"wins {ev.effect['win_rate']:.0%} of seeds)", want=True)]
        if not material(ev.effect):
            return Review("rejected", checks, "no material improvement over inventory settings")
        if control is None:
            return Review("needs_control", checks, "run the same sweep on an unsuspected cell before accepting")
        checks.append(self._check("control cell shows no comparable gain", control.effect,
                                  f"control best '{control.best['label']}' improvement "
                                  f"{control.effect['mean_improvement']:.2f} "
                                  f"(CI {control.effect['ci95'][0]:.2f}..{control.effect['ci95'][1]:.2f})",
                                  want=False))
        if material(control.effect):
            return Review("rejected", checks, "control cell improves too: the effect is not specific to this cell")
        if null is None:
            return Review("needs_null", checks, "test the null hypothesis: the twin is just biased on this cell and "
                                                "nothing is wrong with the network")
        vs_null = stats.paired_effect(null.best["per_seed"], ev.best["per_seed"])   # positive: claim fits better
        beats_null = vs_null["ci95"][0] > NULL_MIN_GAP and vs_null["win_rate"] >= MIN_WIN_RATE
        checks.append({"name": "beats 'the twin is just biased on this cell'", "passed": beats_null,
                       "detail": f"claim cuts the score {vs_null['mean_improvement']:.2f} more than the best constant "
                                 f"offset ('{null.best['label']}'; CI {vs_null['ci95'][0]:.2f}..{vs_null['ci95'][1]:.2f})"})
        if not beats_null:
            return Review("rejected", checks, "a constant twin offset explains the field as well as the setting "
                                              "change: suspect twin calibration, not the network")
        if holdout is None:
            return Review("needs_holdout", checks, "confirm the winning setting on fresh seeds")
        checks.append(self._check("holds on fresh seeds", holdout,
                                  f"improvement {holdout['mean_improvement']:.2f} "
                                  f"(CI {holdout['ci95'][0]:.2f}..{holdout['ci95'][1]:.2f})", want=True))
        if not material(holdout):
            return Review("rejected", checks, "did not replicate on fresh seeds")
        if ev.claim.kind == "tilt":     # the simulator's tables are an independent instrument, for tilt only
            if replication is None:
                return Review("needs_replication", checks, "replicate on a second instrument: the simulator's own "
                                                           "ground-truth tables, not the twin")
            ok = replication["ci95"][0] > REPLICATION_MIN_GAP and replication["win_rate"] >= MIN_WIN_RATE
            checks.append({"name": "replicates on a second instrument (simulator ground truth)", "passed": ok,
                           "detail": f"improvement {replication['mean_improvement']:.2f} on the simulator tables "
                                     f"(CI {replication['ci95'][0]:.2f}..{replication['ci95'][1]:.2f})"})
            if not ok:
                return Review("rejected", checks, "the effect does not appear on the independent simulator tables")
        else:
            checks.append({"name": "replicates on a second instrument", "passed": None,
                           "detail": "not applicable: the simulator tables cover tilt only, so there is no independent "
                                     "instrument for this kind of claim"})
        return Review("accepted", checks, "survived control, the twin-bias null hypothesis, held-out replication"
                                          + (" and the second instrument" if ev.claim.kind == "tilt" else ""))

    @staticmethod
    def _check(name: str, effect: dict, detail: str, want: bool) -> dict:
        return {"name": name, "passed": material(effect) == want, "detail": detail}
