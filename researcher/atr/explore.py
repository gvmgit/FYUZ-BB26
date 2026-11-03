"""Exploration: look for something nobody planted.

Blind tests ask "can the researcher recover a fault we hid?". This asks the other question: with nothing known to be wrong,
can it find a change to the network that is worth making, and survive the same scrutiny? The Scientist scans single-setting
interventions on the twin for the one KPI with headroom (radio-link-failure risk; coverage is already ~99.8% on the
documented network, so it cannot be the objective), pre-registers the best few, and the Critic tries to break them:

  - the gain must survive the twin's own error (each seed also draws a random per-cell dB offset, so the interval
    includes model uncertainty, not just sampling);
  - guardrails: coverage and mean SINR must not get worse;
  - tilt changes must replicate on the simulator's ground-truth tables (a second instrument); azimuth has none;
  - and on fresh seeds under a harsher twin error.

What a surviving finding is: a hypothesis about the network that held on the twin and on the simulator tables. It is NOT
validated on the real network, and the card says so; the proposed next step is a human-approved field trial. The KPIs are
proxies (see experiment.KPI_METRICS).
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import prereg, stats
from .audit import FileAudit

PRIMARY = "rlf_risk_pct"            # lower is better
SCAN_MIN_GAIN = 1.0                 # points of RLF risk a candidate must cut in the scan to be worth testing
GUARD_COVERAGE_PTS = -0.5           # coverage may fall by at most this many points
GUARD_SINR_DB = -0.3                # mean SINR may fall by at most this many dB
TWIN_ERROR_DB = 1.5                 # per-cell twin error drawn for each seed (about its held-out validation error)
STRESS_ERROR_DB = 2.5               # harsher error for the fresh-seed holdout
MIN_GAIN_CI_LOW = 0.5               # the gain's CI lower bound, in points
MIN_WIN_RATE = 0.9
REPLICATION_MIN_GAIN = 0.25         # on the simulator tables, points
HOLDOUT_MIN_GAIN = 0.25


@dataclass
class Hypothesis:
    id: str
    kind: str                    # tilt | azimuth | handover_rule
    cell: Optional[int]          # None for a network-wide handover rule
    from_value: float
    to_value: float
    change: dict
    text: str
    scan: dict = field(default_factory=dict)        # what the scan saw: gain, dsinr, dcoverage
    registration: Optional[dict] = None


@dataclass
class InsightEvidence:
    hypothesis: Hypothesis
    effects: Dict[str, dict]     # each a paired effect (positive = better); network: rlf / sinr / coverage
    base_mean: dict
    treat_mean: dict
    primary: str = "rlf"         # which effect is the headline gain


@dataclass
class InsightReview:
    status: str                  # needs_replication | needs_holdout | accepted | rejected
    checks: List[dict] = field(default_factory=list)
    reason: str = ""


def improvement(base: dict, treat: dict, key: str, sign: int) -> dict:
    """Paired effect, positive when `treat` is better. sign +1: higher is better, -1: lower is better."""
    b, t = [p[key] for p in base["per_seed"]], [p[key] for p in treat["per_seed"]]
    return stats.paired_effect(b, t) if sign < 0 else stats.paired_effect([-x for x in b], [-x for x in t])


def _fmt(e: dict) -> str:
    return (f"{e['mean_improvement']:+.2f} (CI {e['ci95'][0]:.2f}..{e['ci95'][1]:.2f}, wins {e['win_rate']:.0%} of seeds)")


class InsightCritic:
    name = "critic"

    def review(self, ev: InsightEvidence, replication: Optional[dict] = None,
               holdout: Optional[dict] = None) -> InsightReview:
        h, rlf, sinr, cov = ev.hypothesis, ev.effects["rlf"], ev.effects["sinr"], ev.effects["coverage"]
        checks = []
        ok = rlf["ci95"][0] > MIN_GAIN_CI_LOW and rlf["win_rate"] >= MIN_WIN_RATE
        checks.append({"name": f"cuts radio-link-failure risk with the twin allowed its own error ({TWIN_ERROR_DB} dB/cell)",
                       "passed": ok, "detail": f"{_fmt(rlf)} points"})
        if not ok:
            return InsightReview("rejected", checks, "the gain is not distinguishable from zero once the twin's own "
                                                     "error is included")
        guard = sinr["mean_improvement"] >= GUARD_SINR_DB and cov["mean_improvement"] >= GUARD_COVERAGE_PTS
        checks.append({"name": "guardrails: coverage and mean SINR do not get worse", "passed": guard,
                       "detail": f"coverage {cov['mean_improvement']:+.2f} pts (limit {GUARD_COVERAGE_PTS:+g}), "
                                 f"mean SINR {sinr['mean_improvement']:+.2f} dB (limit {GUARD_SINR_DB:+g})"})
        if not guard:
            return InsightReview("rejected", checks, "it trades away coverage or signal quality")
        if h.kind == "tilt":
            if replication is None:
                return InsightReview("needs_replication", checks, "replicate on the simulator's ground-truth tables")
            r_ok = replication["ci95"][0] > REPLICATION_MIN_GAIN and replication["win_rate"] >= 0.8
            checks.append({"name": "replicates on a second instrument (simulator ground truth)", "passed": r_ok,
                           "detail": f"{_fmt(replication)} points on the simulator tables"})
            if not r_ok:
                return InsightReview("rejected", checks, "the gain does not appear on the independent simulator tables")
        else:
            checks.append({"name": "replicates on a second instrument", "passed": None,
                           "detail": "not applicable: the simulator tables cover tilt only"})
        if holdout is None:
            return InsightReview("needs_holdout", checks, f"re-test on fresh seeds under a harsher twin error "
                                                          f"({STRESS_ERROR_DB} dB/cell)")
        h_ok = holdout["ci95"][0] > HOLDOUT_MIN_GAIN and holdout["win_rate"] >= 0.8
        checks.append({"name": f"holds on fresh seeds under a harsher twin error ({STRESS_ERROR_DB} dB/cell)",
                       "passed": h_ok, "detail": f"{_fmt(holdout)} points"})
        if not h_ok:
            return InsightReview("rejected", checks, "it did not hold on fresh seeds under a harsher twin error")
        return InsightReview("accepted", checks, "survived twin error, guardrails, "
                                                 + ("the second instrument, " if h.kind == "tilt" else "")
                                                 + "and fresh-seed replication")


class Explorer:
    def __init__(self, gateway, inventory: dict, store, audit=None, priors=None, emit: Callable = print,
                 seeds=range(1, 9), holdout_seeds=range(101, 109), scan_seeds=(1, 2, 3, 4)):
        self.gw, self.inventory, self.store = gateway, inventory, store
        self.audit = audit or FileAudit()
        self.priors, self.emit = priors, emit
        self.seeds, self.holdout_seeds, self.scan_seeds = list(seeds), list(holdout_seeds), list(scan_seeds)
        self.critic = InsightCritic()
        self.scan_table: List[dict] = []
        self.field_log = None

    # -- tool surface ------------------------------------------------------------------------------------------------
    def _kpi(self, label: str, cells: dict, seeds, sigma: float, instrument: str = "twin", render: bool = False,
             agent: str = "experimenter") -> dict:
        return self.gw.invoke("run_experiment", {"config": {
            "name": label, "objective": "kpi", "metric": PRIMARY, "cells": cells, "twin_error_db": sigma,
            "instrument": instrument, "render": render}, "seeds": list(seeds)}, agent=agent)

    def _candidates(self):
        for cell, inv in self.inventory.items():
            for d in (-4, -2, 2, 4):
                t = inv["tilt"] + d
                if 4 <= t <= 16:
                    yield "tilt", cell, inv["tilt"], t, {"tilt": float(t)}
            for d in (-20, 20):
                az = (inv["azimuth"] + d) % 360
                yield "azimuth", cell, inv["azimuth"], az, {"azimuth": float(az)}

    def scan(self, k: int = 3) -> List[Hypothesis]:
        """Scientist: scan single-setting interventions, shortlist those that help without breaking a guardrail, and
        pre-register the best `k` before any of them is tested."""
        base = self._kpi("scan baseline", {}, self.scan_seeds, 0.0, agent="scientist")
        self.emit(f"[scientist] scanning {sum(1 for _ in self._candidates())} single-setting interventions on the twin; "
                  f"baseline: failure risk {base['mean']['rlf_risk_pct']:.1f}%, coverage {base['mean']['coverage_pct']:.1f}%, "
                  f"mean SINR {base['mean']['mean_sinr_db']:.1f} dB")
        rows = []
        for kind, cell, a, b, change in self._candidates():
            r = self._kpi(f"scan cell {cell} {kind} {a:g}->{b:g}", {str(cell): change}, self.scan_seeds, 0.0,
                          agent="scientist")
            m, m0 = r["mean"], base["mean"]
            rows.append({"kind": kind, "cell": cell, "from": a, "to": b, "change": change,
                         "gain": m0["rlf_risk_pct"] - m["rlf_risk_pct"], "dsinr": m["mean_sinr_db"] - m0["mean_sinr_db"],
                         "dcoverage": m["coverage_pct"] - m0["coverage_pct"], "run_id": r["run_id"]})
        self.scan_table = sorted(rows, key=lambda r: -r["gain"])
        ok = [r for r in rows if r["gain"] >= SCAN_MIN_GAIN and r["dcoverage"] >= GUARD_COVERAGE_PTS
              and r["dsinr"] >= GUARD_SINR_DB]
        blocked = [r for r in rows if r["gain"] >= SCAN_MIN_GAIN and r not in ok]
        for r in blocked:
            self.emit(f"[scientist] set aside (guardrail): cell {r['cell']} {r['kind']} {r['from']:g}->{r['to']:g} cuts "
                      f"risk {r['gain']:.1f} pts but coverage {r['dcoverage']:+.1f} pts, SINR {r['dsinr']:+.1f} dB")
        weight = (lambda kind: self.priors.weight(kind)) if self.priors is not None else (lambda kind: 1.0)
        ok.sort(key=lambda r: -r["gain"] * weight(r["kind"]))
        out = []
        for r in ok[:k]:
            h = Hypothesis(f"insight-{r['kind']}-{r['cell']}-{r['to']:g}", r["kind"], r["cell"], r["from"], r["to"],
                           r["change"], (f"Changing cell {r['cell']} {r['kind']} from {r['from']:g} to {r['to']:g} degrees "
                                         f"will cut radio-link-failure risk by about {r['gain']:.1f} points without "
                                         "lowering coverage or mean SINR."),
                           scan={"gain": round(r["gain"], 2), "dsinr": round(r["dsinr"], 2),
                                 "dcoverage": round(r["dcoverage"], 2)})
            self.register(h)
            out.append(h)
            self.emit(f"[scientist] hypothesis {h.id}: {h.text}")
        if not out:
            self.emit("[scientist] nothing in the scan clears the bar: no hypothesis to test")
        return out

    def register(self, h: Hypothesis) -> Hypothesis:
        h.registration = prereg.register({
            "id": h.id, "kind": h.kind, "cell": h.cell, "hypothesis": h.text,
            "design": {"intervention": f"cell {h.cell} {h.kind} {h.from_value:g} -> {h.to_value:g}",
                       "controls": [f"twin error drawn per seed ({TWIN_ERROR_DB} dB/cell)", "guardrails",
                                    "fresh-seed holdout under a harsher twin error"]
                       + (["replication on the simulator tables"] if h.kind == "tilt" else [])},
            "predicted_effect": {"metric": PRIMARY, "direction": "decrease", "predicted_gain_pts": h.scan["gain"],
                                 "min_gain_ci_low_pts": MIN_GAIN_CI_LOW, "min_win_rate": MIN_WIN_RATE,
                                 "guardrails": {"coverage_pts_floor": GUARD_COVERAGE_PTS, "mean_sinr_db_floor": GUARD_SINR_DB},
                                 "second_instrument_gain_pts": REPLICATION_MIN_GAIN if h.kind == "tilt" else None},
            "null_outcome": ("the gain is indistinguishable from zero once the twin is allowed its own error, or a "
                             "guardrail is breached, or it does not appear on the simulator tables")}, self.audit)
        return h

    def _require(self, h: Hypothesis):
        if not prereg.verify(h.registration):
            raise ValueError(f"hypothesis {h.id} is not pre-registered (or its registration was altered)")

    def test(self, h: Hypothesis) -> InsightEvidence:
        self._require(h)
        self.emit(f"[experimenter] testing {h.id} under twin error ({TWIN_ERROR_DB} dB/cell), {len(self.seeds)} seeds")
        base = self._kpi("baseline (inventory)", {}, self.seeds, TWIN_ERROR_DB)
        treat = self._kpi(f"{h.id}", {str(h.cell): h.change}, self.seeds, TWIN_ERROR_DB)
        ev = InsightEvidence(h, {"rlf": improvement(base, treat, "rlf_risk_pct", -1),
                                 "sinr": improvement(base, treat, "mean_sinr_db", +1),
                                 "coverage": improvement(base, treat, "coverage_pct", +1)},
                             base["mean"], treat["mean"])
        self.emit(f"    failure risk {base['mean']['rlf_risk_pct']:.2f}% -> {treat['mean']['rlf_risk_pct']:.2f}%  "
                  f"(gain {_fmt(ev.effects['rlf'])})")
        return ev

    def replicate(self, h: Hypothesis) -> Optional[dict]:
        if h.kind != "tilt":
            return None
        self.emit(f"[experimenter] replicating {h.id} on the simulator tables")
        base = self._kpi("second-instrument baseline", {}, self.holdout_seeds, 0.0, "simulator")
        treat = self._kpi(f"second instrument: {h.id}", {str(h.cell): h.change}, self.holdout_seeds, 0.0, "simulator")
        eff = improvement(base, treat, "rlf_risk_pct", -1)
        self.emit(f"    simulator tables: gain {_fmt(eff)}")
        return eff

    def holdout(self, h: Hypothesis) -> dict:
        self.emit(f"[experimenter] holdout for {h.id}: fresh seeds, twin error {STRESS_ERROR_DB} dB/cell")
        base = self._kpi("holdout baseline", {}, self.holdout_seeds, STRESS_ERROR_DB)
        treat = self._kpi(f"holdout {h.id}", {str(h.cell): h.change}, self.holdout_seeds, STRESS_ERROR_DB)
        eff = improvement(base, treat, "rlf_risk_pct", -1)
        self.emit(f"    gain {_fmt(eff)}")
        return eff

    def retest(self, card: dict, seeds=range(301, 325), sigma: float = 2.0) -> dict:
        """Follow-up an engineer asked for: more fresh seeds under a harsher twin error, plus the second instrument."""
        c = card["claim"]
        change = {"tilt" if c["kind"] == "tilt" else "azimuth": float(c["value"])}
        base = self._kpi("follow-up baseline", {}, seeds, sigma)
        treat = self._kpi(f"follow-up cell {c['cell']} {c['kind']} {c['value']:g}", {str(c["cell"]): change}, seeds, sigma)
        out = {"seeds": len(list(seeds)), "twin_error_db": sigma, "gain_pts": improvement(base, treat, "rlf_risk_pct", -1)}
        if c["kind"] == "tilt":
            sb = self._kpi("follow-up simulator baseline", {}, seeds, 0.0, "simulator")
            st = self._kpi("follow-up simulator", {str(c["cell"]): change}, seeds, 0.0, "simulator")
            out["second_instrument_gain_pts"] = improvement(sb, st, "rlf_risk_pct", -1)
        g = out["gain_pts"]
        out["still_holds"] = g["ci95"][0] > 0.0 and g["win_rate"] >= 0.8
        return out

    # -- plain pipeline ----------------------------------------------------------------------------------------------
    def explore(self, k: int = 3) -> dict:
        t0, trace = time.time(), []
        base_emit = self.emit
        self.emit = lambda m: (trace.append(m), base_emit(m))
        try:
            results = []
            for h in self.scan(k):
                ev = self.test(h)
                rep = hold = None
                review = self.critic.review(ev)
                while review.status in ("needs_replication", "needs_holdout"):
                    self.emit(f"[critic] {review.reason}")
                    if review.status == "needs_replication":
                        rep = self.replicate(h)
                    else:
                        hold = self.holdout(h)
                    review = self.critic.review(ev, rep, hold)
                self.emit(f"[critic] {h.id}: {review.status.upper()} - {review.reason}")
                results.append((h, ev, review, rep, hold))
            return self.publish(results, trace, t0)
        finally:
            self.emit = base_emit

    # -- what differs between kinds of hypothesis (RuleExplorer overrides these) ---------------------------------------
    metric_label = "Failure-risk reduction (points)"
    sweep_label = "Scanned interventions, by failure-risk reduction (points)"
    caveat = ("Validated on the digital twin and, for tilt, the simulator's tables. NOT validated on the real network: the "
              "KPIs are proxies and the twin has its own error. The next step is a human-approved field trial.")

    def _title(self, h, validated: bool, verdict: str, gain: dict) -> str:
        return (f"Cell {h.cell}: {h.kind} {h.from_value:g} to {h.to_value:g} cuts failure risk "
                f"{gain['mean_improvement']:.1f} pts" if validated else
                f"Cell {h.cell}: {h.kind} {h.from_value:g} to {h.to_value:g} ({verdict})")

    def _claim(self, h) -> dict:
        return {"kind": h.kind, "cell": h.cell, "text": h.text, "value": h.to_value, "from": h.from_value,
                "estimate": f"{h.to_value:g} deg (inventory {h.from_value:g} deg)"}

    def _sweep(self) -> list:
        return [{"label": f"cell {r['cell']} {r['kind']} {r['from']:g}->{r['to']:g}", "mean": -r["gain"],
                 "run_id": r["run_id"], "value": r["to"]} for r in self.scan_table[:8]]

    def _kpi_effects(self, ev) -> dict:
        return {"failure_risk_pts": ev.effects["rlf"], "mean_sinr_db": ev.effects["sinr"],
                "coverage_pts": ev.effects["coverage"]}

    def _maps(self, head) -> dict:
        h = head[0]
        return {"inventory": self._kpi("inventory", {}, self.seeds[:1], 0.0, render=True)["heatmap"],
                "best_fix": self._kpi(f"{h.id}", {str(h.cell): h.change}, self.seeds[:1], 0.0, render=True)["heatmap"]}

    # -- publication -------------------------------------------------------------------------------------------------
    def publish(self, results, trace: List[str], t0: float) -> dict:
        from .anf import AgentMesh
        for h, *_ in results:
            if not prereg.verify(h.registration):
                raise ValueError(f"{h.id}: pre-registration missing or altered; refusing to publish")
        accepted = sorted((r for r in results if r[2].status == "accepted"),
                          key=lambda r: -r[1].effects[r[1].primary]["mean_improvement"])
        head = accepted[0] if accepted else None
        maps = self._maps(head) if head else {}
        total = self.gw.count
        cards = []
        for h, ev, review, rep, hold in results:
            is_head = head is not None and h is head[0]
            validated = review.status == "accepted"
            verdict = ("validated on the twin" + (" and simulator tables" if h.kind == "tilt" else "")) if validated \
                else "refuted" if review.status == "rejected" else "inconclusive"
            gain = ev.effects[ev.primary]
            card = {
                "id": f"{time.strftime('%Y%m%d-%H%M%S')}-{h.id}-{uuid.uuid4().hex[:4]}", "created": time.time(),
                "type": "network_insight", "headline": is_head, "verdict": verdict,
                "title": self._title(h, validated, verdict, gain),
                "claim": self._claim(h), "metric_label": self.metric_label, "sweep_label": self.sweep_label,
                "evidence": {"best": h.id, "effect": gain, "sweep": self._sweep()},
                "kpi_effects": self._kpi_effects(ev),
                "kpi_means": {"baseline": ev.base_mean, "with_change": ev.treat_mean},
                "preregistration": {
                    "hash": h.registration["hash"], "registered_at": h.registration["registered_at"],
                    "verified": prereg.verify(h.registration), "predicted": h.registration["statement"]["predicted_effect"],
                    "null_outcome": h.registration["statement"]["null_outcome"],
                    "design": h.registration["statement"]["design"], "prediction_met": validated},
                "second_instrument": ({"instrument": "simulator ground-truth tables", "effect": rep}
                                      if rep is not None else None),
                "holdout": hold, "checks": review.checks, "review": {"status": review.status, "reason": review.reason},
                "heatmaps": maps if is_head else {},
                "caveat": self.caveat,
                "run_summary": {"experiments_run": total, "wall_seconds": round(time.time() - t0, 1),
                                "engineer_estimate_hours": None, "engineer_estimate_note": ""},
                "trace": trace, "field_log": self.field_log,
            }
            card["escalation"] = AgentMesh.escalation_for(card)
            if self.priors is not None:
                card["priors_used"] = self.priors.describe()
            self.store.put(card)
            cards.append(card)
        self.emit(f"[critic] done: {total} experiments; "
                  + (f"{len(accepted)} insight(s) survived: {head[0].id}" if head else "no hypothesis survived"))
        return {"headline": next((c for c in cards if c["headline"]), None), "cards": cards,
                "experiments": total, "wall_seconds": time.time() - t0}
