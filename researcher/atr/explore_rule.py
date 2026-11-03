"""Exploration of the handover rule: is there a better hysteresis or time-to-trigger than the documented one?

Same discipline as the network search (atr/explore.py): candidates are single-knob changes (one setting at a time), pre-registered
before they are tested, judged on MAVERIC's own interruption cost (50 ms per handover, 1 s per failure tick, per UE-minute) with
guardrails, stress-tested against the twin's error, and re-tested on fresh seeds.

What is different, and stated plainly:
  - The documented network has no handover-rule inventory, so the baseline rule (default 3 dB, 3 ticks) is an ASSUMPTION you
    can override with the real one. The answer depends on it.
  - The KPIs come from simulated moving UEs on the twin's predicted signal. The signal-fluctuation level (shadowing) is my
    assumption, not MAVERIC's, so the holdout re-tests every finding at lower and higher fluctuation (1 and 3 dB) and keeps the
    weaker result.
  - No second instrument exists for mobility (the simulator tables have no motion), so that check is "not applicable" on the card.
  - On the default rule the honest result is usually "nothing is worth changing"; the search says so instead of inventing a finding.
"""
import time
from typing import List, Optional

from . import prereg, stats
from .explore import (Explorer, Hypothesis, InsightEvidence, InsightReview, STRESS_ERROR_DB, TWIN_ERROR_DB, improvement)

SCAN_MIN_GAIN_PCT = 3.0         # interruption-cost reduction (% of baseline) a candidate must show in the scan
GUARD_PING_PONG = -0.05         # ping-pongs per UE-minute may rise by at most this much
GUARD_FAILURE = -0.01           # radio-link-failure ticks per UE-minute may rise by at most this much
MIN_GAIN_CI_LOW_PCT = 1.0       # CI lower bound of the gain, % of baseline
MIN_WIN_RATE = 0.9
HOLDOUT_MIN_GAIN_PCT = 0.5
STRESS_SHADOWING_DB = (1.0, 3.0)
HYST_VALUES = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
TTT_VALUES = (2, 3, 4, 5, 6, 8)
KNOB = {"hyst": ("hysteresis", "dB"), "ttt": ("time-to-trigger", "ticks")}


def pct(effect: dict, base_mean: float) -> dict:
    """Express a paired effect as a percentage of the baseline's mean."""
    if base_mean <= 0:
        raise ValueError("the baseline interruption cost is zero, so there is nothing to reduce")
    k = 100.0 / base_mean
    return {**effect, "mean_improvement": effect["mean_improvement"] * k, "ci95": [x * k for x in effect["ci95"]],
            "unit": "% of baseline"}


def _fmt(e: dict) -> str:
    return f"{e['mean_improvement']:+.2f}% (CI {e['ci95'][0]:.2f}..{e['ci95'][1]:.2f}, wins {e['win_rate']:.0%} of seeds)"


class RuleCritic:
    name = "critic"

    def review(self, ev: InsightEvidence, replication: Optional[dict] = None,
               holdout: Optional[dict] = None) -> InsightReview:
        gain, pp, fl = ev.effects["cost"], ev.effects["ping_pongs"], ev.effects["failures"]
        checks = []
        ok = gain["ci95"][0] > MIN_GAIN_CI_LOW_PCT and gain["win_rate"] >= MIN_WIN_RATE
        checks.append({"name": f"cuts interruption cost with the twin allowed its own error ({TWIN_ERROR_DB} dB/cell)",
                       "passed": ok, "detail": _fmt(gain)})
        if not ok:
            return InsightReview("rejected", checks, "the gain is not distinguishable from zero once the twin's own error "
                                                     "is included")
        guard = pp["mean_improvement"] >= GUARD_PING_PONG and fl["mean_improvement"] >= GUARD_FAILURE
        checks.append({"name": "guardrails: ping-pongs and radio-link failures do not rise", "passed": guard,
                       "detail": f"ping-pongs {-pp['mean_improvement']:+.3f}/UE-min (limit {-GUARD_PING_PONG:+g}), "
                                 f"failure ticks {-fl['mean_improvement']:+.3f}/UE-min (limit {-GUARD_FAILURE:+g})"})
        if not guard:
            return InsightReview("rejected", checks, "it trades more ping-pongs or radio-link failures for fewer handovers")
        checks.append({"name": "replicates on a second instrument", "passed": None,
                       "detail": "not applicable: the simulator tables have no mobility data"})
        if holdout is None:
            return InsightReview("needs_holdout", checks, "re-test on fresh seeds at lower and higher signal fluctuation "
                                                          f"({STRESS_SHADOWING_DB[0]:g} and {STRESS_SHADOWING_DB[1]:g} dB) "
                                                          f"under a harsher twin error ({STRESS_ERROR_DB} dB/cell)")
        h_ok = holdout["ci95"][0] > HOLDOUT_MIN_GAIN_PCT and holdout["win_rate"] >= 0.8
        checks.append({"name": "holds on fresh seeds at other signal-fluctuation levels (the weaker of 1 and 3 dB)",
                       "passed": h_ok, "detail": f"at {holdout['shadowing_db']:g} dB: {_fmt(holdout)}"})
        if not h_ok:
            return InsightReview("rejected", checks, "it did not hold on fresh seeds at other signal-fluctuation levels, so "
                                                     "it depends on my shadowing assumption")
        return InsightReview("accepted", checks, "survived twin error, guardrails, and fresh-seed replication at other "
                                                 "signal-fluctuation levels")


class RuleExplorer(Explorer):
    metric_label = "Interruption-cost reduction (%)"
    sweep_label = "Scanned rule changes, by interruption-cost reduction (%)"
    caveat = ("Validated on the digital twin with a simulated signal-fluctuation model (re-tested at 1 and 3 dB) and an ASSUMED "
              "documented handover rule. The KPIs are simulated, not measured, and there is no second instrument for mobility. "
              "The next step is a human-approved field trial in a limited area.")

    def __init__(self, *a, baseline_rule=(3.0, 3), shadowing_db: float = 2.0, **kw):
        super().__init__(*a, **kw)
        self.base_hyst, self.base_ttt = float(baseline_rule[0]), int(baseline_rule[1])
        self.shadowing_db = shadowing_db
        self.critic = RuleCritic()

    # -- tool surface ------------------------------------------------------------------------------------------------
    def _mob(self, label: str, change: dict, seeds, twin_error: float, shadowing: Optional[float] = None,
             agent: str = "experimenter") -> dict:
        return self.gw.invoke("run_experiment", {"config": {
            "name": label, "objective": "mobility", "baseline_rule": {"hyst": self.base_hyst, "ttt": self.base_ttt},
            "shadowing_db": self.shadowing_db if shadowing is None else shadowing, "twin_error_db": twin_error, **change},
            "seeds": list(seeds)}, agent=agent)

    def _candidates(self):
        for v in HYST_VALUES:
            if v != self.base_hyst:
                yield "hyst", self.base_hyst, v, {"hyst": float(v)}
        for v in TTT_VALUES:
            if v != self.base_ttt:
                yield "ttt", self.base_ttt, v, {"ttt": int(v)}

    def _effects(self, base: dict, treat: dict) -> dict:
        bc = base["mean"]["interruption_s_per_ue_min"]
        return {"cost": pct(improvement(base, treat, "interruption_s_per_ue_min", -1), bc),
                "ping_pongs": improvement(base, treat, "ping_pongs_per_ue_min", -1),
                "failures": improvement(base, treat, "rlf_per_ue_min", -1)}

    def scan(self, k: int = 3) -> List[Hypothesis]:
        base = self._mob("scan baseline rule", {}, self.scan_seeds, 0.0, agent="scientist")
        bm = base["mean"]
        self.emit(f"[scientist] scanning {sum(1 for _ in self._candidates())} single-knob changes to the handover rule; documented "
                  f"rule: hysteresis {self.base_hyst:g} dB, time-to-trigger {self.base_ttt} ticks; baseline: "
                  f"{bm['handovers_per_ue_min']:.2f} handovers, {bm['ping_pongs_per_ue_min']:.2f} ping-pongs and "
                  f"{bm['rlf_per_ue_min']:.3f} failure ticks per UE-minute")
        rows = []
        for knob, a, b, change in self._candidates():
            r = self._mob(f"scan {knob} {a:g}->{b:g}", change, self.scan_seeds, 0.0, agent="scientist")
            m = r["mean"]
            rows.append({"knob": knob, "from": a, "to": b, "change": change, "run_id": r["run_id"],
                         "gain": 100 * (bm["interruption_s_per_ue_min"] - m["interruption_s_per_ue_min"])
                         / bm["interruption_s_per_ue_min"],
                         "dping": bm["ping_pongs_per_ue_min"] - m["ping_pongs_per_ue_min"],
                         "drlf": bm["rlf_per_ue_min"] - m["rlf_per_ue_min"]})
        self.scan_table = sorted(rows, key=lambda r: -r["gain"])
        ok = [r for r in rows if r["gain"] >= SCAN_MIN_GAIN_PCT and r["dping"] >= GUARD_PING_PONG and r["drlf"] >= GUARD_FAILURE]
        for r in (r for r in rows if r["gain"] >= SCAN_MIN_GAIN_PCT and r not in ok):
            self.emit(f"[scientist] set aside (guardrail): {KNOB[r['knob']][0]} {r['from']:g}->{r['to']:g} cuts cost "
                      f"{r['gain']:.1f}% but ping-pongs {-r['dping']:+.3f}, failures {-r['drlf']:+.3f} per UE-minute")
        # Candidates of the same knob with an identical effect are the same finding (hysteresis saturates once it exceeds the
        # signal differences). Keep the smallest change and say what else would do the same.
        groups = {}
        for r in ok:
            groups.setdefault((r["knob"], round(r["gain"], 4), round(r["dping"], 4), round(r["drlf"], 4)), []).append(r)
        ok = []
        for rs in groups.values():
            rs.sort(key=lambda r: abs(r["to"] - r["from"]))
            rs[0]["equivalent"] = [f"{KNOB[x['knob']][0]} {x['to']:g}" for x in rs[1:]]
            if rs[0]["equivalent"]:
                self.emit(f"[scientist] {KNOB[rs[0]['knob']][0]} {rs[0]['to']:g} has the same effect as "
                          f"{', '.join(rs[0]['equivalent'])} in the scan; keeping the smallest change")
            ok.append(rs[0])
        weight = (lambda kind: self.priors.weight(kind)) if self.priors is not None else (lambda kind: 1.0)
        ok.sort(key=lambda r: -r["gain"] * weight("handover_rule"))
        out = []
        for r in ok[:k]:
            name, unit = KNOB[r["knob"]]
            h = Hypothesis(f"insight-handover-{r['knob']}-{r['to']:g}", "handover_rule", None, r["from"], r["to"], r["change"],
                           (f"Changing the handover {name} from {r['from']:g} to {r['to']:g} {unit} will cut interruption cost "
                            f"by about {r['gain']:.0f}% without raising ping-pongs or radio-link failures."),
                           scan={"gain": round(r["gain"], 2), "dping": round(r["dping"], 3), "drlf": round(r["drlf"], 3),
                                 "knob": r["knob"], "equivalent": r.get("equivalent", [])})
            self.register(h)
            out.append(h)
            self.emit(f"[scientist] hypothesis {h.id}: {h.text}")
        if not out:
            best = self.scan_table[0]
            self.emit(f"[scientist] nothing in the scan clears the bar ({SCAN_MIN_GAIN_PCT:g}% cost reduction with guardrails "
                      f"intact): the best alternative, {KNOB[best['knob']][0]} {best['from']:g}->{best['to']:g}, gains only "
                      f"{best['gain']:.1f}%. The documented rule is already near the best in this search space.")
        return out

    def register(self, h: Hypothesis) -> Hypothesis:
        name, unit = KNOB[h.scan["knob"]]
        h.registration = prereg.register({
            "id": h.id, "kind": h.kind, "cell": None, "hypothesis": h.text,
            "design": {"intervention": f"handover {name} {h.from_value:g} -> {h.to_value:g} {unit}, network-wide",
                       "assumed_documented_rule": {"hyst_db": self.base_hyst, "ttt_ticks": self.base_ttt},
                       "controls": [f"twin error drawn per seed ({TWIN_ERROR_DB} dB/cell)", "guardrails",
                                    "fresh-seed holdout at 1 and 3 dB signal fluctuation under a harsher twin error",
                                    "second instrument: not available for mobility"]},
            "predicted_effect": {"metric": "interruption cost, s per UE-minute (MAVERIC weights)", "direction": "decrease",
                                 "predicted_gain_pct": h.scan["gain"], "min_gain_ci_low_pct": MIN_GAIN_CI_LOW_PCT,
                                 "min_win_rate": MIN_WIN_RATE,
                                 "guardrails": {"ping_pongs_per_ue_min_rise_at_most": -GUARD_PING_PONG,
                                                "failure_ticks_per_ue_min_rise_at_most": -GUARD_FAILURE},
                                 "shadowing_db_assumed": self.shadowing_db},
            "null_outcome": ("the gain is indistinguishable from zero once the twin is allowed its own error, or a guardrail "
                             "is breached, or it does not hold at other signal-fluctuation levels")}, self.audit)
        return h

    def test(self, h: Hypothesis) -> InsightEvidence:
        self._require(h)
        self.emit(f"[experimenter] testing {h.id} under twin error ({TWIN_ERROR_DB} dB/cell), {len(self.seeds)} seeds")
        base = self._mob("documented rule", {}, self.seeds, TWIN_ERROR_DB)
        treat = self._mob(h.id, h.change, self.seeds, TWIN_ERROR_DB)
        ev = InsightEvidence(h, self._effects(base, treat), base["mean"], treat["mean"], primary="cost")
        self.emit(f"    interruption cost {base['mean']['interruption_s_per_ue_min']:.4f} -> "
                  f"{treat['mean']['interruption_s_per_ue_min']:.4f} s/UE-min  (gain {_fmt(ev.effects['cost'])})")
        return ev

    def replicate(self, h: Hypothesis) -> Optional[dict]:
        return None      # no second instrument for mobility

    def holdout(self, h: Hypothesis) -> dict:
        self.emit(f"[experimenter] holdout for {h.id}: fresh seeds, twin error {STRESS_ERROR_DB} dB/cell, "
                  f"signal fluctuation {STRESS_SHADOWING_DB[0]:g} and {STRESS_SHADOWING_DB[1]:g} dB")
        runs = []
        for sig in STRESS_SHADOWING_DB:
            base = self._mob("holdout baseline", {}, self.holdout_seeds, STRESS_ERROR_DB, sig)
            treat = self._mob(f"holdout {h.id}", h.change, self.holdout_seeds, STRESS_ERROR_DB, sig)
            eff = self._effects(base, treat)["cost"]
            runs.append({**eff, "shadowing_db": sig})
            self.emit(f"    at {sig:g} dB: gain {_fmt(eff)}")
        weakest = min(runs, key=lambda e: e["ci95"][0])
        return {**weakest, "all_runs": [{"shadowing_db": r["shadowing_db"], "effect": {k: v for k, v in r.items()
                                                                                       if k != "all_runs"}} for r in runs]}

    def retest(self, card: dict, seeds=range(301, 325), sigma: float = 2.0) -> dict:
        c = card["claim"]
        change = {c["knob"]: (float(c["value"]) if c["knob"] == "hyst" else int(c["value"]))}
        runs = {}
        for sig in (1.0, 2.0, 3.0):
            base = self._mob("follow-up baseline", {}, seeds, sigma, sig)
            treat = self._mob("follow-up rule", change, seeds, sigma, sig)
            runs[f"{sig:g} dB"] = self._effects(base, treat)["cost"]
        out = {"seeds": len(list(seeds)), "twin_error_db": sigma, "gain_pct_by_shadowing": runs}
        out["still_holds"] = all(e["ci95"][0] > 0.0 and e["win_rate"] >= 0.8 for e in runs.values())
        return out

    # -- card contents -----------------------------------------------------------------------------------------------
    def _title(self, h, validated: bool, verdict: str, gain: dict) -> str:
        name, unit = KNOB[h.scan["knob"]]
        return (f"Handover {name}: {h.from_value:g} to {h.to_value:g} {unit} cuts interruption cost "
                f"{gain['mean_improvement']:.1f}%" if validated else
                f"Handover {name}: {h.from_value:g} to {h.to_value:g} {unit} ({verdict})")

    def _claim(self, h) -> dict:
        name, unit = KNOB[h.scan["knob"]]
        return {"kind": "handover_rule", "cell": None, "knob": h.scan["knob"], "text": h.text, "value": h.to_value,
                "from": h.from_value, "estimate": f"{name} {h.to_value:g} {unit} (documented {h.from_value:g})",
                "baseline_rule": {"hyst": self.base_hyst, "ttt": self.base_ttt},
                "equivalent_changes": h.scan.get("equivalent", [])}

    def _sweep(self) -> list:
        return [{"label": f"{KNOB[r['knob']][0]} {r['from']:g}->{r['to']:g}", "mean": -r["gain"], "run_id": r["run_id"],
                 "value": r["to"]} for r in self.scan_table[:8]]

    def _kpi_effects(self, ev) -> dict:
        return {"interruption_cost_pct": ev.effects["cost"], "ping_pongs_per_ue_min": ev.effects["ping_pongs"],
                "failure_ticks_per_ue_min": ev.effects["failures"]}

    def _maps(self, head) -> dict:
        return {}        # a rule is not spatial: no heatmap
