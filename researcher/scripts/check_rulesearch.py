"""Checks for the handover-rule search: plumbing, the Critic's rules, honesty about assumptions, and both engines.

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_rulesearch.py
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from atr import explore_rule as XR, followup, mobility as M, prereg, runtime  # noqa: E402
from atr.explore import Hypothesis, InsightEvidence  # noqa: E402
from atr.gateway import ToolError  # noqa: E402
from atr.serve import render_card  # noqa: E402
from atr.strands_explore import ExplorationRun, run_strands_exploration  # noqa: E402
from atr.topology import default_inventory  # noqa: E402
from atr.twin import RFTwin  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


twin, inv = RFTwin.load_or_build(), default_inventory()
quiet = lambda m: None  # noqa: E731


def rt_for(baseline=(3.0, 3)):
    return runtime.build("local", None, 1200, quiet, twin=twin, task="explore", explore_objective="handover", baseline_rule=baseline)


# ---- plumbing: the documented rule, the one-knob rule, the twin-error stress ---------------------------------------------------
rt = rt_for()
g = rt.researcher.gw


def mob(baseline=None, **kw):
    cfg = {"objective": "mobility", **kw}
    if baseline:
        cfg["baseline_rule"] = {"hyst": baseline[0], "ttt": baseline[1]}
    return g.invoke("run_experiment", {"config": cfg, "seeds": [1, 2, 3, 4]}, agent="experimenter")


lab = rt.researcher.runner._mobility_lab(30, 60)
direct = [lab.kpis(lab.attach(s, inv, hyst=0.0, ttt=2, shadowing_db=2.0)) for s in (1, 2, 3, 4)]
via = mob(baseline=(0.0, 2))
check("the stated documented rule becomes the baseline of every experiment",
      np.isclose(via["mean"]["handovers_per_ue_min"], np.mean([k["handovers_per_ue_min"] for k in direct])),
      f"{via['mean']['handovers_per_ue_min']:.3f} handovers/UE-min")
check("one knob changed against the DOCUMENTED rule is allowed", mob(baseline=(0.0, 2), hyst=4.0)["objective"] == "mobility")
for kw, why in [({"baseline": (0.0, 2), "hyst": 4.0, "ttt": 5}, "two knobs against the documented rule"),
                ({"baseline": (0.0, 2), "ttt": 99}, "time-to-trigger out of range"),
                ({"baseline": (30.0, 2)}, "an impossible documented hysteresis")]:
    try:
        mob(**kw)
        check(f"blocked: {why}", False)
    except ToolError:
        check(f"blocked: {why}", True)
check("a change equal to the documented value is not a change", mob(baseline=(0.0, 2), hyst=0.0, ttt=2)["objective"] == "mobility")
a, b = mob(twin_error_db=2.0), mob(twin_error_db=2.0)
c = mob(twin_error_db=0.0)
check("twin error is drawn per seed: deterministic, paired, and it changes the result",
      a["per_seed"] == b["per_seed"] and a["per_seed"] != c["per_seed"]
      and np.array_equal(lab.twin_error(1, 2.0), lab.twin_error(1, 2.0)) and not np.array_equal(lab.twin_error(1, 2.0), lab.twin_error(2, 2.0)))
lab10 = M.MobilityLab(twin, inv, n_ue=10, n_ticks=30)
ident = True
for hy, tt in ((0.0, 2), (3.0, 3)):
    ref = lab10.attach(1, inv, hyst=hy, ttt=tt, shadowing_db=2.0, twin_error_db=2.0, engine="maveric")
    fast = lab10.attach(1, inv, hyst=hy, ttt=tt, shadowing_db=2.0, twin_error_db=2.0)
    r = ref.sort_values(["tick", "ue_id"]).reset_index(drop=True)
    f = fast.sort_values(["tick", "ue_id"]).reset_index(drop=True)
    ident &= bool((r.cell_id.map(lambda x: "RLF" if str(x) == "RLF" else str(int(float(x)))).values == f.cell_id.astype(str).values).all())
check("the fast attachment still equals MAVERIC's with twin error and shadowing both on", ident)

# ---- the Critic's rules ----------------------------------------------------------------------------------------------------
e = lambda mean, lo, win: {"mean_improvement": mean, "ci95": [lo, mean + 2], "win_rate": win, "cohens_d": 3.0, "n": 8}  # noqa: E731
crit = XR.RuleCritic()


def ev(cost, pp=None, fl=None):
    h = Hypothesis("s", "handover_rule", None, 0.0, 4.0, {"hyst": 4.0}, "s", {"gain": 10.0, "knob": "hyst"})
    return InsightEvidence(h, {"cost": cost, "ping_pongs": pp or e(0.1, 0.0, 1.0), "failures": fl or e(0.0, -0.001, 0.5)},
                           {}, {}, primary="cost")


check("a gain not distinguishable from zero is rejected", crit.review(ev(e(1.2, 0.4, 0.75))).status == "rejected")
check("more ping-pongs is rejected even if the cost falls", crit.review(ev(e(10, 8, 1.0), pp=e(-0.2, -0.3, 0.1))).status == "rejected")
check("more failures is rejected", crit.review(ev(e(10, 8, 1.0), fl=e(-0.05, -0.08, 0.1))).status == "rejected")
r1 = crit.review(ev(e(10, 8, 1.0)))
check("a robust, guardrail-clean gain goes to the holdout, and says there is no second instrument",
      r1.status == "needs_holdout" and any(k["passed"] is None and "second instrument" in k["name"] for k in r1.checks))
weak = {**e(5.0, 3.0, 0.75), "shadowing_db": 1.0}
r2 = crit.review(ev(e(10, 8, 1.0)), None, weak)
check("a holdout that is weak at another fluctuation level is rejected as depending on my assumption",
      r2.status == "rejected" and "shadowing assumption" in r2.reason)
check("a holdout that holds is accepted", crit.review(ev(e(10, 8, 1.0)), None, {**e(8, 5, 1.0), "shadowing_db": 3.0}).status == "accepted")

# ---- scanning ----------------------------------------------------------------------------------------------------------------
x = rt.explorer
check("the documented 3 dB / 3 ticks rule: nothing clears the bar, and the search says so",
      x.scan() == [] and x.scan_table[0]["gain"] < XR.SCAN_MIN_GAIN_PCT, f"best alternative {x.scan_table[0]['gain']:.1f}%")
check("every candidate is a single-knob change against the documented rule",
      all(len(r["change"]) == 1 for r in x.scan_table) and len(x.scan_table) == 12)
rt0 = rt_for((0.0, 2))
hyps = rt0.explorer.scan()
check("a poor documented rule (0 dB, 2 ticks) gives pre-registered hypotheses",
      hyps and all(prereg.verify(h.registration) for h in hyps), ", ".join(h.id for h in hyps))
check("equivalent candidates (identical effect) are collapsed to the smallest change, and the card says what else would do",
      any(h.scan["equivalent"] for h in hyps) and all(h.to_value not in (6.0, 8.0) for h in hyps if h.scan["knob"] == "hyst"),
      str([(h.id, h.scan["equivalent"]) for h in hyps if h.scan["equivalent"]]))
check("the scan is deterministic", [h.id for h in hyps] == [h.id for h in rt_for((0.0, 2)).explorer.scan()])
try:
    rt0.explorer.test(Hypothesis("u", "handover_rule", None, 0.0, 4.0, {"hyst": 4.0}, "u", {"knob": "hyst"}))
    check("an unregistered rule hypothesis cannot be tested", False)
except ValueError:
    check("an unregistered rule hypothesis cannot be tested", True)

# ---- both engines, the whole way ------------------------------------------------------------------------------------------------
classic = rt_for((0.0, 2)).explorer.explore()
rt1 = rt_for((0.0, 2))
strands = run_strands_exploration(rt1.explorer, rt1.mesh, rt1.model_for)
ch, sh = classic["headline"], strands["headline"]
check("plain and Strands pipelines reach the same finding in the same number of experiments",
      ch and sh and ch["claim"] == sh["claim"] and classic["experiments"] == strands["experiments"],
      f"{sh['title']} / {strands['experiments']} experiments")
check("the card is about the rule, not a cell, and records the documented rule it assumed",
      sh["claim"]["kind"] == "handover_rule" and sh["claim"]["cell"] is None and sh["claim"]["baseline_rule"] == {"hyst": 0.0, "ttt": 2})
check("it says it rests on an assumed documented rule and simulated KPIs",
      "ASSUMED documented handover rule" in sh["caveat"] and "simulated, not measured" in sh["caveat"])
check("no second instrument is claimed, and there is no heatmap (a rule is not spatial)",
      sh["second_instrument"] is None and not sh["heatmaps"] and any(k["passed"] is None for k in sh["checks"]))
check("the holdout records both fluctuation levels it tried",
      {r["shadowing_db"] for r in sh["holdout"]["all_runs"]} == {1.0, 3.0})
check("a validated rule waits for a human: a field trial in a limited area, with named knob and values",
      sh["escalation"]["status"] == "awaiting_human_approval" and "hysteresis (dB)" in sh["escalation"]["proposed_action"]
      and "limited area" in sh["escalation"]["proposed_action"])
check("every card is pre-registered and verified", all(c["preregistration"]["verified"] for c in strands["cards"]))
page = render_card(sh)
check("the review page labels the rule finding in its own units and shows the caveat",
      "Interruption-cost reduction (%)" in page and "ASSUMED" in page and "Request another experiment" in page)

# ---- an empty result is a result ---------------------------------------------------------------------------------------------------
rt2 = rt_for()
empty = run_strands_exploration(rt2.explorer, rt2.mesh, rt2.model_for)
check("when nothing clears the bar both engines return an empty result, not an error",
      empty["headline"] is None and empty["cards"] == [] and rt_for().explorer.explore()["cards"] == [])
rt4 = rt_for()
run = ExplorationRun(rt4.explorer, rt4.mesh)
try:
    run.tools_for("orchestrator")[0]()
    check("publishing before any scan is still refused", False)
except Exception as ex:  # noqa: BLE001
    check("publishing before any scan is still refused", "scan first" in str(ex))

# ---- the Critic really can say no ------------------------------------------------------------------------------------------------
rejected = rt_for((3.0, 8)).explorer.explore()
check("a gain that depends on my shadowing assumption is rejected (time-to-trigger 8 to 4)",
      rejected["cards"] and all(c["review"]["status"] == "rejected" for c in rejected["cards"])
      and any("shadowing assumption" in c["review"]["reason"] for c in rejected["cards"]),
      rejected["cards"][0]["review"]["reason"][:70] if rejected["cards"] else "no cards")

# ---- follow-up --------------------------------------------------------------------------------------------------------------------
rt3 = rt_for((0.0, 2))
fu = followup.run_one(sh, "", rt3)
check("an engineer's follow-up re-tests a rule finding at three fluctuation levels on 24 fresh seeds",
      fu["seeds"] == 24 and set(fu["gain_pct_by_shadowing"]) == {"1 dB", "2 dB", "3 dB"} and "still_holds" in fu,
      f"still_holds={fu['still_holds']}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
