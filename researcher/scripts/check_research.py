"""Checks for pre-registration, the second instrument, engineer verdicts and exploration.

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_research.py
"""
import json
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import warnings
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from atr import config, explore, finetune_data, followup, prereg, runtime, verdicts  # noqa: E402
from atr.agents import Claim, REPLICATION_MIN_GAP  # noqa: E402
from atr.audit import FileAudit  # noqa: E402
from atr.faults import Fault  # noqa: E402
from atr.gateway import ToolError  # noqa: E402
from atr.scheduler import LocalScheduler  # noqa: E402
from atr.store import SQLiteStore  # noqa: E402
from atr.strands_explore import ExplorationRun, run_strands_exploration  # noqa: E402
from atr.twin import RFTwin  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


quiet = lambda m: None  # noqa: E731
twin = RFTwin.load_or_build()
tmp = Path(tempfile.mkdtemp())

# ---- 2. pre-registration ----------------------------------------------------------------------------------------------
rt = runtime.build("local", Fault("tilt", 2, 6), 1200, quiet, twin=twin)
audit_before = sum(1 for _ in open(config.AUDIT_LOG, encoding="utf-8")) if config.AUDIT_LOG.exists() else 0
out = rt.researcher.investigate()
rows = [json.loads(line) for line in open(config.AUDIT_LOG, encoding="utf-8")][audit_before:]
reg_idx = [i for i, r in enumerate(rows) if r["kind"] == "preregistration"]
first_exp = next(i for i, r in enumerate(rows) if r["kind"] == "tool_call" and r["agent"] == "experimenter")
check("every claim is pre-registered in the audit trail before the Experimenter runs anything",
      len(reg_idx) == len(out["cards"]) and max(reg_idx) < first_exp, f"{len(reg_idx)} registrations")
head = out["headline"]
check("the card carries the sealed prediction, verified, and whether it was met",
      head["preregistration"]["verified"] and head["preregistration"]["prediction_met"]
      and head["preregistration"]["hash"] in {r["hash"] for r in rows if r["kind"] == "preregistration"})
check("refuted claims record that their prediction was not met",
      any(not c["preregistration"]["prediction_met"] for c in out["cards"] if c["verdict"] != "confirmed"))
claim = rt.researcher.scientist.register(Claim("x", "tilt", 1, "t"), 5.0)
try:
    rt.researcher.experimenter.test(Claim("y", "tilt", 1, "unregistered"), rt.researcher.seeds, [1.0] * 8)
    check("the Experimenter refuses an unregistered claim", False)
except ValueError as e:
    check("the Experimenter refuses an unregistered claim", "not pre-registered" in str(e))
claim.registration["statement"]["predicted_effect"]["min_reduction"] = 0.01      # re-judge after the fact
check("changing a threshold after registration is detected", not prereg.verify(claim.registration))
try:
    rt.researcher.experimenter.test(claim, rt.researcher.seeds, [1.0] * 8)
    check("a tampered registration cannot be tested", False)
except ValueError:
    check("a tampered registration cannot be tested", True)
rt.close()

# ---- 3. the second instrument -----------------------------------------------------------------------------------------
rt = runtime.build("local", Fault("tilt", 3, 16), 1200, quiet, twin=twin)
ex, gw = rt.researcher.experimenter, rt.researcher.gw
hold = rt.researcher.holdout_seeds
good = ex.replicate(Claim("t3", "tilt", 3, ""), {"label": "cell 3 tilt 16", "value": 16}, hold)
check("the true tilt fault replicates strongly on the simulator tables",
      good["ci95"][0] > REPLICATION_MIN_GAP, f"improvement {good['mean_improvement']:.2f}, CI low {good['ci95'][0]:.2f}")
bad = ex.replicate(Claim("t2", "tilt", 2, ""), {"label": "cell 2 tilt 8", "value": 8}, hold)
check("a wrong tilt claim does NOT replicate on the second instrument",
      bad["ci95"][0] < REPLICATION_MIN_GAP, f"improvement {bad['mean_improvement']:.2f}")
check("azimuth and cell-off claims have no second instrument (stated, not faked)",
      ex.replicate(Claim("a", "azimuth", 3, ""), {"label": "", "value": 0}, hold) is None)
try:
    gw.invoke("run_experiment", {"config": {"cells": {"3": {"azimuth": 20.0}}, "instrument": "simulator"}, "seeds": [1]},
              agent="experimenter")
    check("the simulator instrument refuses azimuth changes", False)
except ToolError as e:
    check("the simulator instrument refuses azimuth changes", "tilt changes only" in str(e))
try:
    gw.invoke("run_experiment", {"config": {"cells": {"3": {"tilt": 40.0}}, "instrument": "simulator"}, "seeds": [1]},
              agent="experimenter")
    check("safety bounds still apply on the second instrument", False)
except ToolError:
    check("safety bounds still apply on the second instrument", True)
check("the second instrument does not use the twin's fitted model (reads the simulator tables)",
      rt.researcher.runner._sim().tables[(10, 1)].shape == (twin.n,))
out = rt.researcher.investigate()
tilt_card = out["headline"]
check("the confirmed tilt finding records its second-instrument replication",
      tilt_card["second_instrument"]["effect"]["ci95"][0] > REPLICATION_MIN_GAP
      and any("second instrument" in k["name"] and k["passed"] for k in tilt_card["checks"]))
rt.close()
rt = runtime.build("local", Fault("azimuth", 5, 210), 1200, quiet, twin=twin)
az = rt.researcher.investigate()["headline"]
check("an azimuth finding shows the second-instrument check as not applicable",
      any(k["passed"] is None and "second instrument" in k["name"] for k in az["checks"]))
rt.close()

# ---- 4. engineer verdicts, priors, follow-ups -------------------------------------------------------------------------
store = SQLiteStore(tmp / "f.sqlite")
audit = FileAudit(tmp / "audit.jsonl")
sched = LocalScheduler(tmp / "queue.json")
rt = runtime.build("local", Fault("tilt", 3, 16), 1200, quiet, twin=twin)
rt.researcher.store = store
out = rt.researcher.investigate()
cards = {c["claim"]["kind"] + str(c["claim"]["cell"]): c for c in out["cards"]}
head = out["headline"]
c1 = verdicts.record_verdict(store, head["id"], "accept", "Dr Rao", "matches the site visit", audit=audit)
check("an accept verdict is stored on the card and audited",
      c1["engineer_status"] == "accept" and c1["engineer_verdicts"][0]["by"] == "Dr Rao"
      and any('"engineer_verdict"' in line for line in open(tmp / "audit.jsonl", encoding="utf-8")))
for bad_args, why in [(("nonsense", "x"), "unknown verdict"), (("accept", ""), "no name")]:
    try:
        verdicts.record_verdict(store, head["id"], bad_args[0], bad_args[1])
        check(f"a verdict is refused: {why}", False)
    except (ValueError, PermissionError):
        check(f"a verdict is refused: {why}", True)
rejected = [c for c in out["cards"] if not c["headline"]][:3]
for c in rejected:
    verdicts.record_verdict(store, c["id"], "reject", "Dr Rao")
pri = verdicts.priors_from(store)
check("priors count the latest accept/reject per card", pri.counts == {"tilt": [1, sum(c["claim"]["kind"] == "tilt" for c in rejected)]
                                                                 , **{k: v for k, v in pri.counts.items() if k != "tilt"}}
      and pri.n_verdicts() == 1 + len(rejected), str(pri.describe()))
p2 = verdicts.Priors({"azimuth": [0, 4], "tilt": [3, 0], "cell_off": [0, 0]})
check("priors reorder kinds by engineer verdicts, ties keep the Scientist's order",
      p2.order_kinds(["azimuth", "cell_off", "tilt"]) == ["tilt", "cell_off", "azimuth"]
      and verdicts.Priors().order_kinds(["tilt", "azimuth", "cell_off"]) == ["tilt", "azimuth", "cell_off"])
sc = rt.researcher.scientist
sc.priors = p2
sym = sc.observe(rt.researcher.seeds)
claims = sc.propose(sym)
by_cell = {}
for c in claims:
    by_cell.setdefault(c.cell, []).append(c.kind)
check("with priors the Scientist tries the favoured kind first but still tests every kind",
      all(sorted(k) == ["azimuth", "cell_off", "tilt"] and k.index("tilt") < k.index("azimuth") for k in by_cell.values()),
      str(by_cell))
sc.priors = None
v3 = verdicts.record_verdict(store, cards["azimuth3"]["id"], "request_experiment", "Dr Rao", "",
                             "re-check on more seeds", scheduler=sched, audit=audit)
check("'request another experiment' queues a follow-up that survives a restart",
      LocalScheduler(tmp / "queue.json").pending() == 1 and v3["follow_ups"][0]["status"] == "queued")
done = verdicts.run_followups(store, LocalScheduler(tmp / "queue.json"),
                              lambda card, req: followup.run_one(card, req, rt), audit=audit)
res = store.get(cards["azimuth3"]["id"])["follow_ups"][-1]
check("the follow-up re-tests the saved field log on 32 fresh seeds and attaches the result",
      len(done) == 1 and res["status"] == "done" and res["result"]["seeds"] == 32 and "still_holds" in res["result"],
      f"still_holds={res['result'].get('still_holds')}")
tilt_fu = followup.retest_fault_card(head, rt.researcher.runner.twin, rt.researcher.inventory)
check("a tilt follow-up also uses the second instrument and still holds",
      tilt_fu["still_holds"] and "second_instrument_effect" in tilt_fu)
recs = finetune_data.records_from_verdicts(store)
check("accepted/rejected findings become labelled training records, the engineer's call being the label",
      recs and all(r["meta"]["source"] == "engineer" for r in recs)
      and {r["meta"]["label"]["survives_critic"] for r in recs} == {True, False}
      and all(json.loads(r["messages"][2]["content"])["reason"].startswith("engineer_") for r in recs),
      f"{len(recs)} records")
check("engineer labels are kept apart from the generated cases (split 'engineer')",
      all(r["meta"]["split"] == "engineer" for r in recs))

server = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (__import__("atr.serve", fromlist=["Handler"]).Handler,),
                                                      {"store": store}))
threading.Thread(target=server.serve_forever, daemon=True).start()
base_url = f"http://127.0.0.1:{server.server_address[1]}"
page = urllib.request.urlopen(base_url + "/").read().decode()
check("the review page shows verdict buttons and the pre-registration", "Request another experiment" in page
      and "Pre-registered" in page)
data = urllib.parse.urlencode({"card_id": cards["cell_off3"]["id"], "verdict": "reject", "by": "Web User"}).encode()
try:
    opener = urllib.request.build_opener(type("NoRedirect", (urllib.request.HTTPRedirectHandler,),
                                              {"redirect_request": lambda *a, **k: None}))
    opener.open(base_url + "/verdict", data)
    status = 200
except urllib.error.HTTPError as e:
    status = e.code
check("the page's form records a verdict (303 redirect)", status == 303
      and store.get(cards["cell_off3"]["id"])["engineer_status"] == "reject")
try:
    urllib.request.urlopen(base_url + "/verdict", urllib.parse.urlencode({"card_id": "nope", "verdict": "accept",
                                                                           "by": "x"}).encode())
    check("a verdict for an unknown card is rejected", False)
except urllib.error.HTTPError as e:
    check("a verdict for an unknown card is rejected", e.code == 400)
server.shutdown()
rt.close()

# ---- 1. exploration -----------------------------------------------------------------------------------------------------
rt = runtime.build("local", None, 1200, quiet, twin=twin, task="explore")
x = rt.explorer
hyps = x.scan()
check("the scan covers 30 single-setting interventions", len(x.scan_table) == 30)
check("every hypothesis is pre-registered before it is tested", hyps and all(prereg.verify(h.registration) for h in hyps))
blocked = [r for r in x.scan_table if r["gain"] >= explore.SCAN_MIN_GAIN and r["dcoverage"] < explore.GUARD_COVERAGE_PTS]
check("a change that cuts risk but loses coverage is set aside at the scan (guardrail)",
      blocked and not any((b["cell"], b["to"]) == (h.cell, h.to_value) for b in blocked for h in hyps),
      f"{[(b['cell'], b['kind'], b['to'], round(b['dcoverage'], 1)) for b in blocked]}")
check("the scan is deterministic", [h.id for h in hyps] == [h.id for h in runtime.build(
    "local", None, 1200, quiet, twin=twin, task="explore").explorer.scan()])
tilt_h = next((h for h in x.scan_table if h["kind"] == "tilt" and h["gain"] > 0), None)
th = explore.Hypothesis("t", "tilt", tilt_h["cell"], tilt_h["from"], tilt_h["to"], tilt_h["change"], "t", {"gain": 1.0})
x.register(th)
rep = x.replicate(th)
check("a tilt hypothesis is replicated on the simulator tables", rep is not None and "mean_improvement" in rep)
azh = explore.Hypothesis("a", "azimuth", 4, 90, 110, {"azimuth": 110.0}, "a", {"gain": 1.0})
check("an azimuth hypothesis has no second instrument", x.replicate(azh) is None)
try:
    x.test(explore.Hypothesis("u", "tilt", 1, 10, 8, {"tilt": 8.0}, "u"))
    check("an unregistered hypothesis cannot be tested", False)
except ValueError:
    check("an unregistered hypothesis cannot be tested", True)

eff = lambda mean, lo, win: {"mean_improvement": mean, "ci95": [lo, mean + 1], "win_rate": win, "cohens_d": 3.0, "n": 8}  # noqa: E731
def synth(rlf, sinr, cov, kind="tilt"):
    h = explore.Hypothesis("s", kind, 2, 10, 6, {"tilt": 6.0}, "s", {"gain": 1.0})
    return explore.InsightEvidence(h, {"rlf": rlf, "sinr": sinr, "coverage": cov}, {}, {})
crit = explore.InsightCritic()
r1 = crit.review(synth(eff(1.2, 0.6, 0.75), eff(0.5, 0, 1), eff(0, 0, 1)))
check("the Critic rejects a gain that is not robust to the twin's own error",
      r1.status == "rejected" and "twin's own error" in r1.reason)
r2 = crit.review(synth(eff(1.5, 0.8, 1.0), eff(0.5, 0, 1), eff(-2.0, -3, 0.2)))
check("the Critic rejects a change that trades away coverage", r2.status == "rejected" and "coverage" in r2.reason)
r3 = crit.review(synth(eff(1.5, 0.8, 1.0), eff(-1.0, -2, 0.2), eff(0, 0, 1)))
check("the Critic rejects a change that trades away SINR", r3.status == "rejected")
r4 = crit.review(synth(eff(1.5, 0.8, 1.0), eff(0.5, 0, 1), eff(0, 0, 1)))
check("a robust, guardrail-clean tilt hypothesis is then sent to the second instrument", r4.status == "needs_replication")
r5 = crit.review(synth(eff(1.5, 0.8, 1.0), eff(0.5, 0, 1), eff(0, 0, 1)), eff(0.1, -0.2, 0.5))
check("...and rejected if it does not replicate there", r5.status == "rejected" and "simulator" in r5.reason)
r6 = crit.review(synth(eff(1.5, 0.8, 1.0), eff(0.5, 0, 1), eff(0, 0, 1), kind="azimuth"))
check("an azimuth hypothesis goes straight to the holdout (second instrument not applicable)",
      r6.status == "needs_holdout" and any(c["passed"] is None for c in r6.checks))
rt.close()

rt = runtime.build("local", None, 1200, quiet, twin=twin, task="explore")
classic = rt.explorer.explore()
rt.close()
rt = runtime.build("local", None, 1200, quiet, twin=twin, task="explore")
strands = run_strands_exploration(rt.explorer, rt.mesh, rt.model_for)
rt.close()
ch, sh = classic["headline"], strands["headline"]
check("the plain and Strands pipelines reach the same insight in the same number of experiments",
      ch and sh and ch["claim"] == sh["claim"] and classic["experiments"] == strands["experiments"],
      f"{sh['title'] if sh else None} / {strands['experiments']} experiments")
check("the insight card says what it is and is not (twin-derived, not field-validated)",
      sh["type"] == "network_insight" and "NOT validated on the real network" in sh["caveat"])
check("a validated insight waits for a human: a field trial, not a network change",
      sh["escalation"]["status"] == "awaiting_human_approval" and "Field trial" in sh["escalation"]["proposed_action"])
check("the card shows its guardrail numbers and the heatmaps before and after",
      set(sh["kpi_effects"]) == {"failure_risk_pts", "mean_sinr_db", "coverage_pts"} and sh["heatmaps"].get("best_fix"))
check("the Critic also rejected hypotheses (it is not rubber-stamping)",
      any(c["review"]["status"] == "rejected" for c in strands["cards"]) and
      all(c["preregistration"]["verified"] for c in strands["cards"]))

rt = runtime.build("local", None, 1200, quiet, twin=twin, task="explore")
run = ExplorationRun(rt.explorer, rt.mesh)
publish = run.tools_for("orchestrator")[0]
try:
    publish()
    check("publish_insights is refused before any scan", False)
except Exception as e:  # noqa: BLE001
    check("publish_insights is refused before any scan", "no hypotheses" in str(e))
scan, = run.tools_for("scientist")
test, replicate, holdout = run.tools_for("experimenter")
review = run.tools_for("critic")[0]
scan()
first = next(iter(run.hyps))
test(first)
review(first)
try:
    publish()
    check("publish_insights is refused while a hypothesis lacks a final verdict", False)
except Exception as e:  # noqa: BLE001
    check("publish_insights is refused while a hypothesis lacks a final verdict", "without a final" in str(e))
rt.close()

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)

