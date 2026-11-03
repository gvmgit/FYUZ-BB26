"""Commands:  status | build-twin | blind-test | demo | sweep | approve | serve | plant-fault | reveal

--mode local|aws (or ATR_MODE) is the toggle; see atr/modes.py for what each AWS service needs.
"""
import argparse
import json
import logging
import random
import time
import warnings

from . import config  # noqa: F401  (sets up sys.path for radp)
from . import modes
from .faults import Fault

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def parse_fault(spec: str) -> Fault:
    """kind:cell[:value] e.g. tilt:3:16  azimuth:1:200  cell_off:5"""
    parts = spec.split(":")
    return Fault(parts[0], int(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0)


def random_fault(seed: int) -> Fault:
    from .topology import default_inventory
    inv = default_inventory()
    rng = random.Random(seed)
    cell = rng.choice(sorted(inv))
    kind = rng.choice(["tilt", "azimuth", "cell_off"])
    if kind == "tilt":
        return Fault(kind, cell, rng.choice([4, 6, 7, 13, 15, 16]))
    if kind == "azimuth":
        return Fault(kind, cell, (inv[cell]["azimuth"] + rng.choice([-120, -60, 60, 120, 180])) % 360)
    return Fault(kind, cell)


def pick_fault(a) -> Fault:
    return parse_fault(a.fault) if a.fault else random_fault(a.random_seed or 0)


def blind_test(fault: Fault, n_ue: int = 1200, mode=None, engine: str = "strands", quiet: bool = False) -> dict:
    from . import runtime
    from .strands_agents import run_strands_investigation
    rt = runtime.build(mode, fault, n_ue, (lambda m: None) if quiet else print)
    try:
        out = (run_strands_investigation(rt.researcher, rt.mesh, rt.model_for) if engine == "strands"
               else rt.researcher.investigate())
    finally:
        rt.close()
    head, truth = out["headline"], (rt.world.truth() if rt.world else None)
    print("\n=== REVEAL ===")
    print(f"hidden fault : {truth}")
    located = False
    if head:
        c = head["claim"]
        located = truth is not None and c["kind"] == truth["kind"] and c["cell"] == truth["cell"]
        print(f"researcher   : {c['kind']} on cell {c['cell']}, estimate {c['estimate']}")
        print(f"fault located: {'YES' if located else 'NO'}"
              + (f", estimated value {c['value']:g} vs true {truth['value']:g}"
                 if located and truth["kind"] != "cell_off" else ""))
        print(f"experiments  : {out['experiments']}   wall time: {out['wall_seconds']:.0f}s")
    else:
        print("researcher   : no claim survived the Critic")
    return {"truth": truth, "headline": head, "experiments": out["experiments"], "located": located}


def research_once(fault, a, quiet: bool = False) -> dict:
    from .research_cycle import ResearchCycle
    loud = ("[protocol]", "[observer]", "[lab]", "[critic] lab", "[artefacts]")
    emit = (lambda m: None) if quiet else (lambda m: print(m) if m.startswith(loud) else None)
    out = ResearchCycle(fault, a.mode, emit, windows=a.windows, seed0=a.seed, engine=a.engine).run()
    head, truth = out.get("headline"), out.get("truth")
    located = bool(head and truth and head["claim"]["kind"] == truth["kind"] and head["claim"]["cell"] == truth["cell"])
    replicated = bool(head and (head.get("lab_replication") or {}).get("passed"))
    print("\n=== RESEARCH CYCLE ===")
    print(f"lab's secret fault : {truth}")
    if head:
        print(f"finding            : {head['title']} ({head['verdict']})")
        print(f"observed in        : {head['observation']['description']}")
        if head.get("lab_replication"):
            print(f"lab replication    : {'passed' if replicated else 'did not pass'}; "
                  + "; ".join(f"{k}: {v}" for k, v in head["lab_replication"]["rival_status"].items()))
        for art in head.get("artefacts", []):
            if art["type"] == "early_warning_indicator":
                print(f"early warning      : {art['rule']} -> {art['status']}")
        print(f"fault located      : {'YES' if located else 'NO'}")
        print(f"next               : {head['escalation'].get('proposed_action') or head['escalation']['reason']}")
    else:
        print("finding            : none survived the Critic")
    print(f"experiments {out.get('experiments')}, lab campaigns {out.get('lab_campaigns')}, {out.get('seconds')}s; card stored, "
          f"graph updated; review at http://localhost:8765/console/ (atr.cli serve)")
    return {"located": located, "replicated": replicated, "truth": truth}


def cmd_status(a):
    svc = modes.resolve(a.mode)
    print(f"mode: {svc.mode}   region: {svc.region}")
    missing = svc.missing_env()
    for s in modes.SERVICES:
        note = f"  MISSING: {', '.join(missing[s])}" if s in missing else ""
        print(f"  {s:<17} {svc.choice[s]}{note}")
    print("\nAWS mode is ready." if svc.mode == "aws" and not missing else
          "\nLocal mode: nothing else needed." if svc.mode == "local" and not any(svc.aws(s) for s in modes.SERVICES)
          else "\nSet the missing variables above before running.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "aws"], help="the toggle (default: ATR_MODE or local)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("build-twin").add_argument("--retrain", action="store_true")
    for name in ("blind-test", "demo", "plant-fault"):
        t = sub.add_parser(name)
        t.add_argument("--fault", help="kind:cell[:value], e.g. tilt:3:16 (the teammate's secret)")
        t.add_argument("--random-seed", type=int)
        t.add_argument("--n-ue", type=int, default=1200)
        if name == "blind-test":
            t.add_argument("--engine", choices=["strands", "classic"], default="strands")
        if name == "demo":
            t.add_argument("--pace", type=float, default=0.0, help="seconds between steps for a live walkthrough")
    sw = sub.add_parser("sweep")
    sw.add_argument("--n", type=int, default=10)
    sub.add_parser("reveal")
    ap_ = sub.add_parser("approve")
    ap_.add_argument("card_id")
    ap_.add_argument("--by", required=True, help="name of the person approving")
    rg = sub.add_parser("registry", help="models and tools the checkpoint lets agents call (withdraw = pause)")
    rg.add_argument("action", choices=["list", "approve", "withdraw"])
    rg.add_argument("--kind", choices=["Model", "Tool"], default="Model")
    rg.add_argument("--name", help="model id or tool name")
    rg.add_argument("--by", default="operator")
    pc = sub.add_parser("policy-check", help="ask the checkpoint for one decision and print its record (exit 3 = DENY)")
    pc.add_argument("--principal-type", choices=["Agent", "Human"], default="Agent")
    pc.add_argument("--principal", required=True)
    pc.add_argument("--action", required=True)
    pc.add_argument("--resource", help="model id, card id or tool name (default: derived from the action)")
    pc.add_argument("--run-id")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8765)
    ft = sub.add_parser("finetune-twin", help="investigate, approve, fine-tune the twin, re-test on a fresh survey")
    ft.add_argument("--fault", default="tilt:3:16")
    ft.add_argument("--bias", default="1:-3,4:2.5", help="twin mismatch in the simulated field: cell:dB,cell:dB")
    ft.add_argument("--approved-by", required=True, help="the person approving the finding the twin will learn from")
    ft.add_argument("--bench", type=int, default=15, help="planted faults for the fresh-survey re-test (0 = skip)")
    mb = sub.add_parser("mobility", help="handover / radio-link-failure KPIs for moving UEs on the twin")
    mb.add_argument("--fault", help="plant this fault and compare with the healthy network, kind:cell[:value]")
    mb.add_argument("--shadowing", type=float, default=2.0, help="signal fluctuation in dB (assumption; 0 = MAVERIC's smooth)")
    mb.add_argument("--seeds", type=int, default=8, help="how many UE-track seeds to average")
    mb.add_argument("--sweep-rule", action="store_true", help="also tabulate hysteresis x time-to-trigger")
    dt = sub.add_parser("detectors", help="early-warning detector rules from accepted findings")
    dt.add_argument("action", choices=["build", "report", "scan"])
    dt.add_argument("--include-unreviewed", action="store_true",
                    help="build from confirmed findings no engineer has accepted yet (demo only)")
    dt.add_argument("--fault", help="scan: plant this fault in a simulated survey, kind:cell[:value]")
    dt.add_argument("--bias", help="scan: twin mismatch in the simulated survey, cell:dB,cell:dB")
    dt.add_argument("--seed", type=int, default=9001, help="scan: survey seed")
    dt.add_argument("--investigate", action="store_true", help="scan: if a rule fires, run the full investigation")
    xp = sub.add_parser("explore", help="search the twin for a change worth making (no planted fault)")
    xp.add_argument("--k", type=int, default=3, help="how many hypotheses to test")
    xp.add_argument("--engine", choices=["strands", "classic"], default="strands")
    xp.add_argument("--objective", choices=["network", "handover"], default="network",
                    help="network: cell tilt/azimuth; handover: hysteresis and time-to-trigger")
    xp.add_argument("--baseline-hyst", type=float, default=3.0, help="handover: the documented hysteresis (dB)")
    xp.add_argument("--baseline-ttt", type=int, default=3, help="handover: the documented time-to-trigger (ticks)")
    vd = sub.add_parser("verdict", help="engineer verdict on a finding card")
    vd.add_argument("card_id")
    vd.add_argument("decision", choices=["accept", "reject", "request"])
    vd.add_argument("--by", required=True)
    vd.add_argument("--comment", default="")
    vd.add_argument("--request", default="", help="with 'request': what to test next")
    sub.add_parser("run-followups", help="run the experiments engineers asked for")
    ex = sub.add_parser("export-dataset", help="generate language-model training data from blind tests")
    ex.add_argument("--out", default="finetune/data")
    ex.add_argument("--repeats", type=int, default=1, help="train-split repeats (more = more data, slower)")
    ev = sub.add_parser("eval-ranker", help="score baselines (and optionally a tuned model endpoint) on the test split")
    ev.add_argument("--data", default="finetune/data")
    ev.add_argument("--endpoint", help="SageMaker endpoint hosting the tuned model")
    rs = sub.add_parser("research", help="one autonomous research cycle: lab campaign -> ingest -> observe -> investigate "
                                         "-> replicate in the lab -> analyse -> falsify -> publish -> graph -> artefacts")
    rs.add_argument("--fault", help="fault the lab holds secretly, kind:cell[:value] (omit for a healthy lab)")
    rs.add_argument("--random-seed", type=int, help="pick a random secret fault")
    rs.add_argument("--windows", type=int, default=6, help="one-minute KPI windows in the overnight campaign")
    rs.add_argument("--seed", type=int, default=101)
    rs.add_argument("--engine", choices=["strands", "classic"], default="strands")
    rs.add_argument("--nightly", type=int, default=0, help="queue N cycles with random secret faults and run them")
    gr = sub.add_parser("graph", help="finding graph: summary and the recommended test strategy for a fault class")
    gr.add_argument("--kind", default="tilt", choices=["tilt", "azimuth", "cell_off"])
    sub.add_parser("protocol", help="write the Step Functions definition to deploy/stepfunctions/protocol.asl.json")
    tn = sub.add_parser("train-nlm", help="LoRA fine-tune the research-tuned language model on accepted findings")
    tn.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    tn.add_argument("--epochs", type=int, default=3)
    tn.add_argument("--include-unreviewed", action="store_true", help="also train on confirmed findings nobody accepted yet")
    tn.add_argument("--data", default="finetune/data")
    a = ap.parse_args()

    if a.cmd == "status":
        cmd_status(a)
    elif a.cmd == "build-twin":
        from .twin import RFTwin
        print("held-out validation RMSE (dB):", RFTwin.load_or_build(retrain=a.retrain).validate())
    elif a.cmd == "blind-test":
        blind_test(pick_fault(a), a.n_ue, a.mode, a.engine)
    elif a.cmd == "demo":
        from .demo import run_demo
        run_demo(pick_fault(a), a.mode, a.n_ue, a.pace)
    elif a.cmd == "sweep":
        from . import runtime
        svc = modes.resolve(a.mode)
        rt = runtime.build(a.mode, emit=lambda m: None) if svc.aws("scheduler") else None
        sched = rt.scheduler if rt else __import__("atr.scheduler", fromlist=["x"]).LocalScheduler()
        for i in range(1, a.n + 1):
            sched.submit({"random_seed": i}, priority=i % 3)
        if svc.aws("scheduler"):
            print(f"queued {a.n} investigations on SQS; a worker consumes them")
        else:
            rows = sched.run_all(lambda job: blind_test(random_fault(job["random_seed"]), quiet=True))
            ok = sum(r["result"]["located"] for r in rows)
            print(f"\nsweep: {ok}/{len(rows)} hidden faults located")
    elif a.cmd == "approve":
        from .checkpoint import PolicyViolation
        from .runtime import make_checkpoint
        from .serve import make_store
        store = make_store()
        card = store.get(a.card_id)
        if not card:
            raise SystemExit(f"no card {a.card_id}")
        # the approval joins the investigation's trace, and is itself a policy decision (only a Human may approve)
        cp = make_checkpoint(a.mode, run_id=card.get("run_id"), unit="approval", trace_id=card.get("trace_id"))
        try:
            cp.enforce("Human", a.by, "ApproveChange", a.card_id)
        except PolicyViolation as e:
            raise SystemExit(f"refused: {e}")
        card["escalation"].update({"status": "approved", "approved_by": a.by, "approved_at": time.time()})
        store.put(card)
        cp.record({"kind": "human_approval", "card": a.card_id, "by": a.by})
        print(f"{a.card_id}: approved by {a.by} (recorded in trace {cp.trace_id}; no change is made to the live network)")
    elif a.cmd == "registry":
        from .runtime import make_checkpoint
        cp = make_checkpoint(a.mode, unit="registry")
        if a.action == "list":
            for item in cp.registry.list():
                print(f"  {item['id']:<55} {item['status']}")
        else:
            status = {"approve": "approved", "withdraw": "withdrawn"}[a.action]
            item = cp.registry.set(a.kind, a.name, status, by=a.by)
            cp.record({"kind": "registry_change", "item": item["id"], "status": status, "by": a.by})
            print(f"{item['id']}: {status} by {a.by} (run {cp.run_id})")
    elif a.cmd == "policy-check":
        from .runtime import make_checkpoint
        cp = make_checkpoint(a.mode, run_id=a.run_id, unit="policy")
        rec = cp.decide(a.principal_type, a.principal, a.action, a.resource)
        print(json.dumps(rec, indent=2, default=str))
        raise SystemExit(0 if rec["decision"] == "ALLOW" else 3)
    elif a.cmd == "plant-fault":
        from . import aws_adapters as aws
        from .twin import RFTwin, default_inventory
        from .world import World
        fault = pick_fault(a)
        twin = RFTwin.load_or_build()
        df = World(twin, default_inventory(), fault).field_observation(a.n_ue)
        print(aws.plant_fault(df.to_csv(index=False), {"kind": fault.kind, "cell": fault.cell, "value": fault.value}))
    elif a.cmd == "mobility":
        import numpy as np
        from .mobility import MobilityLab
        from .topology import default_inventory
        from .twin import RFTwin
        twin, inv = RFTwin.load_or_build(), default_inventory()
        lab, seeds = MobilityLab(twin, inv, 30, 60), range(1, a.seeds + 1)

        def summarise(settings, hyst=3.0, ttt=3):
            ks = [lab.kpis(lab.attach(s, settings, hyst=hyst, ttt=ttt, shadowing_db=a.shadowing)) for s in seeds]
            m = {k: float(np.mean([x[k] for x in ks])) for k in ("handovers", "ping_pongs", "rlf", "mro_metric", "rlf_ue_share_pct")}
            m["share"] = {c: float(np.mean([x["time_share_by_cell"][c] for x in ks])) for c in twin.cells}
            return m

        print(f"{lab.n_ue} UEs x {lab.n_ticks} s, {a.seeds} track seeds, shadowing {a.shadowing} dB, 1 grid unit = {lab.unit_m:.0f} m "
              f"(handover rule: hysteresis 3 dB, time-to-trigger 3 ticks; failure = SINR below MAVERIC's threshold)")
        show = lambda tag, m: print(f"  {tag:<22} handovers {m['handovers']:6.1f}  ping-pongs {m['ping_pongs']:5.1f}  "   # noqa: E731
                                    f"RLF ticks {m['rlf']:5.2f}  MRO metric {m['mro_metric']:6.2f}  time share "
                                    + " ".join(f"c{c}:{v:.2f}" for c, v in m["share"].items()))
        base = summarise(inv)
        show("healthy network", base)
        if a.fault:
            f = parse_fault(a.fault)
            show(f"with {f.kind} on cell {f.cell}", summarise(f.apply(inv)))
        if a.sweep_rule:
            net = parse_fault(a.fault).apply(inv) if a.fault else inv
            print("\n  handover rule sweep: hysteresis (dB) x time-to-trigger (ticks) -> handovers / ping-pongs / MRO metric")
            for hy in (0.0, 1.0, 3.0, 6.0, 9.0):
                print("   hyst %.0f: " % hy + "   ".join(
                    "ttt%d %s" % (tt, "%5.1f/%4.1f/%5.2f" % tuple(summarise(net, hy, tt)[k] for k in ("handovers", "ping_pongs", "mro_metric")))
                    for tt in (2, 3, 5, 8)))
    elif a.cmd == "detectors":
        from . import detector as D
        from .topology import default_inventory
        from .twin import RFTwin
        twin, inv = RFTwin.load_or_build(), default_inventory()
        if a.action == "build":
            from .serve import make_store
            rules = D.rules_from_store(make_store(), twin, inv, a.include_unreviewed)
            if not rules:
                raise SystemExit("no engineer-accepted confirmed fault findings to build from. Accept one with "
                                 "`verdict <card_id> accept --by NAME` (or use --include-unreviewed for a demo).")
            D.save_rules(rules)
            for r in rules:
                v = r["validation"]
                print(f"{r['id']}: cell {r['cell']} resembles {r['resembles']['kind']} ({r['resembles']['estimate']}): "
                      f"{r['status'].upper()}  false alarms {v['false_alarm']['fired']}/{v['false_alarm']['of']}, "
                      f"detected {v['detection']['fired']}/{v['detection']['of']}  - {r['gate_reason']}")
            print(f"{len(rules)} rule(s) written to {D.RULES_FILE}")
        elif a.action == "report":
            rules = D.load_rules()
            if not rules:
                raise SystemExit("no rules yet: run `detectors build` first")
            rep = D.network_report(rules, twin, inv)
            print(f"{rep['rules']} deployable rule(s) run together on {rep['surveys']} healthy surveys none of them were "
                  f"tuned on: {rep['surveys_that_triggered_any_rule']} triggered a rule "
                  f"({rep['network_false_alarm_rate']:.0%}, 95% interval {rep['ci95'][0]:.0%} to {rep['ci95'][1]:.0%}); "
                  f"each false trigger would start an investigation of about 90 experiments.")
        else:
            from .world import World
            rules = D.load_rules()
            if not rules:
                raise SystemExit("no rules yet: run `detectors build` first")
            bias = {int(k): float(v) for k, v in (p.split(":") for p in a.bias.split(",") if p)} if a.bias else None
            fault = parse_fault(a.fault) if a.fault else None
            obs = World(twin, inv, fault, bias).field_observation(1200, a.seed)
            results = D.scan(rules, twin, inv, obs)
            print(f"survey: fault={fault}, twin mismatch={bias}")
            for r in results:
                print(f"  {r['rule']}: {'FIRED' if r['fired'] else 'quiet'}  {r['values']}"
                      + (f"  over: {r['features_over_threshold']}; resembles {r['resembles']['kind']} "
                         f"({r['resembles']['estimate']})" if r["fired"] else ""))
            fired = [r for r in results if r["fired"]]
            print("verdict:", f"investigate cell(s) {sorted({r['cell'] for r in fired})}" if fired
                  else "nothing to investigate (saves ~90 experiments)")
            if fired and a.investigate:
                from . import runtime
                rt = runtime.build(a.mode, fault, 1200, print, twin_bias=bias, field_seed=a.seed, twin=twin)
                rt.researcher.investigate()
                rt.close()
    elif a.cmd == "explore":
        from . import runtime
        from .strands_explore import run_strands_exploration
        rt = runtime.build(a.mode, None, 1200, print, task="explore", explore_objective=a.objective,
                           baseline_rule=(a.baseline_hyst, a.baseline_ttt))
        try:
            out = (run_strands_exploration(rt.explorer, rt.mesh, rt.model_for) if a.engine == "strands"
                   else rt.explorer.explore(a.k))
        finally:
            rt.close()
        head = out["headline"]
        print("\n=== EXPLORATION RESULT ===")
        print(f"{len(out['cards'])} hypotheses tested, {out['experiments']} experiments, "
              f"{sum(1 for c in out['cards'] if c['review']['status'] == 'accepted')} survived the Critic")
        if head:
            print(f"best surviving finding: {head['title']}")
            print(f"  {head['escalation']['proposed_action']}")
            print(f"  caveat: {head['caveat']}")
        else:
            tested = len(out["cards"])
            if a.objective == "handover":
                print("nothing survived: " + (f"{tested} change(s) were tested and the Critic rejected them all (see the trace for why)"
                                              if tested else "no change in the scan clears the bar: the documented handover rule is "
                                                             "already near the best this search can find"))
            else:
                print("nothing survived: the documented network has no change worth making that passes the Critic")
    elif a.cmd == "verdict":
        from .anf import AgentMesh
        from .serve import make_scheduler, make_store
        from .verdicts import record_verdict
        decision = "request_experiment" if a.decision == "request" else a.decision
        card = record_verdict(make_store(), a.card_id, decision, a.by, a.comment, a.request,
                              scheduler=make_scheduler(), audit=AgentMesh().audit)
        print(f"{a.card_id}: {decision} recorded by {a.by}"
              + ("; follow-up queued (run `run-followups`)" if decision == "request_experiment" else ""))
    elif a.cmd == "run-followups":
        from . import followup, runtime
        from .scheduler import LocalScheduler
        from .verdicts import run_followups
        rt = runtime.build(a.mode, None, 1200, lambda m: None)
        sched = LocalScheduler(config.RUNS / "queue.json")
        done = run_followups(rt.store, sched, lambda card, req: followup.run_one(card, req, rt), audit=rt.mesh.audit)
        for d in done:
            r = d["result"]
            print(f"{d['job'].get('card_id')}: {'still holds' if r.get('still_holds') else r.get('error', 'does NOT clearly hold')}")
        print(f"{len(done)} follow-up(s) run")
    elif a.cmd == "finetune-twin":
        from .finetune_twin import before_after
        bias = {int(k): float(v) for k, v in (p.split(":") for p in a.bias.split(",") if p)} if a.bias else None
        fault = parse_fault(a.fault)
        bench = [random_fault(s) for s in range(1, a.bench + 1)] if a.bench else None
        before_after(fault, bias, a.approved_by, bench_faults=bench)
    elif a.cmd == "export-dataset":
        from .finetune_data import generate, records_from_verdicts
        from .serve import make_store
        print(generate(a.out, repeats=a.repeats))
        eng = records_from_verdicts(make_store())
        if eng:
            with open(f"{a.out}/engineer.jsonl", "w", encoding="utf-8") as f:
                f.write("".join(json.dumps(r) + "\n" for r in eng))
            print(f"{len(eng)} records from engineer verdicts -> {a.out}/engineer.jsonl")
    elif a.cmd == "eval-ranker":
        from . import finetune_eval as fe
        from .finetune_data import load
        train, test = load(f"{a.data}/train.jsonl"), load(f"{a.data}/test.jsonl")
        extra = {f"tuned model ({a.endpoint})": fe.EndpointPredictor(a.endpoint)} if a.endpoint else None
        print(fe.format_report(fe.baseline_report(train, test, extra)))
    elif a.cmd == "research":
        if a.nightly:
            from .scheduler import LocalScheduler
            sched = LocalScheduler()
            for i in range(1, a.nightly + 1):
                sched.submit({"random_seed": i}, priority=5)
            rows = sched.run_all(lambda job: research_once(random_fault(job["random_seed"]), a, quiet=True))
            print(f"\nnightly: {sum(r['result']['located'] for r in rows)}/{len(rows)} secret lab faults located and "
                  f"{sum(r['result']['replicated'] for r in rows)} replicated in the lab")
        else:
            research_once(parse_fault(a.fault) if a.fault else (random_fault(a.random_seed) if a.random_seed else None), a)
    elif a.cmd == "graph":
        from .graph import make_graph
        g = make_graph()
        print("finding graph:", g.summary())
        s = g.test_strategy(a.kind)
        print(f"\nrecommended test strategy for {a.kind} faults ({s['findings']} confirmed finding(s)):")
        for i, step in enumerate(s["strategy"], 1):
            print(f"  {i}. {step}")
    elif a.cmd == "protocol":
        from . import protocol
        print("wrote", protocol.write_definition(config.ROOT / "deploy" / "stepfunctions" / "protocol.asl.json"))
    elif a.cmd == "train-nlm":
        from .nlm import train_and_evaluate
        print(json.dumps(train_and_evaluate(a.base, a.epochs, a.include_unreviewed, a.data), indent=1))
    elif a.cmd == "reveal":
        from . import aws_adapters as aws
        print("hidden fault:", aws.read_truth())
    else:
        from .serve import serve
        serve(a.port)


if __name__ == "__main__":
    main()

