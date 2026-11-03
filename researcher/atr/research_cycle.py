"""One autonomous research cycle, run as the experiment protocol (protocol.py) over the emulated lab.

  RunLabCampaign  the lab executor runs an overnight campaign (the lab holds a hidden injected fault, or none)
  Ingest          KPI/alarm streams -> feature tables; interface logs -> log lake -> parsers -> events; text -> search
  Observe         cross-modal patterns from the feature tables and log search
  Investigate     the Strands agent team (Scientist, Experimenter, Critic) works the drive-test survey on the twin:
                  pre-registered claims, one-setting interventions, controls, twin-bias null, second instrument
  CrossCheck      adds the survey's per-cell suspicion as a fourth modality; does the confirmed claim sit on a
                  cross-modal pattern?
  PreRegister / ReplicateAcrossProfiles / Analyse / Falsify
                  the claim and its rivals are injected into a clean lab under two traffic profiles (Map); the Analyst
                  computes effect sizes; the replication rule decides
  Publish / UpdateFindingGraph / DeriveArtefacts / QueueRetraining
                  the card is enriched and stored, the finding graph grows, artefacts are derived and validated, and
                  the language-model retraining job is queued when enough new findings have accumulated

Agents never see the lab's injected fault: the investigation gets the survey, the Observer gets telemetry.
"""
import json
import time
import uuid
from typing import Callable, Optional

from . import config, protocol
from .anf import AgentMesh
from .artefacts import append_to_corpus, early_warning_indicator, fine_tuning_sample, load_corpus
from .faults import Fault
from .gateway import Gateway, Tool
from .graph import make_graph
from .ingest import Ingestion
from .observer import Observer
from .replicate import PROFILES, Analyst, LabReplicator, combine

NLM_STATE = config.RUNS / "nlm_state.json"
RETRAIN_EVERY = 3                 # queue a retraining job after this many new corpus samples
BASELINE_ID = "baseline-busy_hour"

from .gateway import LAB_SCHEMA  # noqa: E402,F811 (shared with deploy/provision_gateway.py)


def lab_gateway(executor, audit, policy) -> Gateway:
    gw = Gateway(audit=audit, policy=policy)
    gw.register(Tool("lab_execute", "Drive the lab through the test automation API: traffic profile, handover rule, "
                     "parameters, campaigns and clean-lab replications. Never touches the live network.",
                     LAB_SCHEMA, lambda **a: executor.execute(**a), validator=executor.validate))
    return gw


class ResearchCycle:
    def __init__(self, lab_fault: Optional[Fault] = None, mode: Optional[str] = None, emit: Callable[[str], None] = print,
                 windows: int = 6, seed0: int = 101, engine: str = "strands", twin=None, ingestion: Optional[Ingestion] = None,
                 store=None, graph=None, reps: int = 4):
        from . import modes
        from .lab import EmulatedLab, LabExecutor
        from .topology import default_inventory
        from .twin import RFTwin
        self.svc = modes.resolve(mode)
        self.mode, self.emit, self.engine = mode, emit or (lambda m: None), engine
        self.windows, self.seed0 = windows, seed0
        self.twin = twin or RFTwin.load_or_build()
        self.inventory = default_inventory()
        self.ing = ingestion or make_ingestion(self.svc)
        self.lab = EmulatedLab(self.twin, self.inventory)
        self.lab.inject_fault(lab_fault)            # the teammate's secret, held by the lab
        self.executor = LabExecutor(self.lab, sink=self.ing.ingest)
        self.mesh = AgentMesh()
        if self.svc.aws("lab"):      # the lab executor is a Lambda target on AgentCore Gateway
            import os
            from .aws_adapters import RemoteLabGateway
            self.gw = RemoteLabGateway(os.environ["ATR_LAB_GATEWAY_URL"], os.environ.get("ATR_GATEWAY_TOKEN"),
                                       self.mesh.audit, policy=self.mesh.authorize)
        else:
            self.gw = lab_gateway(self.executor, self.mesh.audit, self.mesh.authorize)
        self.analyst = Analyst(self.mesh)
        self.replicator = LabReplicator(self.executor, self.inventory, list(self.twin.cells), audit=self.mesh.audit,
                                        analyst=self.analyst, reps=reps)
        self.observer = Observer(self.ing.features, self.ing.search, list(self.twin.cells))
        self.graph = graph or make_graph(self.svc)
        self._store = store
        self.ctx: dict = {}

    @property
    def store(self):
        if self._store is None:
            from .serve import make_store
            self._store = make_store()
        return self._store

    # -- tasks (each takes the state machine's state and returns a small JSON result) ---------------------------------
    def tasks(self) -> dict:
        return {n: getattr(self, "t_" + n) for n in (
            "lab_campaign", "ingest", "observe", "investigate", "cross_check", "preregister_replication", "lab_replicate",
            "analyse", "falsify", "publish", "update_graph", "derive_artefacts", "queue_retraining")}

    def t_lab_campaign(self, s):
        have = self.ing.features.query("SELECT COUNT(*) AS n FROM kpi_window WHERE campaign = ?", (BASELINE_ID,))[0]["n"]
        if not have:
            self.gw.invoke("lab_execute", {"action": "replicate", "windows": 12, "seed0": 9001, "campaign_id": BASELINE_ID,
                                           "profile": "busy_hour", "with_logs": True}, agent="protocol")
        cid = f"campaign-{s['run_id']}"
        self.gw.invoke("lab_execute", {"action": "set_traffic", "profile": "busy_hour"}, agent="protocol")
        r = self.gw.invoke("lab_execute", {"action": "run_campaign", "windows": self.windows, "seed0": self.seed0,
                                           "campaign_id": cid}, agent="protocol")
        self.ctx["campaign"] = cid
        return r["campaign"]

    def t_ingest(self, s):
        cid = self.ctx["campaign"]
        q = lambda t: self.ing.features.query(f"SELECT COUNT(*) AS n FROM {t} WHERE campaign = ?", (cid,))[0]["n"]   # noqa: E731
        return {"kpi_window": q("kpi_window"), "alarm": q("alarm"), "iface_event": q("iface_event"),
                "indexed_docs": len(getattr(self.ing.search, "docs", [])) or None}

    def t_observe(self, s):
        self.mesh.authorize("observer", "query_features")
        obs = self.observer.observe(self.ctx["campaign"], BASELINE_ID)
        self.ctx["observation"] = obs
        self.emit(f"[observer] {Observer.describe(obs)}")
        return {"cross_modal_cells": [p["cell"] for p in obs["cross_modal"]], "description": Observer.describe(obs)}

    def t_investigate(self, s):
        from . import runtime
        truth = self.lab.truth()
        fault = Fault(truth["kind"], truth["cell"], truth["value"]) if truth else None
        rt = runtime.build(self.mode, fault, 1200, self.emit, twin=self.twin, field_seed=self.seed0 + 12000)
        try:
            if self.engine == "strands":
                from .strands_agents import run_strands_investigation
                out = run_strands_investigation(rt.researcher, rt.mesh, rt.model_for)
            else:
                out = rt.researcher.investigate()
        finally:
            rt.close()
        sym = getattr(rt.researcher, "last_symptom", None) or {}
        self.ctx.update(investigation=out, suspicion={int(k): float(v) for k, v in (sym.get("suspicion") or {}).items()},
                        symptom=sym.get("symptom_text", ""))
        h = out["headline"]
        return {"headline": h["title"] if h else None, "experiments": out["experiments"], "cards": len(out["cards"])}

    def t_cross_check(self, s):
        obs = self.observer.observe(self.ctx["campaign"], BASELINE_ID, rf_suspicion=self.ctx["suspicion"])
        self.ctx["observation"] = obs
        h = self.ctx["investigation"]["headline"]
        pat = next((p for p in obs["patterns"] if h and p["cell"] == h["claim"]["cell"]), None)
        res = {"headline_found": h is not None, "description": Observer.describe(obs)}
        if h:
            res.update(headline_cell=h["claim"]["cell"], modalities=pat["flagged"] if pat else [],
                       on_cross_modal_pattern=bool(pat and pat["cross_modal"]))
        self.emit(f"[observer] with the survey added: {res['description']}")
        return res

    def _second_cell(self, cell):
        sus = self.ctx["suspicion"]
        others = [c for c in sorted(sus, key=sus.get, reverse=True) if c != cell]
        return others[0] if others else None

    def t_preregister_replication(self, s):
        h = self.ctx["investigation"]["headline"]["claim"]
        target, rivals, reg = self.replicator.preregister(h, self._second_cell(h["cell"]), PROFILES)
        self.ctx.update(repl_target=target, repl_rivals=rivals, repl_registration=reg, replications={})
        self.emit(f"[protocol] replication pre-registered (hash {reg['hash'][:12]}): {target} vs rivals "
                  + ", ".join(f"{r['kind']}-{r['cell']}" for r in rivals))
        return {"profiles": list(PROFILES), "hash": reg["hash"], "rivals": [f"{r['kind']}-{r['cell']}" for r in rivals]}

    def t_lab_replicate(self, s):
        p = s["profile"]
        r = self.replicator.replicate_profile(self.ctx["repl_target"], self.ctx["observation"]["signature"],
                                              self.ctx["repl_rivals"], p)
        self.ctx["replications"][p] = r
        self.emit(f"[lab] {p}: injected claim reproduces the observed signature at {r['claim_similarity']:.2f}; rivals "
                  + ", ".join(f"{k} {v:.2f}" for k, v in r["rival_similarity"].items()))
        return {"profile": p, "claim_similarity": r["claim_similarity"], "reproduces": r["reproduces"]}

    def t_analyse(self, s):
        self.mesh.authorize("analyst", "effect_size")
        excluded = {}
        for c in self.ctx["investigation"]["cards"]:
            if c["verdict"].startswith("refuted") or c["verdict"].startswith("ruled"):
                excluded[c["claim"]["id"] if "id" in c["claim"] else f"{c['claim']['kind']}-{c['claim']['cell']}"] = \
                    (c.get("review") or {}).get("reason") or c["verdict"]
        res = combine(self.ctx["replications"], self.ctx["repl_rivals"], excluded)
        res.update(preregistration=self.ctx["repl_registration"], campaigns=self.replicator.campaigns)
        self.ctx["lab_replication"] = res
        return {"reproduces": res["reproduces"], "rival_status": res["rival_status"]}

    def t_falsify(self, s):
        res = self.ctx["lab_replication"]
        self.emit(f"[critic] lab replication: {'PASSED' if res['passed'] else 'FAILED'}; "
                  + "; ".join(f"{k}: {v}" for k, v in res["rival_status"].items()))
        return {"replicated": res["passed"]}

    def t_publish(self, s):
        inv = self.ctx["investigation"]
        obs = self.ctx["observation"]
        compact = {k: obs[k] for k in ("campaign", "baseline", "windows", "baseline_windows", "patterns", "z")}
        compact["description"] = Observer.describe(obs)
        published = []
        for card in inv["cards"]:
            card = self.store.get(card["id"]) or card
            card["observation"] = compact
            card["protocol"] = {"run_id": s["run_id"], "campaign": self.ctx["campaign"]}
            if card["headline"] and "lab_replication" in self.ctx:
                lr = self.ctx["lab_replication"]
                card["lab_replication"] = lr
                card["checks"] = [k for k in card["checks"] if k["name"] != "lab replication across traffic profiles"] + [{
                    "name": "lab replication across traffic profiles", "passed": lr["passed"],
                    "detail": "; ".join(f"{p}: reproduces {v['claim_similarity']:.2f}" for p, v in lr["profiles"].items())
                              + "; " + "; ".join(f"{k} {v}" for k, v in lr["rival_status"].items())}]
                if not lr["passed"]:
                    card["caveat"] = ((card.get("caveat") or "") + " Lab replication did not pass: "
                                      + ", ".join(k for k, v in lr["rival_status"].items() if v == "NOT excluded")).strip()
                    card["alternatives_not_excluded"] = [k for k, v in lr["rival_status"].items() if v == "NOT excluded"]
                card["escalation"] = AgentMesh.escalation_for(card)
            self.store.put(card)
            published.append(card["id"])
        self.ctx["published"] = published
        head = inv["headline"]
        return {"cards": published, "headline": head["id"] if head else None}

    def _headline_card(self):
        h = self.ctx["investigation"]["headline"]
        return self.store.get(h["id"]) if h else None

    def t_update_graph(self, s):
        out = None
        for cid in self.ctx["published"]:
            out = self.graph.add_finding(self.store.get(cid))
        return out or {}

    def t_derive_artefacts(self, s):
        card = self._headline_card()
        if card is None or card["verdict"] != "confirmed":
            return {"artefacts": []}
        arts = []
        ewi = early_warning_indicator(card, self.ctx["observation"], self.replicator)
        arts.append(ewi)
        strat = self.graph.test_strategy(card["claim"]["kind"])
        arts.append({"id": f"{card['id']}/test-strategy", "type": "test_strategy", **strat})
        sample = fine_tuning_sample(card, self.ctx["observation"], self.ctx["suspicion"], self.ctx["symptom"])
        n = append_to_corpus(sample)
        arts.append({"id": sample["id"], "type": "fine_tuning_sample", "corpus_size": n,
                     "used_for_training_when": "an engineer accepts the finding"})
        arts.append({"id": f"{card['id']}/detector", "type": "detector_rule",
                     "status": "built when an engineer accepts the finding (atr.cli detectors build)"})
        card["artefacts"] = arts
        self.store.put(card)
        self.graph.add_finding(card)
        self.emit(f"[artefacts] early-warning indicator: {ewi['rule']} -> {ewi['status']} "
                  f"(false alarms {ewi['validation']['false_alarm']['fired']}/{ewi['validation']['false_alarm']['of']}, "
                  f"detected {ewi['validation']['detection']['fired']}/{ewi['validation']['detection']['of']})")
        self.emit(f"[artefacts] test strategy for {card['claim']['kind']} faults: " + " -> ".join(strat["strategy"][:4]))
        return {"artefacts": [{"id": a["id"], "type": a["type"], "status": a.get("status")} for a in arts]}

    def t_queue_retraining(self, s):
        state = json.loads(NLM_STATE.read_text()) if NLM_STATE.exists() else {"trained_on": 0}
        n = len(load_corpus())
        queued = n - state.get("trained_on", 0) >= RETRAIN_EVERY
        if queued:
            from .serve import make_scheduler
            make_scheduler().submit({"job": "retrain-nlm", "corpus_size": n}, priority=8)
        return {"corpus_size": n, "trained_on": state.get("trained_on", 0), "retraining_queued": queued}

    # -- run ---------------------------------------------------------------------------------------------------------------
    def run(self) -> dict:
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        t0 = time.time()
        if self.svc.aws("protocol"):   # Step Functions runs the same definition; tasks run in deploy/lambda_fn/protocol_handler.py
            from .aws_adapters import StepFunctionsProtocol
            sfn = StepFunctionsProtocol(region=self.svc.region)
            arn = sfn.start({"run_id": run_id})
            res = sfn.wait(arn)
            return {"run_id": run_id, "execution": arn, **res, "history": sfn.history(arn),
                    "seconds": round(time.time() - t0, 1)}
        sm = protocol.LocalStateMachine(protocol.definition(), self.tasks(), emit=self.emit)
        try:
            final = sm.run({"run_id": run_id})
        finally:
            self.ctx["history"] = sm.history
        card = self._headline_card()
        if card is not None:
            card["protocol"] = {"run_id": run_id, "campaign": self.ctx["campaign"], "history": sm.history,
                                "engine": "local ASL interpreter" if not self.svc.aws("protocol") else "AWS Step Functions"}
            self.store.put(card)
        return {"run_id": run_id, "final_state": final, "history": sm.history, "headline": card,
                "truth": self.lab.truth(), "seconds": round(time.time() - t0, 1),
                "lab_campaigns": self.replicator.campaigns, "experiments": self.ctx["investigation"]["experiments"]}


def make_ingestion(svc=None) -> Ingestion:
    from . import modes
    svc = svc or modes.resolve()
    kw = {}
    if any(svc.aws(s) for s in ("stream", "loglake", "features", "search")):
        from . import aws_adapters as aws
        if svc.aws("stream"):
            kw["bus"] = aws.KinesisStreams(region=svc.region)
        if svc.aws("loglake"):
            kw["lake"] = aws.S3LogLake(region=svc.region)
        if svc.aws("features"):
            kw["features"] = aws.AthenaFeatureStore(region=svc.region)
        if svc.aws("search"):
            kw["search"] = aws.OpenSearchLogSearch(region=svc.region)
    return Ingestion(**kw)
