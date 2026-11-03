"""The Telecom Researcher as Strands agents.

    Orchestrator --ask_scientist--> Scientist   (observe_symptom, propose_claims)
                 --ask_experimenter--> Experimenter (test_claim, run_control, run_holdout -> gateway run_experiment)
                 --ask_critic--> Critic         (review_claim)
                 --publish_findings--> finding cards

Specialists are Strands agents exposed to the Orchestrator as tools ("agents as tools"). The scientific work
happens in the tools (agents.py); the model only chooses which tool to call next. Publication is guarded in code:
`publish_findings` refuses to run until every claim has a terminal Critic verdict, and an accepted claim must have
a control run and a held-out replication on record - whatever the model decides.
"""
import functools
import json
import time
from typing import Callable, Dict, List

from strands import Agent, tool

from . import prereg
from .anf import AgentMesh
from .researcher import Researcher

PROMPTS = {
    "scientist": ("You are the Scientist in a telecom network research team. Use observe_symptom once, then "
                  "propose_claims once, and reply with the propose_claims JSON unchanged."),
    "experimenter": ("You are the Experimenter. You change exactly one network setting at a time on the MAVERIC "
                     "digital twin through your tools. Requests look like 'test <claim_id>', 'control <claim_id>', "
                     "'null <claim_id>', 'holdout <claim_id>' or 'replicate <claim_id>'. Call the matching tool once "
                     "and reply with its JSON unchanged."),
    "critic": ("You are the Critic. Try to disprove findings. For 'review <claim_id>' call review_claim once and "
               "reply with its JSON unchanged. Never accept a claim yourself; only the tool decides."),
    "orchestrator": ("You orchestrate a network-fault investigation. 1) ask_scientist for claims. 2) For each claim: "
                     "ask_experimenter 'test <id>', then ask_critic 'review <id>'. If the Critic answers "
                     "needs_control ask_experimenter 'control <id>'; if needs_null ask 'null <id>'; if needs_holdout "
                     "ask 'holdout <id>'; if needs_replication ask 'replicate <id>'; then ask the "
                     "Critic to review again, until accepted or rejected. 3) Only then call publish_findings. "
                     "Never publish early. Always use the exact claim_ids from the Scientist's JSON (for example "
                     "'tilt-3'); never invent ids. Test every claim before asking the Critic to review it."),
}


class Investigation:
    """Shared ledger for one run: what each agent produced, and what the publication guard checks."""

    def __init__(self, researcher: Researcher, mesh: AgentMesh):
        self.r, self.mesh = researcher, mesh
        self.t0 = time.time()
        self.trace: List[str] = []
        self.symptom = None
        self.base_scores: List[float] = []
        self.claims, self.evidence, self.controls, self.holdouts, self.reviews = {}, {}, {}, {}, {}
        self.nulls, self.replications = {}, {}
        self.result = None
        self.tokens = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        self.last_output: Dict[str, str] = {}     # agent -> the exact JSON its last tool returned

    def claim(self, claim_id: str):
        """Look up a claim; an unknown id gets an error that names the real ones, so a model can correct itself."""
        if claim_id not in self.claims:
            raise ValueError(f"unknown claim_id '{claim_id}'. Registered claim_ids: {sorted(self.claims)}")
        return self.claims[claim_id]

    def tested(self, claim_id: str):
        self.claim(claim_id)
        if claim_id not in self.evidence:
            raise ValueError(f"claim '{claim_id}' has not been tested yet: ask the Experimenter 'test {claim_id}' first")
        return self.evidence[claim_id]

    def say(self, msg: str) -> None:
        self.trace.append(msg)
        self.r.emit(msg)

    def recorded(self, agent: str):
        """Keep the exact JSON a specialist's tool returned, so the Orchestrator receives it verbatim instead of the
        specialist model's paraphrase (with a real LLM, paraphrases lose ids such as 'tilt-3')."""
        def wrap(fn):
            @functools.wraps(fn)
            def inner(*a, **kw):
                out = fn(*a, **kw)
                self.last_output[agent] = out
                return out
            return inner
        return wrap

    # -- tools: each is authorised by the mesh and audited ----------------------------------------------------
    def tools_for(self, agent: str) -> list:
        r, mesh, say = self.r, self.mesh, self.say

        @tool
        @self.recorded("scientist")
        def observe_symptom() -> str:
            """Compare the field measurement log with the twin under inventory settings; report the symptom."""
            mesh.authorize("scientist", "observe_symptom")
            say("[scientist] comparing field log with the twin under inventory settings")
            self.symptom = r.scientist.observe(r.seeds)
            r.last_symptom = self.symptom
            self.base = self.symptom["baseline"]
            self.base_scores = [p["score"] for p in self.base["per_seed"]]
            say(f"[scientist] symptom: {self.symptom['symptom_text']}; baseline discrepancy "
                f"{self.base['mean']['score']:.2f}")
            say(f"[scientist] suspicion by cell: {self.symptom['suspicion']}")
            if getattr(r, "vision", None) and self.base.get("heatmap"):
                seen = r.vision.describe(self.base["heatmap"], self.symptom["symptom_text"])
                r.extra["visual_observation"] = seen
                say(f"[scientist] reading the coverage heatmap: {seen}")
            if getattr(r, "nlm", None):        # the network language model, behind the same checkpoint as every model
                nlm_id = getattr(r.nlm, "model_id", "network-lm")
                mesh.checkpoint.enforce("Agent", "scientist", "InvokeModel", nlm_id)
                opinion = r.nlm.second_opinion(self.symptom["symptom_text"], self.symptom["suspicion"], getattr(r, "inventory", None))
                r.extra["network_lm_opinion"] = {"model": nlm_id, "text": opinion}
                mesh.checkpoint.record({"kind": "nlm_opinion", "agent": "scientist", "model": nlm_id, "text": opinion[:500]})
                say(f"[scientist] network language model ({nlm_id}) second opinion: {opinion}")
            out = {"symptom": self.symptom["symptom_text"], "suspicion": self.symptom["suspicion"]}
            if r.extra.get("network_lm_opinion"):
                out["network_lm_opinion"] = r.extra["network_lm_opinion"]
            return json.dumps(out)

        @tool
        @self.recorded("scientist")
        def propose_claims() -> str:
            """Rank suspect cells and write testable claims. Returns JSON with claim_ids and claim texts."""
            mesh.authorize("scientist", "propose_claims")
            claims = r.scientist.propose(self.symptom)
            for c in claims:
                self.claims[c.id] = c
                say(f"[scientist] claim {c.id}: {c.text}")
            mesh.checkpoint.record({"kind": "claims", "agent": "scientist", "claims": [c.id for c in claims],
                              "symptom": self.symptom["symptom_text"], "suspicion": self.symptom["suspicion"]})
            return json.dumps({"claim_ids": [c.id for c in claims],
                               "claims": [{"id": c.id, "kind": c.kind, "cell": c.cell, "text": c.text} for c in claims]})

        @tool
        @self.recorded("experimenter")
        def test_claim(claim_id: str) -> str:
            """Test a claim by sweeping the one setting it names, one value at a time. Args: claim_id."""
            mesh.authorize("experimenter", "test_claim")
            claim = self.claim(claim_id)
            say(f"[experimenter] testing {claim_id} one setting at a time")
            ev = r.experimenter.test(claim, r.seeds, self.base_scores, emit=say)
            self.evidence[claim_id] = ev
            return json.dumps({"claim_id": claim_id, "best": ev.best["label"], "best_score": round(ev.best["mean"], 2),
                               "improvement": round(ev.effect["mean_improvement"], 2)})

        @tool
        @self.recorded("experimenter")
        def run_control(claim_id: str) -> str:
            """Run the same one-setting sweep on a cell nobody suspects. Args: claim_id."""
            mesh.authorize("experimenter", "run_control")
            claim = self.claim(claim_id)
            ccell = r.scientist.control_cell(claim)
            say(f"[experimenter] control run on cell {ccell}")
            self.controls[claim_id] = r.experimenter.control(claim, ccell, r.seeds, self.base_scores, emit=say)
            return json.dumps({"claim_id": claim_id, "control_cell": ccell,
                               "control_improvement": round(self.controls[claim_id].effect["mean_improvement"], 2)})

        @tool
        @self.recorded("experimenter")
        def run_null(claim_id: str) -> str:
            """Test the null hypothesis that nothing is wrong with the network and the twin is just biased on the
            claimed cell (sweeps a constant dB offset). Args: claim_id."""
            mesh.authorize("experimenter", "run_null")
            say(f"[experimenter] null hypothesis: is the twin just biased on cell {self.claim(claim_id).cell}?")
            self.nulls[claim_id] = r.experimenter.null(self.claim(claim_id), r.seeds, self.base_scores, emit=say)
            n = self.nulls[claim_id]
            return json.dumps({"claim_id": claim_id, "best_offset": n.best["label"], "best_score": round(n.best["mean"], 2)})

        @tool
        @self.recorded("experimenter")
        def run_replication(claim_id: str) -> str:
            """Replicate a tilt claim on the second instrument (the simulator's ground-truth tables, not the twin).
            Args: claim_id."""
            mesh.authorize("experimenter", "run_replication")
            say("[experimenter] replicating on the second instrument (simulator tables)")
            ev = self.tested(claim_id)
            self.replications[claim_id] = r.experimenter.replicate(self.claim(claim_id), ev.best, r.holdout_seeds,
                                                                   emit=say)
            rep = self.replications[claim_id]
            return json.dumps({"claim_id": claim_id,
                               "replication_improvement": None if rep is None else round(rep["mean_improvement"], 2)})

        @tool
        @self.recorded("experimenter")
        def run_holdout(claim_id: str) -> str:
            """Re-test the winning setting against inventory on fresh seeds. Args: claim_id."""
            mesh.authorize("experimenter", "run_holdout")
            say("[experimenter] holdout replication")
            ev = self.tested(claim_id)
            self.holdouts[claim_id] = r.experimenter.holdout(self.claim(claim_id), ev.best, r.holdout_seeds, emit=say)
            return json.dumps({"claim_id": claim_id,
                               "holdout_improvement": round(self.holdouts[claim_id]["mean_improvement"], 2)})

        @tool
        @self.recorded("critic")
        def review_claim(claim_id: str) -> str:
            """Critic verdict on a claim given the evidence so far. Returns status needs_control, needs_holdout,
            accepted or rejected. Args: claim_id."""
            mesh.authorize("critic", "review_claim")
            ev = self.tested(claim_id)
            review = r.critic.review(ev, self.controls.get(claim_id), self.holdouts.get(claim_id),
                                     self.nulls.get(claim_id), self.replications.get(claim_id))
            self.reviews[claim_id] = review
            say(f"[critic] {claim_id}: {review.status.upper()} - {review.reason}")
            mesh.checkpoint.record({"kind": "verdict", "agent": "critic", "claim": claim_id, "status": review.status,
                              "reason": review.reason[:300]})
            return json.dumps({"claim_id": claim_id, "status": review.status, "reason": review.reason,
                               "checks": review.checks})

        @tool
        def publish_findings() -> str:
            """Publish finding cards. Refused until every claim has a final Critic verdict."""
            mesh.authorize("orchestrator", "publish_findings")
            if not self.claims:
                raise ValueError("cannot publish: no investigation has been run (no claims on record)")
            unfinished = [i for i in self.claims
                          if self.reviews.get(i) is None or self.reviews[i].status not in ("accepted", "rejected")]
            if unfinished:
                raise ValueError(f"cannot publish: claims without a final Critic verdict: {unfinished}")
            for i, rv in self.reviews.items():
                if rv.status == "accepted" and not (i in self.controls and i in self.nulls and i in self.holdouts):
                    raise ValueError(f"cannot publish {i}: accepted without control, null test and holdout on record")
                if rv.status == "accepted" and self.claims[i].kind == "tilt" and self.replications.get(i) is None:
                    raise ValueError(f"cannot publish {i}: accepted tilt claim without a second-instrument replication")
                if not prereg.verify(self.claims[i].registration):
                    raise ValueError(f"cannot publish {i}: pre-registration missing or altered")
            results = [(self.claims[i], self.evidence[i], self.reviews[i], self.controls.get(i), self.holdouts.get(i),
                        self.nulls.get(i), self.replications.get(i)) for i in self.claims]
            out = r.finalize(results, self.base, self.trace, self.t0, say)
            for card in out["cards"]:
                card["tokens"] = dict(self.tokens)
                card["escalation"] = AgentMesh.escalation_for(card)
                card["run_id"], card["trace_id"] = mesh.run_id, mesh.checkpoint.trace_id
                r.store.put(card)
            self.result = out
            head = out["headline"]
            mesh.checkpoint.record({"kind": "published", "agent": "orchestrator", "headline": head["title"] if head else None,
                              "cards": len(out["cards"]), "experiments": out["experiments"], "tokens": dict(self.tokens),
                              "seconds": round(time.time() - self.t0, 1)})
            return json.dumps({"published": len(out["cards"]), "headline": head["title"] if head else None,
                               "experiments": out["experiments"]})

        return {"scientist": [observe_symptom, propose_claims],
                "experimenter": [test_claim, run_control, run_null, run_holdout, run_replication],
                "critic": [review_claim],
                "orchestrator": [publish_findings]}[agent]


def _ask(agent: Agent, inv: Investigation, caller: str, callee: str, text: str) -> str:
    inv.mesh.authorize(caller, f"ask_{callee}")
    inv.mesh.message(caller, callee, text)
    agent.messages.clear()                  # stateless specialists: keeps token cost flat across many calls
    outputs = getattr(inv, "last_output", {})        # the exploration ledger keeps no verbatim outputs
    outputs.pop(callee, None)
    result = agent(text)
    usage = getattr(getattr(result, "metrics", None), "accumulated_usage", None) or {}
    for k in inv.tokens:
        inv.tokens[k] += int(usage.get(k, 0))
    # the tool's exact JSON when it ran (ids and numbers intact); otherwise the specialist's own words (e.g. an error)
    return outputs.pop(callee, None) or str(result)


def build_investigation_agent(researcher: Researcher, mesh: AgentMesh,
                              model_for: Callable[[str], object]) -> tuple:
    """Return (orchestrator_agent, investigation). `model_for(role)` supplies each agent's model."""
    inv = Investigation(researcher, mesh)
    specialists: Dict[str, Agent] = {
        role: Agent(model=model_for(role), tools=inv.tools_for(role), system_prompt=PROMPTS[role],
                    name=role, callback_handler=None)
        for role in ("scientist", "experimenter", "critic")}

    @tool
    def ask_scientist(request: str) -> str:
        """Ask the Scientist to observe the symptom and propose testable claims."""
        return _ask(specialists["scientist"], inv, "orchestrator", "scientist", request)

    @tool
    def ask_experimenter(task: str) -> str:
        """Ask the Experimenter to 'test <claim_id>', 'control <claim_id>' or 'holdout <claim_id>'."""
        return _ask(specialists["experimenter"], inv, "orchestrator", "experimenter", task)

    @tool
    def ask_critic(task: str) -> str:
        """Ask the Critic to 'review <claim_id>'."""
        return _ask(specialists["critic"], inv, "orchestrator", "critic", task)

    orchestrator = Agent(model=model_for("orchestrator"),
                         tools=[ask_scientist, ask_experimenter, ask_critic] + inv.tools_for("orchestrator"),
                         system_prompt=PROMPTS["orchestrator"], name="orchestrator", callback_handler=None)
    return orchestrator, inv


def run_strands_investigation(researcher: Researcher, mesh: AgentMesh, model_for: Callable[[str], object]) -> dict:
    orchestrator, inv = build_investigation_agent(researcher, mesh, model_for)
    mesh.checkpoint.record({"kind": "request", "agent": "orchestrator",
                            "text": "Investigate the network symptom in the field log and publish findings."})
    inv.say("[orchestrator] starting investigation of the field symptom")
    result = orchestrator("Investigate the network symptom in the field log and publish findings.")
    usage = getattr(getattr(result, "metrics", None), "accumulated_usage", None) or {}
    for k in inv.tokens:
        inv.tokens[k] += int(usage.get(k, 0))
    if inv.result is None:
        raise RuntimeError("orchestrator finished without publishing findings")
    inv.result["tokens"] = inv.tokens
    for card in inv.result["cards"]:
        card["tokens"] = dict(inv.tokens)
        researcher.store.put(card)
    return inv.result
