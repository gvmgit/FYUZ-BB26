"""Exploration as Strands agents: the same orchestrator-plus-three-specialists shape as strands_agents.py.

    Orchestrator --ask_scientist--> Scientist    (scan_network)
                 --ask_experimenter--> Experimenter (test_insight, replicate_insight, holdout_insight)
                 --ask_critic--> Critic          (review_insight)
                 --publish_insights--> finding cards

`publish_insights` is guarded in code: every hypothesis needs a final Critic verdict, an accepted one needs its second
instrument result (tilt) and holdout on record, and every pre-registration must verify.
"""
import json
import time
from typing import Callable

from strands import Agent, tool

from . import prereg
from .anf import AgentMesh
from .explore import Explorer
from .strands_agents import _ask

PROMPTS = {
    "scientist": ("You are the Scientist. Call scan_network once and reply with its JSON unchanged."),
    "experimenter": ("You are the Experimenter. You test network changes on the digital twin through your tools. Requests "
                     "look like 'test <hypothesis_id>', 'replicate <hypothesis_id>' or 'holdout <hypothesis_id>'. Call the "
                     "matching tool once and reply with its JSON unchanged."),
    "critic": ("You are the Critic. Try to disprove findings. For 'review <hypothesis_id>' call review_insight once and reply "
               "with its JSON unchanged. Never accept a hypothesis yourself; only the tool decides."),
    "orchestrator": ("You orchestrate a search for changes worth making. 1) ask_scientist to scan. 2) For each hypothesis: "
                     "ask_experimenter 'test <id>', then ask_critic 'review <id>'. If the Critic answers needs_replication "
                     "ask_experimenter 'replicate <id>'; if needs_holdout ask 'holdout <id>'; then ask the Critic to review "
                     "again, until accepted or rejected. 3) Only then call publish_insights. Never publish early."),
}


class ExplorationRun:
    def __init__(self, explorer: Explorer, mesh: AgentMesh):
        self.x, self.mesh = explorer, mesh
        self.t0 = time.time()
        self.trace = []
        self.hyps, self.evidence, self.reps, self.holds, self.reviews = {}, {}, {}, {}, {}
        self.scanned = False
        self.result = None
        self.tokens = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        base = explorer.emit
        explorer.emit = lambda m: (self.trace.append(m), base(m))

    def say(self, msg: str):
        self.x.emit(msg)

    def tools_for(self, agent: str) -> list:
        x, mesh = self.x, self.mesh

        @tool
        def scan_network() -> str:
            """Scan single-setting interventions on the twin and pre-register the best few hypotheses."""
            mesh.authorize("scientist", "scan_network")
            self.scanned = True
            for h in x.scan():
                self.hyps[h.id] = h
            return json.dumps({"hypothesis_ids": list(self.hyps),
                               "hypotheses": [{"id": h.id, "kind": h.kind, "cell": h.cell, "text": h.text}
                                              for h in self.hyps.values()], "scanned": len(x.scan_table)})

        @tool
        def test_insight(hypothesis_id: str) -> str:
            """Test a hypothesis on the twin with the twin's own error drawn per seed. Args: hypothesis_id."""
            mesh.authorize("experimenter", "test_insight")
            ev = self.evidence[hypothesis_id] = x.test(self.hyps[hypothesis_id])
            g = ev.effects[ev.primary]
            return json.dumps({"hypothesis_id": hypothesis_id, "gain": round(g["mean_improvement"], 2),
                               "unit": g.get("unit", "points")})

        @tool
        def replicate_insight(hypothesis_id: str) -> str:
            """Replicate a tilt hypothesis on the simulator's ground-truth tables (second instrument). Args: hypothesis_id."""
            mesh.authorize("experimenter", "replicate_insight")
            rep = self.reps[hypothesis_id] = x.replicate(self.hyps[hypothesis_id])
            return json.dumps({"hypothesis_id": hypothesis_id,
                               "second_instrument_gain_pts": None if rep is None else round(rep["mean_improvement"], 2)})

        @tool
        def holdout_insight(hypothesis_id: str) -> str:
            """Re-test on fresh seeds under a harsher twin error. Args: hypothesis_id."""
            mesh.authorize("experimenter", "holdout_insight")
            hold = self.holds[hypothesis_id] = x.holdout(self.hyps[hypothesis_id])
            return json.dumps({"hypothesis_id": hypothesis_id, "holdout_gain_pts": round(hold["mean_improvement"], 2)})

        @tool
        def review_insight(hypothesis_id: str) -> str:
            """Critic verdict: needs_replication, needs_holdout, accepted or rejected. Args: hypothesis_id."""
            mesh.authorize("critic", "review_insight")
            rv = x.critic.review(self.evidence[hypothesis_id], self.reps.get(hypothesis_id), self.holds.get(hypothesis_id))
            self.reviews[hypothesis_id] = rv
            self.say(f"[critic] {hypothesis_id}: {rv.status.upper()} - {rv.reason}")
            return json.dumps({"hypothesis_id": hypothesis_id, "status": rv.status, "reason": rv.reason,
                               "checks": rv.checks})

        @tool
        def publish_insights() -> str:
            """Publish insight cards. Refused until every hypothesis has a final Critic verdict."""
            mesh.authorize("orchestrator", "publish_insights")
            if not self.hyps:
                if not self.scanned:
                    raise ValueError("cannot publish: no hypotheses on record (scan first)")
                # the scan ran and nothing cleared the bar: that is a result, published as an empty set
                self.result = x.publish([], self.trace, self.t0)
                return json.dumps({"published": 0, "headline": None, "experiments": self.result["experiments"]})
            unfinished = [i for i in self.hyps if self.reviews.get(i) is None
                          or self.reviews[i].status not in ("accepted", "rejected")]
            if unfinished:
                raise ValueError(f"cannot publish: hypotheses without a final Critic verdict: {unfinished}")
            for i, rv in self.reviews.items():
                h = self.hyps[i]
                if rv.status == "accepted":
                    if i not in self.holds or (h.kind == "tilt" and self.reps.get(i) is None):
                        raise ValueError(f"cannot publish {i}: accepted without its holdout / second-instrument result")
                if not prereg.verify(h.registration):
                    raise ValueError(f"cannot publish {i}: pre-registration missing or altered")
            results = [(self.hyps[i], self.evidence[i], self.reviews[i], self.reps.get(i), self.holds.get(i))
                       for i in self.hyps]
            self.result = x.publish(results, self.trace, self.t0)
            for card in self.result["cards"]:
                card["tokens"] = dict(self.tokens)
                x.store.put(card)
            head = self.result["headline"]
            return json.dumps({"published": len(self.result["cards"]), "headline": head["title"] if head else None,
                               "experiments": self.result["experiments"]})

        return {"scientist": [scan_network], "experimenter": [test_insight, replicate_insight, holdout_insight],
                "critic": [review_insight], "orchestrator": [publish_insights]}[agent]


def run_strands_exploration(explorer: Explorer, mesh: AgentMesh, model_for: Callable[[str], object]) -> dict:
    run = ExplorationRun(explorer, mesh)
    specialists = {role: Agent(model=model_for(role), tools=run.tools_for(role), system_prompt=PROMPTS[role],
                               name=role, callback_handler=None) for role in ("scientist", "experimenter", "critic")}

    @tool
    def ask_scientist(request: str) -> str:
        """Ask the Scientist to scan the network for changes worth making."""
        return _ask(specialists["scientist"], run, "orchestrator", "scientist", request)

    @tool
    def ask_experimenter(task: str) -> str:
        """Ask the Experimenter to 'test <id>', 'replicate <id>' or 'holdout <id>'."""
        return _ask(specialists["experimenter"], run, "orchestrator", "experimenter", task)

    @tool
    def ask_critic(task: str) -> str:
        """Ask the Critic to 'review <id>'."""
        return _ask(specialists["critic"], run, "orchestrator", "critic", task)

    orchestrator = Agent(model=model_for("orchestrator"),
                         tools=[ask_scientist, ask_experimenter, ask_critic] + run.tools_for("orchestrator"),
                         system_prompt=PROMPTS["orchestrator"], name="orchestrator", callback_handler=None)
    run.say("[orchestrator] starting a search for network changes worth making")
    result = orchestrator("Search the network for changes worth making and publish what survives.")
    usage = getattr(getattr(result, "metrics", None), "accumulated_usage", None) or {}
    for k in run.tokens:
        run.tokens[k] += int(usage.get(k, 0))
    if run.result is None:
        raise RuntimeError("orchestrator finished without publishing")
    run.result["tokens"] = run.tokens
    for card in run.result["cards"]:
        card["tokens"] = dict(run.tokens)
        explorer.store.put(card)
    return run.result
