"""Orchestrator: runs Scientist -> Experimenter -> Critic and publishes finding cards.

Local stand-in for the AgentCore scheduler/interpreter. The Critic gates publication: a claim is only
marked 'confirmed' after a control run and a held-out replication.
"""
import time
import uuid
from typing import Callable, List

from . import agents, config, prereg, stats
from .anf import AgentMesh
from .audit import FileAudit
from .reasoner import Reasoner, TemplateReasoner
from .store import FindingStore

H2H_MIN_GAP = 0.3   # score units the winner must beat a rival by (CI lower bound) to rule it out
ENGINEER_MINUTES_PER_MANUAL_RUN = 20   # ASSUMPTION for the demo comparison; not measured


class Researcher:
    def __init__(self, gateway, runner, inventory, store: FindingStore, reasoner: Reasoner = None,
                 emit: Callable[[str], None] = print, seeds=range(1, 9), holdout_seeds=range(101, 109),
                 audit=None, priors=None):
        self.gw, self.runner, self.inventory, self.store = gateway, runner, inventory, store
        self.reasoner = reasoner or TemplateReasoner()
        self.emit = emit
        self.audit = audit or FileAudit()
        self.seeds, self.holdout_seeds = list(seeds), list(holdout_seeds)
        self.extra: dict = {}     # extra fields copied onto every card (e.g. network-LM second opinion)
        self.field_log = None     # path of the saved field log, so a follow-up can re-test the same evidence
        self.scientist = agents.Scientist(gateway, inventory, self.reasoner, audit=self.audit, priors=priors)
        self.experimenter = agents.Experimenter(gateway, inventory)
        self.critic = agents.Critic()

    def investigate(self) -> dict:
        t0 = time.time()
        trace: List[str] = []

        def say(msg):
            trace.append(msg)
            self.emit(msg)

        say("[scientist] comparing field log with the twin under inventory settings")
        symptom = self.scientist.observe(self.seeds)
        self.last_symptom = symptom      # kept for dataset export
        base = symptom["baseline"]
        base_scores = [p["score"] for p in base["per_seed"]]
        say(f"[scientist] symptom: {symptom['symptom_text']}; baseline discrepancy {base['mean']['score']:.2f}")
        say(f"[scientist] suspicion by cell: {symptom['suspicion']}")
        claims = self.scientist.propose(symptom)

        results = []
        for claim in claims:
            say(f"[scientist] claim {claim.id}: {claim.text}")
            say(f"[experimenter] testing {claim.id} one setting at a time")
            ev = self.experimenter.test(claim, self.seeds, base_scores, emit=say)
            review = self.critic.review(ev)
            control = holdout = null = replication = None
            while review.status in ("needs_control", "needs_null", "needs_holdout", "needs_replication"):
                say(f"[critic] {review.reason}")
                if review.status == "needs_control":
                    ccell = self.scientist.control_cell(claim)
                    say(f"[experimenter] control run on cell {ccell}")
                    control = self.experimenter.control(claim, ccell, self.seeds, base_scores, emit=say)
                elif review.status == "needs_null":
                    say(f"[experimenter] null hypothesis: is the twin just biased on cell {claim.cell}?")
                    null = self.experimenter.null(claim, self.seeds, base_scores, emit=say)
                elif review.status == "needs_replication":
                    say("[experimenter] replicating on the second instrument (simulator tables)")
                    replication = self.experimenter.replicate(claim, ev.best, self.holdout_seeds, emit=say)
                else:
                    say("[experimenter] holdout replication")
                    holdout = self.experimenter.holdout(claim, ev.best, self.holdout_seeds, emit=say)
                review = self.critic.review(ev, control, holdout, null, replication)
            say(f"[critic] {claim.id}: {review.status.upper()} - {review.reason}")
            results.append((claim, ev, review, control, holdout, null, replication))
        return self.finalize(results, base, trace, t0, say)

    def finalize(self, results, base, trace, t0, say) -> dict:
        """Rank accepted claims, rule out rivals head-to-head, render heatmaps, store the finding cards."""
        if self.field_log is None and self.runner is not None:
            # keep the field log with the published cards so an engineer's follow-up can re-test the same evidence
            log = config.RUNS / "field" / f"field_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}.csv"
            log.parent.mkdir(parents=True, exist_ok=True)
            self.runner.obs.to_csv(log, index=False)
            self.field_log = str(log)
        for r in results:   # a result is only published against the hypothesis that was sealed before it was tested
            if not prereg.verify(r[0].registration):
                raise ValueError(f"claim {r[0].id}: pre-registration missing or altered; refusing to publish")
        accepted = sorted([r for r in results if r[2].status == "accepted"],
                          key=lambda r: r[1].effect["mean_improvement"], reverse=True)
        # Head-to-head: a rival that also passed review is only kept as an alternative if its best fit is
        # statistically indistinguishable from the winner's (paired over the same seeds).
        h2h = {}
        if accepted:
            win = accepted[0][1]
            for r in accepted[1:]:
                eff = stats.paired_effect(r[1].best["per_seed"], win.best["per_seed"])
                h2h[r[0].id] = {"winner_better_by": eff["mean_improvement"], "ci95": eff["ci95"],
                                "ruled_out": eff["ci95"][0] > H2H_MIN_GAP}
                say(f"[critic] head-to-head {accepted[0][0].id} vs {r[0].id}: winner better by "
                    f"{eff['mean_improvement']:.2f} (CI {eff['ci95'][0]:.2f}..{eff['ci95'][1]:.2f}) -> "
                    f"{'rival ruled out' if h2h[r[0].id]['ruled_out'] else 'rival NOT excluded'}")
        wall = time.time() - t0
        headline = None
        cards = []
        for claim, ev, review, control, holdout, null, replication in results:
            is_head = bool(accepted) and claim is accepted[0][0]
            heatmap_fix = None
            if is_head:
                change = {"tilt": {"tilt": ev.best["value"]}, "azimuth": {"azimuth": ev.best["value"]},
                          "cell_off": {"enabled": False}}[claim.kind]
                fixed = self.gw.invoke("run_experiment", {"config": {"name": f"best fix: {ev.best['label']}",
                                       "cells": {str(claim.cell): change}, "render": True},
                                       "seeds": self.seeds[:1]}, agent="experimenter")
                heatmap_fix = fixed.get("heatmap")
            card = self._card(claim, ev, review, control, holdout, null, replication, is_head, base, heatmap_fix, h2h)
            cards.append(card)
        total = self.gw.count
        for card in cards:
            card["run_summary"] = {
                "experiments_run": total, "wall_seconds": round(time.time() - t0, 1),
                "engineer_estimate_hours": round(total * ENGINEER_MINUTES_PER_MANUAL_RUN / 60, 1),
                "engineer_estimate_note": f"assumes {ENGINEER_MINUTES_PER_MANUAL_RUN} min per manual what-if run "
                                          "(assumption, not measured)"}
            card["trace"] = trace
            card["escalation"] = AgentMesh.escalation_for(card)   # human-in-the-loop on every path
            card["type"] = "fault_finding"
            card["field_log"] = self.field_log
            sym = getattr(self, "last_symptom", None)
            if sym is not None:        # what the Scientist saw, so an engineer's verdict can become a training label
                from .finetune_data import diagnostics_view
                card["diagnostics"] = diagnostics_view(sym, self.inventory)
            if self.scientist.priors is not None:
                card["priors_used"] = self.scientist.priors.describe()
            card.update(self.extra)
            self.store.put(card)
            if card["headline"]:
                headline = card
        say(f"[critic] done: {total} experiments in {time.time() - t0:.0f}s; "
            f"{'finding published: ' + headline['title'] if headline else 'no claim survived'}")
        return {"headline": headline, "cards": cards, "experiments": total, "wall_seconds": wall}

    def _card(self, claim, ev, review, control, holdout, null, replication, is_head, base, heatmap_fix, h2h):
        verdict = ("confirmed" if is_head else
                   ("ruled out (head-to-head)" if h2h.get(claim.id, {}).get("ruled_out")
                    else "alternative (not excluded)") if review.status == "accepted" else
                   "refuted" if review.status == "rejected" else "inconclusive")
        if claim.kind == "cell_off":
            estimate = "cell is off"
        else:
            inv_val = self.inventory[claim.cell]["azimuth" if claim.kind == "azimuth" else "tilt"]
            estimate = f"{ev.best['value']:g} deg (inventory {inv_val:g} deg)"
        card = {
            "id": f"{time.strftime('%Y%m%d-%H%M%S')}-{claim.id}-{uuid.uuid4().hex[:4]}",
            "created": time.time(),
            "title": f"Cell {claim.cell}: {claim.kind.replace('_', ' ')} {('≈ ' + estimate) if verdict == 'confirmed' else '(' + verdict + ')'}",
            "headline": is_head,
            "verdict": verdict,
            "claim": {"kind": claim.kind, "cell": claim.cell, "text": claim.text, "estimate": estimate,
                      "value": ev.best["value"]},
            "symptom": {"baseline_score": base["mean"]["score"], "baseline_rlf_risk_pct": base["mean"]["rlf_risk_pct"],
                        "baseline_coverage_pct": base["mean"]["coverage_pct"]},
            "evidence": {"best": ev.best["label"], "effect": ev.effect,
                         "sweep": [{k: s[k] for k in ("label", "value", "mean", "run_id")} for s in ev.sweep]},
            "control": ({"best": control.best["label"], "effect": control.effect} if control else None),
            "holdout": holdout,
            "checks": review.checks,
            "review": {"status": review.status, "reason": review.reason},
            "heatmaps": {"inventory": base.get("heatmap"), "best_fix": heatmap_fix},
        }
        reg = claim.registration
        card["preregistration"] = {
            "hash": reg["hash"], "registered_at": reg["registered_at"], "verified": prereg.verify(reg),
            "predicted": reg["statement"]["predicted_effect"], "null_outcome": reg["statement"]["null_outcome"],
            "design": reg["statement"]["design"],
            "prediction_met": review.status == "accepted"}   # accepted = every predicted threshold was met
        if replication is not None:
            card["second_instrument"] = {"instrument": "simulator ground-truth tables", "effect": replication}
        if null:
            card["null_hypothesis"] = {"hypothesis": f"nothing is wrong with the network; the twin is biased on cell {claim.cell}",
                                       "best": null.best["label"], "best_offset_db": null.best["value"],
                                       "best_score": null.best["mean"]}
            if review.status == "rejected" and "twin offset" in review.reason:
                card["twin_calibration_alert"] = {
                    "cell": claim.cell, "estimated_offset_db": null.best["value"],
                    "message": (f"The field is explained as well by cell {claim.cell} reading {null.best['value']:+g} dB "
                                "versus the twin as by a setting change: the twin is probably mis-calibrated there. "
                                "Take a calibration survey and fine-tune the twin before acting on this cell.")}
        if is_head:
            card["narrative"] = self.reasoner.explain("finding", {
                "headline": f"{claim.id}: {estimate}.",
                "base": float(sum(ev.baseline_per_seed) / len(ev.baseline_per_seed)),
                "best": ev.best["mean"], "ci": ev.effect["ci95"], "d": ev.effect["cohens_d"]})
            card["alternatives_not_excluded"] = [i for i, v in h2h.items() if not v["ruled_out"]]
            card["ruled_out_head_to_head"] = [i for i, v in h2h.items() if v["ruled_out"]]
            card["head_to_head"] = h2h
        return card
