"""Engineer verdicts on finding cards, and what the researcher learns from them.

An engineer can accept a finding, reject it, or ask for another experiment. Verdicts are stored on the card (so they
travel with it to DynamoDB), audited, and used three ways:

  priors     - accept/reject counts per fault kind become Beta(accepts+1, rejects+1) weights. They REORDER what the
               Scientist tries first and which exploration hypotheses get tested first. They never skip a claim:
               the Critic still gates everything, so a prior can change the order of work but not the evidence needed.
  follow-ups - 'request experiment' puts a job on the scheduler; `run_followups` re-tests the card's claim on fresh
               seeds (and the second instrument for tilt) and attaches the result to the card.
  training   - accepted and rejected findings become labelled records for the language-model dataset.

This is distinct from `approve` (atr.cli approve): `approve` authorises a proposed ACTION; a verdict judges the FINDING.
"""
import time
from typing import Dict, List, Optional

VERDICTS = ("accept", "reject", "request_experiment")


def record_verdict(store, card_id: str, verdict: str, by: str, comment: str = "", request: str = "",
                   scheduler=None, audit=None) -> dict:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    if not by:
        raise PermissionError("a verdict needs the name of the engineer giving it")
    card = store.get(card_id)
    if card is None:
        raise KeyError(f"no card {card_id}")
    entry = {"verdict": verdict, "by": by, "comment": comment, "request": request, "at": time.time()}
    card.setdefault("engineer_verdicts", []).append(entry)
    if verdict in ("accept", "reject"):
        card["engineer_status"] = verdict
    else:
        card["engineer_status"] = card.get("engineer_status", "follow_up_requested")
        if scheduler is not None:
            card.setdefault("follow_ups", []).append({"status": "queued", "request": request, "at": time.time()})
            scheduler.submit({"job": "follow_up", "card_id": card_id, "request": request}, priority=1)
    store.put(card)
    if audit is not None:
        audit.write({"kind": "engineer_verdict", "card": card_id, "verdict": verdict, "by": by})
    return card


class Priors:
    """Beta(accepts+1, rejects+1) weight per fault kind, from the latest accept/reject on each card."""

    def __init__(self, counts: Optional[Dict[str, List[int]]] = None):
        self.counts = counts or {}

    def weight(self, kind: str) -> float:
        a, r = self.counts.get(kind, [0, 0])
        return (a + 1) / (a + r + 2)

    def order_kinds(self, kinds: List[str]) -> List[str]:
        """Highest-weight kind first; ties keep the Scientist's own order (sorted() is stable)."""
        return sorted(kinds, key=lambda k: -self.weight(k))

    def n_verdicts(self) -> int:
        return sum(a + r for a, r in self.counts.values())

    def describe(self) -> Dict[str, dict]:
        return {k: {"accepted": a, "rejected": r, "weight": round(self.weight(k), 3)}
                for k, (a, r) in sorted(self.counts.items())}


def priors_from(store) -> Priors:
    counts: Dict[str, List[int]] = {}
    for card in store.list():
        status = card.get("engineer_status")
        if status in ("accept", "reject"):
            kind = card["claim"]["kind"]
            counts.setdefault(kind, [0, 0])[0 if status == "accept" else 1] += 1
    return Priors(counts)


def run_followups(store, scheduler, run_one, audit=None) -> List[dict]:
    """Work through queued follow-up jobs. `run_one(card, request) -> dict` does the re-test."""
    out = []

    def handle(job):
        if job.get("job") != "follow_up":
            return {"skipped": job}
        card = store.get(job["card_id"])
        if card is None:
            return {"error": f"no card {job['card_id']}"}
        result = run_one(card, job.get("request", ""))
        card.setdefault("follow_ups", []).append({"status": "done", "request": job.get("request", ""),
                                                  "result": result, "at": time.time()})
        store.put(card)
        if audit is not None:
            audit.write({"kind": "follow_up_done", "card": job["card_id"]})
        return result

    for item in scheduler.run_all(handle):
        out.append(item)
    return out
