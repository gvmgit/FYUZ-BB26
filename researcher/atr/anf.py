"""Agent mesh: agent-to-agent messages, the checkpoint in front of every call, and human-in-the-loop escalation.

The mesh holds no policy. Every tool call and agent-to-agent request goes to the checkpoint (atr/checkpoint.py), which
asks the policy decision point - Amazon Verified Permissions in AWS mode, the same policies/atr.cedar evaluated
locally offline - and writes a decision record with the run id and trace id. A denied call raises PolicyViolation.
"""
from typing import Optional

from .audit import AuditSink, FileAudit
from .checkpoint import Checkpoint, PolicyViolation, local_checkpoint  # noqa: F401  (PolicyViolation re-exported)


class AgentMesh:
    def __init__(self, audit: Optional[AuditSink] = None, checkpoint: Optional[Checkpoint] = None):
        self.audit = audit or FileAudit()
        self.checkpoint = checkpoint or local_checkpoint(self.audit)
        if self.checkpoint.audit is None:
            self.checkpoint.audit = self.audit

    @property
    def run_id(self) -> str:
        return self.checkpoint.run_id

    def authorize(self, agent: str, action: str) -> None:
        self.checkpoint.enforce("Agent", agent, action)

    def message(self, sender: str, receiver: str, summary: str) -> None:
        self.checkpoint.record({"kind": "agent_message", "from": sender, "to": receiver, "summary": summary[:300]})

    @staticmethod
    def escalation_for(card: dict) -> dict:
        """Human-in-the-loop: agents never change the live network. A finding becomes a proposed action that
        waits for a named person; ambiguous or unconfirmed results are escalated for review instead."""
        c = card["claim"]
        if card.get("type") == "network_insight":
            if card["review"]["status"] == "accepted" and c["kind"] == "handover_rule":
                return {"status": "awaiting_human_approval",
                        "proposed_action": (f"Field trial: change the handover "
                                            f"{ {'hyst': 'hysteresis (dB)', 'ttt': 'time-to-trigger (ticks)'}.get(c['knob'], c['knob']) } "
                                            f"from {c['from']:g} to {c['value']:g} in a limited area, re-measure handovers, "
                                            "ping-pongs and radio-link failures, and roll back if either failures or "
                                            "ping-pongs rise"),
                        "reason": "validated on the twin only; a person must approve any change to the live network"}
            if card["review"]["status"] == "accepted":
                return {"status": "awaiting_human_approval",
                        "proposed_action": (f"Field trial: set cell {c['cell']} {c['kind']} from {c['from']:g} to "
                                            f"{c['value']:g} degrees, re-measure failure risk, coverage and SINR, and roll "
                                            "back if either coverage or SINR gets worse"),
                        "reason": "validated on the twin only; a person must approve any change to the live network"}
            return {"status": "needs_human_review", "proposed_action": None,
                    "reason": "the hypothesis did not survive the Critic; shown for transparency"}
        alert = card.get("twin_calibration_alert")
        if alert:
            return {"status": "needs_human_review",
                    "proposed_action": (f"Run a calibration survey on cell {alert['cell']} and fine-tune the twin "
                                        f"(estimated twin offset {alert['estimated_offset_db']:+g} dB)"),
                    "reason": "the Critic found the twin, not the network, is the likelier explanation"}
        if card["verdict"] == "confirmed" and not card.get("alternatives_not_excluded"):
            action = ("Dispatch a field engineer to verify cell %s %s (found %s)" % (c["cell"], c["kind"].replace("_", " "), c["estimate"]))
            return {"status": "awaiting_human_approval", "proposed_action": action,
                    "reason": "critical-operation boundary: no automatic change to the live network"}
        return {"status": "needs_human_review", "proposed_action": None,
                "reason": "result is ambiguous or unconfirmed; a person should decide next steps"}
