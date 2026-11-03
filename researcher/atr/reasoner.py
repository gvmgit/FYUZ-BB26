"""Reasoner: the one place an LLM is used to put findings into words.

All scientific decisions (what to test, what counts as evidence) are made by code in agents.py, so the system
works offline and results are reproducible. On AWS, implement `explain` with Claude on Bedrock (Strands model
provider) and pass the heatmap PNGs as image content; the agents do not change.
"""
from typing import Protocol


class Reasoner(Protocol):
    def explain(self, role: str, facts: dict) -> str: ...


class TemplateReasoner:
    """Deterministic, offline wording."""

    def explain(self, role: str, facts: dict) -> str:
        if role == "claim":
            return (f"Cell {facts['cell']}: {facts['what']}. Suspected because {facts['why']}. "
                    f"Testable: changing only this setting should lower the field-vs-twin discrepancy.")
        if role == "finding":
            return (f"{facts['headline']} Evidence: discrepancy score fell from {facts['base']:.2f} to "
                    f"{facts['best']:.2f} (95% CI of improvement {facts['ci'][0]:.2f} to {facts['ci'][1]:.2f}, "
                    f"d={facts['d']:.1f}); a control cell showed no comparable gain and a held-out resample "
                    f"confirmed it.")
        return str(facts)
