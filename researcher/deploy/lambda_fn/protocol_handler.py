"""Step Functions task Lambda: runs one task of the experiment protocol (atr/protocol.py) in the same container image as
the run_experiment Lambda (it carries the MAVERIC twin and the emulated lab).

Event: {"task": "<name>", "input": <state>}. Heavy per-run context (observation, investigation output, replication results)
lives in S3 (aws_adapters.S3Context) because Step Functions passes at most 256 KB between states. The lab's injected fault
is read from the truth table, which this function's role may read and the agents' role may not.

STATUS: written; the task functions are the ones run locally by `atr.cli research`; the Lambda wiring is NOT deployed.
"""
import os

os.environ.setdefault("ATR_RUNS_DIR", "/tmp/runs")

from atr import aws_adapters as aws  # noqa: E402
from atr.faults import Fault  # noqa: E402
from atr.research_cycle import ResearchCycle  # noqa: E402

_cycle = None


def _rc() -> ResearchCycle:
    global _cycle
    if _cycle is None:
        truth = aws.read_truth()
        fault = Fault(truth["kind"], int(truth["cell"]), float(truth.get("value", 0.0))) if truth else None
        _cycle = ResearchCycle(lab_fault=fault, mode=os.environ.get("ATR_MODE", "aws"), emit=print)
    return _cycle


def _restore(ctx: dict) -> dict:
    if "suspicion" in ctx:
        ctx["suspicion"] = {int(k): float(v) for k, v in ctx["suspicion"].items()}
    return ctx


def handler(event, context=None):
    state = event["input"]
    run_id = state["run_id"]
    store = aws.S3Context()
    rc = _rc()
    rc.ctx = _restore(store.load(run_id))
    result = rc.tasks()[event["task"]](state)
    store.save(run_id, rc.ctx)
    return result
