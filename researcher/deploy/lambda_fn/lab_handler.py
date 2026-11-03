"""AgentCore Gateway Lambda target for the `lab_execute` tool: the emulated lab behind the test automation API's slot.

Telemetry from each campaign is shipped by the executor's sink straight into the ingestion services (Kinesis streams for
KPI windows and alarms, the S3 log lake for interface logs), so the agents receive only the summary.

STATUS: written; LabExecutor is tested locally (scripts/check_research_pipeline.py); this wiring is NOT deployed.
"""
import json
import os

os.environ.setdefault("ATR_RUNS_DIR", "/tmp/runs")

from atr import aws_adapters as aws  # noqa: E402
from atr.faults import Fault  # noqa: E402
from atr.lab import EmulatedLab, LabExecutor  # noqa: E402
from atr.research_cycle import make_ingestion  # noqa: E402
from atr.topology import default_inventory  # noqa: E402
from atr.twin import RFTwin  # noqa: E402

_exec = None


def _executor() -> LabExecutor:
    global _exec
    if _exec is None:
        lab = EmulatedLab(RFTwin.load_or_build(), default_inventory())
        truth = aws.read_truth()
        lab.inject_fault(Fault(truth["kind"], int(truth["cell"]), float(truth.get("value", 0.0))) if truth else None)
        _exec = LabExecutor(lab, sink=make_ingestion().ingest)
    return _exec


def handler(event, context=None):
    """AgentCore Gateway passes the tool arguments as the event."""
    ex = _executor()
    ex.validate(event)
    out = ex.execute(**event)
    return json.loads(json.dumps({"campaign": out["campaign"], "ok": True} if "campaign" in out else out))
