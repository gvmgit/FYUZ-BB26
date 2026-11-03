"""Audit trail sink. Every tool call and agent-to-agent request is recorded (traceability barrier).

Local: JSON lines on disk. AWS: DynamoDB table (see aws_adapters.DynamoAudit) - same `write` interface.
"""
import json
import time
from typing import Protocol

from . import config


class AuditSink(Protocol):
    def write(self, record: dict) -> None: ...


class FileAudit:
    def __init__(self, path=config.AUDIT_LOG):
        self.path = path

    def write(self, record: dict) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
