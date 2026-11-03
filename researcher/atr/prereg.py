"""Pre-registration: a hypothesis is sealed before any experiment on it runs.

The statement carries the claim, the experiment design, the predicted effect with its thresholds, and the null outcome.
It is hashed (SHA-256 over canonical JSON, including the registration time) and written to the audit trail at the moment
of registration. Later changes to any field, including the thresholds, make `verify` fail, so a result cannot be quietly
re-judged against thresholds chosen after seeing it. The hash is evidence of order, not of identity: anyone with write
access to the store could still replace a whole card, which is why the audit record is written separately.
"""
import hashlib
import json
import time
from typing import Optional


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def register(statement: dict, audit=None) -> dict:
    body = {"statement": statement, "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    digest = hashlib.sha256(_canon(body).encode()).hexdigest()
    if audit is not None:
        audit.write({"kind": "preregistration", "claim": statement.get("id"), "hash": digest})
    return {**body, "hash": digest}


def verify(registration: Optional[dict]) -> bool:
    if not registration:
        return False
    body = {"statement": registration["statement"], "registered_at": registration["registered_at"]}
    return hashlib.sha256(_canon(body).encode()).hexdigest() == registration.get("hash")
