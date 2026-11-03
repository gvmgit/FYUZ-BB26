"""Scheduler: queue and prioritise investigations (the architecture's AgentCore Scheduler role).

Local: priority queue, optionally persisted to a JSON file so a job queued by one command (an engineer asking for another
experiment) is still there when a worker runs later. Lower number runs first. AWS: SQS (aws_adapters.SqsScheduler) fed by
the nightly EventBridge rule in the CDK stack and by `atr.cli schedule`; a worker consumes the queue.
"""
import heapq
import itertools
import json
from pathlib import Path
from typing import Callable, List, Optional


class LocalScheduler:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._heap, self._n = [], itertools.count()
        if self.path and self.path.exists():
            for item in json.loads(self.path.read_text() or "[]"):
                heapq.heappush(self._heap, (item["priority"], next(self._n), item["job"]))

    def _save(self):
        if self.path:
            self.path.write_text(json.dumps([{"priority": p, "job": j} for p, _, j in sorted(self._heap)], indent=1))

    def submit(self, job: dict, priority: int = 5) -> str:
        n = next(self._n)
        heapq.heappush(self._heap, (priority, n, job))
        self._save()
        return f"local-{n}"

    def pending(self) -> int:
        return len(self._heap)

    def run_all(self, handler: Callable[[dict], dict]) -> List[dict]:
        out = []
        while self._heap:
            priority, _, job = heapq.heappop(self._heap)
            self._save()
            out.append({"priority": priority, "job": job, "result": handler(job)})
        return out
