"""The evidence chain: a target declared before each run, one run id end to end, per-item outputs, a record that says
pass or fail, and a hash manifest so the rebuild can prove nothing changed afterwards.

Layout (in the repository under evidence/, mirrored to s3://$ATR_EVIDENCE_BUCKET/evidence/):
    frozen/<test set>.json         frozen before any run; its sha256 goes into every declaration that uses it
    declared/<run_id>.json         written (and uploaded) before the run starts
    items/<run_id>.jsonl           one line per item, keyed by run_id and item index
    records/<run_id>.json          the grader's verdict on that run against the declared target
    results.json                   the reported Level 1 results, one entry per unit
    MANIFEST.sha256                sha256 of every file above
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1] / "evidence"
PREFIX = "evidence/"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Evidence:
    def __init__(self, root: Optional[Path] = None, bucket: Optional[str] = None, s3=None, region: Optional[str] = None):
        self.root = Path(root or ROOT)
        self.bucket = bucket if bucket is not None else os.environ.get("ATR_EVIDENCE_BUCKET")
        self._s3, self.region = s3, region or os.environ.get("ATR_REGION") or "us-east-2"

    @property
    def s3(self):
        """A client per use unless one was injected: a long run outlives short-lived credentials."""
        if self._s3 is not None:
            return self._s3
        import boto3
        return boto3.client("s3", region_name=self.region)

    # ---- writing
    def _write(self, rel: str, text: str) -> str:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        if self.bucket:
            self.s3.put_object(Bucket=self.bucket, Key=PREFIX + rel, Body=text.encode("utf-8"),
                               ContentType="application/json")
        return f"s3://{self.bucket}/{PREFIX}{rel}" if self.bucket else str(path)

    def declare(self, run_id: str, unit: str, target: dict, grader: str, model_id: Optional[str] = None,
                test_set: Optional[str] = None, extra: Optional[dict] = None) -> dict:
        """Write the declaration before the run: the target is a prediction, not a result chosen afterwards."""
        if (self.root / "declared" / f"{run_id}.json").exists():
            raise FileExistsError(f"run {run_id} is already declared")
        decl = {"run_id": run_id, "unit": unit, "target": target, "grader": grader, "model_id": model_id,
                "declared_at": now(), **(extra or {})}
        if test_set:
            decl.update(test_set=test_set, test_set_sha256=sha256_file(self.root / "frozen" / test_set))
        decl["location"] = self._write(f"declared/{run_id}.json", json.dumps(decl, indent=2))
        return decl

    def items(self, run_id: str, items: Iterable[dict]) -> List[dict]:
        rows = []
        for i, row in enumerate(items):
            row = dict(row)
            if row.get("run_id") not in (None, run_id):     # e.g. a rebuilt trace's records belong to the run they trace
                row["source_run_id"] = row.pop("run_id")
            rows.append({"run_id": run_id, "item": i, **row})
        self._write(f"items/{run_id}.jsonl", "".join(json.dumps(r, default=str) + "\n" for r in rows))
        return rows

    def record(self, run_id: str, record: dict) -> dict:
        record = {"run_id": run_id, "recorded_at": now(), **record}
        self._write(f"records/{run_id}.json", json.dumps(record, indent=2, default=str))
        return record

    def results(self, results: dict) -> None:
        self._write("results.json", json.dumps(results, indent=2, sort_keys=True))

    def manifest(self) -> str:
        files = sorted(p for p in self.root.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
        text = "".join(f"{sha256_file(p)}  {p.relative_to(self.root).as_posix()}\n" for p in files)
        self._write("MANIFEST.sha256", text)
        return text

    # ---- reading (no AWS)
    def load_declared(self, run_id: str) -> dict:
        return json.loads((self.root / "declared" / f"{run_id}.json").read_text(encoding="utf-8"))

    def load_items(self, run_id: str) -> List[dict]:
        path = self.root / "items" / f"{run_id}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load_record(self, run_id: str) -> dict:
        return json.loads((self.root / "records" / f"{run_id}.json").read_text(encoding="utf-8"))

    def run_ids(self) -> List[str]:
        return sorted(p.stem for p in (self.root / "declared").glob("*.json"))

    def verify_manifest(self) -> List[str]:
        """Files whose bytes differ from the manifest, or that are missing / not listed."""
        listed = {}
        for line in (self.root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
            digest, rel = line.split("  ", 1)
            listed[rel] = digest
        bad = [rel for rel, d in listed.items() if not (self.root / rel).exists() or sha256_file(self.root / rel) != d]
        present = {p.relative_to(self.root).as_posix() for p in self.root.rglob("*")
                   if p.is_file() and p.name != "MANIFEST.sha256"}
        return bad + sorted(f"unlisted: {r}" for r in present - set(listed))
