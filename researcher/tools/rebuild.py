"""Rebuild: reprint the reported Level 1 results from evidence/ with no AWS access. Exits 1 if anything changed.

    python tools/rebuild.py

It checks that every file matches evidence/MANIFEST.sha256, that each run's frozen test set still has the hash its
declaration names, that each declaration was written before its record, and it re-grades every run from its per-item
records with the same rule-based grader, comparing the result with the stored record and with evidence/results.json.
"""
import json
import sys

from _common import ROOT  # noqa: F401  (import path)

from atr import grader  # noqa: E402
from atr.evidence import Evidence, sha256_file  # noqa: E402


def main() -> int:
    ev = Evidence(bucket="")
    problems = [f"manifest: {p}" for p in ev.verify_manifest()]
    results = json.loads((ev.root / "results.json").read_text())
    regraded = {}
    print(f"{'run id':<44} {'unit':<10} {'metric':<28} {'value':>7}  target  result")
    for run_id in ev.run_ids():
        decl = ev.load_declared(run_id)
        try:
            rec = ev.load_record(run_id)
        except FileNotFoundError:
            problems.append(f"{run_id}: declared but no record (a run with no result)")
            continue
        if decl["declared_at"] > rec["recorded_at"]:
            problems.append(f"{run_id}: declaration written after the record")
        if decl.get("test_set") and sha256_file(ev.root / "frozen" / decl["test_set"]) != decl["test_set_sha256"]:
            problems.append(f"{run_id}: frozen test set {decl['test_set']} changed since the declaration")
        again = grader.GRADERS[decl["unit"]](ev.load_items(run_id), decl["target"])
        if again["value"] != rec["result"]["value"] or again["pass"] != rec["pass"]:
            problems.append(f"{run_id}: re-grading gives {again['value']}/{again['pass']}, record says "
                            f"{rec['result']['value']}/{rec['pass']}")
        regraded[run_id] = again
        print(f"{run_id:<44} {decl['unit']:<10} {again['metric']:<28} {again['value']:>7}  {decl['target']['value']:>6}  "
              f"{'PASS' if again['pass'] else 'FAIL'}")
    print("\nReported Level 1 results (evidence/results.json):")
    for unit, r in sorted(results["units"].items()):
        mine = regraded.get(r["run_id"])
        same = mine is not None and mine["value"] == r["value"] and mine["pass"] == r["pass"]
        if not same:
            problems.append(f"results.json {unit}: does not match the re-graded run {r['run_id']}")
        print(f"  {unit:<10} {r['metric']:<28} {r['value']:>7}  {'PASS' if r['pass'] else 'FAIL'}  run {r['run_id']}"
              f"{'' if same else '   <-- MISMATCH'}")
    if problems:
        print("\nDATA CHANGED OR INCOMPLETE:\n  " + "\n  ".join(problems))
        return 1
    print("\nrebuild OK: evidence unchanged, every result reproduces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
