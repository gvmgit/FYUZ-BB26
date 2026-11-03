"""The pass/fail check on the records: one line per Track 3 unit, exit 0 only if every reported unit is complete.

    python tools/check_units.py

A unit is complete with all four parts: (1) a target declared before the run, (2) output produced by this repository's
code (the run's items and record exist and were written by tools/track3_units.py), (3) per-item records keyed to one run
id, (4) this check returning pass. No AWS access.
"""
import json
import sys

from _common import ROOT  # noqa: F401

from atr.evidence import Evidence  # noqa: E402

UNITS = ("allow", "deny", "pause", "trace", "diagnosis", "nlm_tuning")


def main() -> int:
    ev = Evidence(bucket="")
    units = json.loads((ev.root / "results.json").read_text())["units"]
    ok_all = True
    for unit in UNITS:
        r = units.get(unit)
        if not r:
            print(f"FAIL  {unit:<10} no reported run")
            ok_all = False
            continue
        run_id = r["run_id"]
        decl, rec, items = ev.load_declared(run_id), ev.load_record(run_id), ev.load_items(run_id)
        parts = {"declared_first": decl["declared_at"] <= rec["recorded_at"] and decl["unit"] == unit,
                 "produced_by_pipeline": rec.get("grader") == decl["grader"] and bool(items),
                 "one_run_id": all(i["run_id"] == run_id for i in items) and rec["run_id"] == run_id,
                 "check_passes": bool(rec["pass"])}
        ok = all(parts.values())
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {unit:<10} {r['metric']}={r['value']} (target {decl['target']['value']})  "
              f"run {run_id}  " + " ".join(k for k, v in parts.items() if not v))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
