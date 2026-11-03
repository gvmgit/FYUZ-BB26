"""After the runs: pick the reported run per unit, write evidence/results.json and the manifest, mirror both to the bucket.

    python tools/finalize.py [--run <unit>=<run id> ...]

By default the latest run of each unit is reported. Earlier runs stay in evidence/ with their records (a failure keeps
its record) and are listed in judging.json.
"""
import argparse
import json

from _common import load_env

load_env()

from atr.evidence import Evidence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="unit=run_id to report instead of the latest")
    a = ap.parse_args()
    ev = Evidence()
    chosen = dict(x.split("=", 1) for x in a.run)
    latest = {}
    for run_id in ev.run_ids():
        try:
            rec = ev.load_record(run_id)
        except FileNotFoundError:
            continue
        latest[rec["unit"]] = run_id
    units = {}
    for unit, run_id in {**latest, **chosen}.items():
        rec = ev.load_record(run_id)
        units[unit] = {"run_id": run_id, "metric": rec["result"]["metric"], "value": rec["result"]["value"],
                       "target": rec["target"]["value"], "pass": rec["pass"]}
    ev.results({"track": "track-3", "team_code": "team09", "units": units,
                "complete_units": sum(u["pass"] for u in units.values())})
    ev.manifest()
    print(json.dumps(units, indent=2))


if __name__ == "__main__":
    main()
