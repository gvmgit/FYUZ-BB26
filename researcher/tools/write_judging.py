"""Write judging.json from evidence/ (up to 20 run ids, the reported results) and copy it to the bucket under judging/.

    python tools/write_judging.py
"""
import json

from _common import ROOT, TEAM_BUCKET, load_env

load_env()

from atr.evidence import Evidence  # noqa: E402


def main():
    ev = Evidence()
    results = json.loads((ev.root / "results.json").read_text())
    reported = [u["run_id"] for u in results["units"].values()]
    others = [r for r in reversed(ev.run_ids()) if r not in reported]
    doc = {
        "primary_track": "track-3",
        "team_code": "team09",
        "title": "Autonomous Telecom Researcher",
        "evidence": {"bucket": TEAM_BUCKET, "prefix": "evidence/", "repo_path": "researcher/evidence/",
                     "audit_table_prefix": "TelecomResearcher-Audit", "audit_index": "trace",
                     "log_groups": ["/atr/audit", "/aws/bedrock-agentcore/runtimes/telecom_researcher-*"]},
        "run_ids": (reported + others)[:20],
        "live_test_command": "python3 researcher/tools/show_record.py --run-id {run_id} --item {item}",
        "rebuild_command": "python3 researcher/tools/rebuild.py",
        "check_command": "python3 researcher/tools/check_units.py",
        "reported_results": {u: {r["metric"]: r["value"], "target": r["target"], "pass": r["pass"], "run_id": r["run_id"]}
                             for u, r in sorted(results["units"].items())},
        "complete_units": results["complete_units"],
        "frozen_test_sets": {d["test_set"]: d["test_set_sha256"] for d in (ev.load_declared(r["run_id"]) for r in results["units"].values())
                             if d.get("test_set")},
        "grader": "rule-based-grader-v1 (atr/grader.py); no model grades",
        "model_artifact_s3_prefix": (ev.load_record(results["units"]["nlm_tuning"]["run_id"])["adapter"].rsplit("/", 1)[0] + "/"
                                     if "nlm_tuning" in results["units"] else "none"),
        "data": "DATA.md",
        "gaps": "GAPS.md",
        "region": "us-east-2",
    }
    text = json.dumps(doc, indent=2) + "\n"
    (ROOT.parent / "judging.json").write_text(text, encoding="utf-8", newline="\n")      # the repository root
    if ev.bucket:
        ev.s3.put_object(Bucket=ev.bucket, Key="judging/judging.json", Body=text.encode(), ContentType="application/json")
        print(f"judging.json -> s3://{ev.bucket}/judging/judging.json")
    print(text)


if __name__ == "__main__":
    main()
