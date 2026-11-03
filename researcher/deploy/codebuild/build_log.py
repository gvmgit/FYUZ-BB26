"""Print a CodeBuild build's status and the last lines of its log (UTF-8 safe on Windows consoles).

    python deploy/codebuild/build_log.py <build-id> [lines]
"""
import sys

import boto3

bid, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30
b = boto3.client("codebuild").batch_get_builds(ids=[bid])["builds"][0]
print(b["buildStatus"], b["currentPhase"], "started", b["startTime"].strftime("%H:%M"))
for p in b.get("phases", []):
    if p.get("phaseStatus") not in (None, "SUCCEEDED"):
        print(" ", p["phaseType"], p["phaseStatus"], (p.get("contexts") or [{}])[0].get("message", "")[:300])
ev = boto3.client("logs").get_log_events(logGroupName="/atr/codebuild", logStreamName=bid.split(":")[1],
                                         startFromHead=False, limit=n)["events"]
for e in ev:
    sys.stdout.buffer.write((e["message"].rstrip()[:200] + "\n").encode("utf-8", "replace"))
