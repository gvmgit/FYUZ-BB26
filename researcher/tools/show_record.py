"""The read-only live test: show one item of one run, where it lives, and the times AWS recorded for it.

    python tools/show_record.py --run-id <run id> --item <n> [--window 2026-11-02T10:00:00Z,2026-11-02T12:00:00Z]
    python tools/show_record.py --window <start>,<end> [--item <n>]      # no run id: find the runs in that window

A reported run (evidence/declared/<run_id>.json exists) shows its declaration, items and grader record; item n is the
n-th line of its items file. Any other run seen in the account, such as the demo run in the video, has no declaration:
item n is then the n-th decision record of its trace.

Read calls only (they keep working after the account freeze): S3 GetObject / HeadObject on the team bucket, a DynamoDB
Query on the audit table's trace index, and a CloudWatch Logs Insights query. No model, agent, Lambda or Athena call.
Uses the caller's own AWS credentials (team account 651343917891, us-east-2).
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

from _common import REGION, TEAM_BUCKET, load_env

load_env()

import boto3  # noqa: E402
from boto3.dynamodb.conditions import Key  # noqa: E402

AUDIT_GROUP = "/atr/audit"
RUNTIME_PREFIX = "/aws/bedrock-agentcore/runtimes/telecom_researcher"


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def s3_json(s3, key: str):
    obj = s3.get_object(Bucket=TEAM_BUCKET, Key=key)
    return obj["Body"].read().decode("utf-8"), obj["LastModified"]


def s3_json_or_none(s3, key: str):
    try:
        return s3_json(s3, key)
    except s3.exceptions.NoSuchKey:
        return None


def epoch_of(text: str) -> float:
    return datetime.fromisoformat(text.strip().replace("Z", "+00:00")).timestamp()


def log_groups(logs) -> list:
    groups = [AUDIT_GROUP] + [g["logGroupName"] for g in logs.describe_log_groups(logGroupNamePrefix=RUNTIME_PREFIX)["logGroups"]]
    return [g for g in groups if logs.describe_log_groups(logGroupNamePrefix=g)["logGroups"]]


def run_query(logs, groups, start, end, query) -> list:
    q = logs.start_query(logGroupNames=groups, startTime=int(start), endTime=int(end) + 1, limit=10000, queryString=query)
    while True:
        r = logs.get_query_results(queryId=q["queryId"])
        if r["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            return [{f["field"]: f["value"] for f in row} for row in r.get("results", [])]
        time.sleep(1)


def runs_in_window(s3, start: float, end: float) -> list:
    """Every run active in the window: reported runs from their S3 declaration and record times, other runs (such as
    the demo run) from the run ids in CloudWatch Logs Insights. Read-only."""
    found = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=TEAM_BUCKET, Prefix="evidence/declared/"):
        for o in page.get("Contents", []):
            rid = o["Key"].rsplit("/", 1)[-1][:-len(".json")]
            decl = json.loads(s3_json(s3, o["Key"])[0])
            rec = s3_json_or_none(s3, f"evidence/records/{rid}.json")
            first = epoch_of(decl["declared_at"])
            last = epoch_of(json.loads(rec[0])["recorded_at"]) if rec else first
            if first <= end and last >= start:
                found[rid] = {"rid": rid, "first": iso(first), "last": iso(last), "how": f"reported run, unit {decl['unit']}"}
    logs = boto3.client("logs")
    for r in run_query(logs, log_groups(logs), start, end,
                       "fields @timestamp, @message | parse @message '\"run_id\": \"*\"' as rid | filter ispresent(rid) "
                       "| stats count(*) as events, min(@timestamp) as first, max(@timestamp) as last by rid"):
        if r["rid"] not in found:
            found[r["rid"]] = {"rid": r["rid"], "first": r["first"], "last": r["last"], "how": f"not a reported run, {r['events']} log events"}
    return sorted(found.values(), key=lambda r: r["first"])


def audit_table_name() -> str:
    if os.environ.get("ATR_AUDIT_TABLE"):
        return os.environ["ATR_AUDIT_TABLE"]
    names = boto3.client("dynamodb").list_tables()["TableNames"]
    return next(n for n in names if n.startswith("TelecomResearcher-Audit"))


def insights(trace_id: str, start: float, end: float) -> list:
    logs = boto3.client("logs")
    groups = log_groups(logs)
    return run_query(logs, groups, start, end, f'fields @timestamp, @ingestionTime, @log, @message | filter @message like "{trace_id}" '
                                               f'| sort @timestamp asc'), groups


def stream_events(trace_id: str, start: float, end: float) -> list:
    """The trace's own /atr/audit stream, read with FilterLogEvents. Logs Insights does not search events stamped before
    the log group was created; the first run after deploy (the allow unit) wrote a few seconds before that."""
    logs = boto3.client("logs")
    out, kw = [], dict(logGroupName=AUDIT_GROUP, logStreamNamePrefix=trace_id.replace(":", "-").replace("*", "-")[:500],
                       startTime=int(start * 1000), endTime=int(end * 1000) + 999)
    for page in logs.get_paginator("filter_log_events").paginate(**kw):
        out += [{"@timestamp": iso(e["timestamp"] / 1000).replace("T", " ").rstrip("Z"),
                 "@ingestionTime": iso(e["ingestionTime"] / 1000).replace("T", " ").rstrip("Z"),
                 "@log": f"{AUDIT_GROUP}:{e['logStreamName']}", "@message": e["message"]} for e in page["events"]]
    return sorted(out, key=lambda e: e["@timestamp"])


def trace_rows(table: str, trace_id: str) -> list:
    t = boto3.resource("dynamodb", region_name=REGION).Table(table)
    q = dict(IndexName="trace", KeyConditionExpression=Key("trace_id").eq(trace_id))
    rows, page = [], t.query(**q)
    rows += page["Items"]
    while "LastEvaluatedKey" in page:
        page = t.query(ExclusiveStartKey=page["LastEvaluatedKey"], **q)
        rows += page["Items"]
    return sorted(rows, key=lambda r: float(r["epoch"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", help="a run id from judging.json, or any run seen in the account")
    ap.add_argument("--item", type=int, default=0)
    ap.add_argument("--window", help="start,end (ISO UTC): restricts the CloudWatch search; without --run-id, finds the runs in it")
    a = ap.parse_args()
    if not a.run_id and not a.window:
        ap.error("give --run-id, or --window to find the runs in a time window")
    window = None
    if a.window:
        w0, w1 = a.window.split(",")
        window = (epoch_of(w0), epoch_of(w1))
    s3 = boto3.client("s3", region_name=REGION)

    if not a.run_id:
        found = runs_in_window(s3, *window)
        print(f"RUNS ACTIVE IN {iso(window[0])} .. {iso(window[1])}  (S3 declarations and CloudWatch Logs Insights)")
        for r in found:
            print(f"  {r['rid']:<44} {r['first']} .. {r['last']}   {r['how']}")
        if len(found) != 1:
            print(f"\n{len(found)} runs in the window. Re-run with --run-id <one of them> --item <n>." if found else "\nNo runs in the window.")
            return
        a.run_id = found[0]["rid"]
        print(f"\nOne run in the window: showing {a.run_id}\n")

    declared = s3_json_or_none(s3, f"evidence/declared/{a.run_id}.json")
    table = audit_table_name()
    if declared:
        decl_text, decl_at = declared
        decl = json.loads(decl_text)
        items_text, items_at = s3_json(s3, f"evidence/items/{a.run_id}.jsonl")
        items = [json.loads(x) for x in items_text.splitlines() if x.strip()]
        rec_text, rec_at = s3_json(s3, f"evidence/records/{a.run_id}.json")
        rec = json.loads(rec_text)
        if not 0 <= a.item < len(items):
            raise SystemExit(f"run {a.run_id} has {len(items)} items: --item must be 0 to {len(items) - 1}")
        item = items[a.item]

        print(f"RUN {a.run_id}   unit {decl['unit']}   target {decl['target']}   grader {decl['grader']}")
        print(f"  declaration  s3://{TEAM_BUCKET}/evidence/declared/{a.run_id}.json   declared_at {decl['declared_at']}   S3 LastModified {decl_at.isoformat()}")
        print(f"  items        s3://{TEAM_BUCKET}/evidence/items/{a.run_id}.jsonl   ({len(items)} items)   S3 LastModified {items_at.isoformat()}")
        print(f"  record       s3://{TEAM_BUCKET}/evidence/records/{a.run_id}.json   result {rec['result']['metric']}={rec['result']['value']}  "
              f"{'PASS' if rec['pass'] else 'FAIL'}   S3 LastModified {rec_at.isoformat()}")
        if decl.get("test_set"):
            print(f"  frozen test set {decl['test_set']}  sha256 {decl['test_set_sha256']}")
        if rec.get("training_job"):
            print(f"  SageMaker training job {rec['training_job']}   adapter {rec.get('adapter')}")
        print(f"\nITEM {a.item}")
        print(json.dumps(item, indent=2, default=str)[:6000])
        trace_id = item.get("trace_id") or (item.get("record") or {}).get("trace_id") or a.run_id
        rows = trace_rows(table, trace_id)
    else:
        # a run seen in the account but not a reported unit (for example the demo run): its trace is the evidence
        trace_id = a.run_id
        rows = trace_rows(table, trace_id)
        if not rows:
            raise SystemExit(f"no declaration in s3://{TEAM_BUCKET}/evidence/declared/ and no audit records for {a.run_id}")
        if not 0 <= a.item < len(rows):
            raise SystemExit(f"run {a.run_id} has {len(rows)} decision records: --item must be 0 to {len(rows) - 1}")
        print(f"RUN {a.run_id}   not a reported unit: no declaration in s3://{TEAM_BUCKET}/evidence/declared/")
        print(f"  its evidence is the trace: {len(rows)} records in DynamoDB {table} (index trace) and in CloudWatch Logs")
        print(f"\nITEM {a.item}  (decision record {a.item} of the trace)")
        print(json.dumps(rows[a.item], indent=2, default=str)[:6000])

    print(f"\nDYNAMODB  table {table}  index trace  trace_id = {trace_id}  ->  {len(rows)} records")
    shown = rows if len(rows) <= 10 else rows[:5] + [{"kind": f"... {len(rows) - 10} more ..."}] + rows[-5:]
    for r in shown:
        if "epoch" in r:
            print(f"  {iso(r['epoch'])}  {r.get('kind'):<14} {r.get('principal', r.get('agent', '')):<24} "
                  f"{r.get('action', '')!s:<18} {r.get('decision', '')!s:<6} {','.join(r.get('policies', []) or [])}")
        else:
            print(f"  {r['kind']}")

    if rows:
        start, end = float(rows[0]["epoch"]) - 60, float(rows[-1]["epoch"]) + 60
    else:
        start, end = time.time() - 7 * 86400, time.time()
    if window:
        inside = sum(1 for r in rows if window[0] <= float(r["epoch"]) <= window[1])
        print(f"  {inside} of {len(rows)} records fall in the window {iso(window[0])} .. {iso(window[1])}")
        if rows and not inside:
            print(f"  note: this trace runs {iso(float(rows[0]['epoch']))} .. {iso(float(rows[-1]['epoch']))}, outside the window")
        start, end = window
    events, groups = insights(trace_id, start, end)
    print(f"\nCLOUDWATCH LOGS INSIGHTS  groups {groups}\n  query: filter @message like \"{trace_id}\"  window {iso(start)} .. {iso(end)}"
          f"  ->  {len(events)} events")
    if not events:
        events = stream_events(trace_id, start, end)
        if events:
            print(f"CLOUDWATCH LOGS  FilterLogEvents on {AUDIT_GROUP}, stream {trace_id}  ->  {len(events)} events  "
                  f"(stamped before the log group's creation, which Logs Insights does not search)")
    for e in events[:3] + events[-3:] if len(events) > 6 else events:
        print(f"  @timestamp {e.get('@timestamp')}  @ingestionTime {e.get('@ingestionTime')}  {e.get('@log', '').split(':')[-1]}  "
              f"{e.get('@message', '')[:140]}")
    if rows and events:
        first_db, first_cw = float(rows[0]["epoch"]), datetime.fromisoformat(events[0]["@timestamp"].replace(" ", "T") + "+00:00").timestamp()
        print(f"\nTIMES  first DynamoDB record {iso(first_db)}  first CloudWatch event {iso(first_cw)}  "
              f"difference {abs(first_cw - first_db):.1f} s")


if __name__ == "__main__":
    main()
