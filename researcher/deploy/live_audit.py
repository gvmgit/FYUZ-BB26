"""Live view of an investigation running on AWS: follows the DynamoDB audit trail (agent messages, tool calls through
AgentCore Gateway, pre-registrations, policy denials) and the finding cards as they are published.

    . .\\deploy\\aws_env.ps1 ; python deploy/live_audit.py [--since-minutes 5]
"""
import json
import sys
import time
from decimal import Decimal

sys.path.insert(0, "deploy")
import aws_session  # noqa: E402  (self-refreshing credentials from the `atr` login)
from boto3.dynamodb.conditions import Attr  # noqa: E402

o = next(iter(json.load(open("deploy/cdk/outputs.json")).values()))


class _Table:
    """DynamoDB table that re-resolves through a fresh session on every scan."""

    def __init__(self, name):
        self.name = name

    def scan(self, **kw):
        return aws_session.resource("dynamodb").Table(self.name).scan(**kw)


audit, findings = _Table(o["AuditTable"]), _Table(o["FindingsTable"])
since = time.time() - 60 * float(sys.argv[sys.argv.index("--since-minutes") + 1]) if "--since-minutes" in sys.argv else time.time()
start, seen, cards_seen = since, set(), set()
counts = {"experiments": 0, "failed": 0, "messages": 0}
COL = {"orchestrator": "\033[96m", "scientist": "\033[94m", "experimenter": "\033[93m", "critic": "\033[91m",
       "analyst": "\033[92m", "ok": "\033[92m", "bad": "\033[91m", "dim": "\033[90m", "end": "\033[0m"}


def c(role, text):
    return f"{COL.get(role, '')}{text}{COL['end']}"


def scan(table, attr, after):
    items, kw = [], {"FilterExpression": Attr(attr).gt(Decimal(str(after)))}   # DynamoDB numbers are Decimals
    while True:
        r = table.scan(**kw)
        items += r["Items"]
        if "LastEvaluatedKey" not in r:
            return items
        kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]


print(c("dim", "Live view of the Telecom Researcher on AWS (Ctrl-C to stop). Waiting for agent activity..."))
while True:
    try:
        new = sorted((i for i in scan(audit, "epoch", start - 5) if i["id"] not in seen), key=lambda i: float(i["epoch"]))
        for i in new:
            seen.add(i["id"])
            k, ts = i.get("kind"), str(i.get("ts", ""))[11:19]
            if k == "agent_message":
                counts["messages"] += 1
                print(f"{c('dim', ts)}  {c(i['from'], i['from'])} -> {c(i['to'], i['to'])}: {str(i.get('summary'))[:110]}")
            elif k == "tool_call":
                ok = i.get("ok")
                counts["experiments" if ok else "failed"] += 1
                args = i.get("args") or {}
                label = (args.get("config") or {}).get("name", "") if isinstance(args, dict) else ""
                mark = c("ok", "OK ") if ok else c("bad", "ERR")
                print(f"{c('dim', ts)}    {mark} {c(i.get('agent'), i.get('agent'))} {i.get('tool')} {str(label)[:60]}"
                      + ("" if ok else c("bad", f"  {str(i.get('error'))[:90]}")) + c("dim", f"  ({i.get('seconds')}s)"))
            elif k == "preregistration":
                print(f"{c('dim', ts)}  {c('scientist', 'sealed')} claim {i.get('claim')}  hash {str(i.get('hash'))[:12]}")
            elif k == "decision" and i.get("decision") == "DENY":
                print(f"{c('dim', ts)}  {c('bad', 'DENIED')} {i.get('principal')} -> {i.get('action')} on {i.get('resource')} "
                      f"({', '.join(i.get('policies') or []) or 'no permit'})")
        if new:
            start = max(start, max(float(i["epoch"]) for i in new))
        for card in scan(findings, "created", since):
            if card["id"] not in cards_seen:
                cards_seen.add(card["id"])
                head = " <== HEADLINE" if card.get("headline") else ""
                print(c("ok" if card.get("verdict") == "confirmed" else "dim",
                        f"          FINDING: {card.get('title')}  [{card.get('verdict')}]{head}"))
        sys.stdout.write(c("dim", f"\r          experiments {counts['experiments']}  failed {counts['failed']}  "
                                  f"messages {counts['messages']}  findings {len(cards_seen)}   "))
        sys.stdout.flush()
        time.sleep(3)
    except KeyboardInterrupt:
        print()
        break
    except Exception as e:  # noqa: BLE001 - keep the live view running through transient errors
        if "Expired" in str(e):
            aws_session.expire_now()
        print(c("bad", f"\n(read error, retrying: {str(e)[:120]})"))
        time.sleep(5)
