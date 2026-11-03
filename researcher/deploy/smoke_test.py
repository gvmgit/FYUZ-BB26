"""Smoke-test the deployed stack: AgentCore Runtime and Amplify status, plant a secret fault, call both Lambdas, and
(with --investigate) run a full investigation inside AgentCore Runtime.

    python deploy/smoke_test.py deploy/cdk/outputs.json [--investigate]
"""
import json
import sys
import time

import boto3

o = next(iter(json.load(open(sys.argv[1])).values()))
ok = []


def check(name, cond, detail=""):
    ok.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


ctl = boto3.client("bedrock-agentcore-control")
rt = next(r for r in ctl.list_agent_runtimes()["agentRuntimes"] if r["agentRuntimeName"] == "telecom_researcher")
for _ in range(36):
    st = ctl.get_agent_runtime(agentRuntimeId=rt["agentRuntimeId"])["status"]
    if st not in ("CREATING", "UPDATING"):
        break
    time.sleep(10)
check("AgentCore Runtime ready", st == "READY", st)

amp = boto3.client("amplify")
job = amp.list_jobs(appId=o["ConsoleAppId"], branchName=o["ConsoleBranch"], maxResults=1)["jobSummaries"][0]
check("review console published on Amplify", job["status"] == "SUCCEED", f"{job['status']} -> {o['ConsoleUrl']}")

gw = boto3.client("bedrock-agentcore-control")
g = next(x for x in gw.list_gateways()["items"] if x["name"] == "telecom-researcher")
targets = {t["name"]: t["status"] for t in gw.list_gateway_targets(gatewayIdentifier=g["gatewayId"])["items"]}
check("AgentCore Gateway has both tools", set(targets) >= {"twin", "lab"}, targets)

lam = boto3.client("lambda")


def invoke(fn, payload):
    r = lam.invoke(FunctionName=fn, Payload=json.dumps(payload).encode())
    body = json.loads(r["Payload"].read() or b"{}")
    return r.get("FunctionError"), body


err, body = invoke(o["LabExecutorLambdaArn"], {"action": "set_traffic", "profile": "night"})
check("lab executor Lambda answers", not err and body.get("ok"), err or body)

err, body = invoke(o["RunExperimentLambdaArn"], {"config": {"name": "baseline", "cells": {}}, "seeds": [1, 2]})
check("run_experiment Lambda runs the twin on the planted field log", not err and "per_seed" in json.dumps(body),
      (err, str(body)[:200]) if err else f"{len(body.get('per_seed', []))} seeds")

if "--investigate" in sys.argv:
    from botocore.config import Config
    dp = boto3.client("bedrock-agentcore", config=Config(read_timeout=900, retries={"max_attempts": 0}))
    t0 = time.time()
    r = dp.invoke_agent_runtime(agentRuntimeArn=rt["agentRuntimeArn"], payload=json.dumps({"mode": "aws"}).encode(),
                                contentType="application/json", accept="application/json")
    out = json.loads(r["response"].read())
    check("full investigation inside AgentCore Runtime", out.get("headline"), f"{out} in {time.time() - t0:.0f}s")

print(f"\n{sum(ok)}/{len(ok)} smoke checks passed")
sys.exit(0 if all(ok) else 1)
