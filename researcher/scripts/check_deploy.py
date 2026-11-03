"""Checks the deployment artifacts as far as is possible without an AWS account, Docker or Node.

 - the Lambda handler runs the real twin and returns JSON-serialisable results with an S3 heatmap reference
 - the AgentCore Gateway API calls are valid against botocore's service model (parameter validation)
 - the SigV4 signer produces a signed request
 - every file the Dockerfiles COPY exists, and the deploy Python compiles

NOT checked here: `cdk synth`, `docker build`, and any call that reaches AWS.

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_deploy.py
"""
import json
import py_compile
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HACK = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy" / "lambda_fn"))
sys.path.insert(0, str(ROOT / "deploy"))
warnings.filterwarnings("ignore")

results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# 1. Lambda handler against the real twin
from atr.twin import RFTwin, default_inventory  # noqa: E402
from atr.faults import Fault  # noqa: E402
from atr.world import World  # noqa: E402
import handler as lam  # noqa: E402

twin = RFTwin.load_or_build()
field = World(twin, default_inventory(), Fault("tilt", 3, 16)).field_observation(600)
uploads = []
event = {"config": {"name": "cell 3 tilt 16", "cells": {"3": {"tilt": 16.0}}, "render": True}, "seeds": [1, 2, 3]}
res = lam.handler(event, None, field_loader=lambda: field,
                  uploader=lambda path, key: uploads.append((path, key)) or f"s3://bucket/{key}")
check("handler returns a JSON-serialisable result", bool(json.dumps(res)))
check("handler returns an S3 reference for the heatmap and uploads exactly that file",
      res["heatmap"].startswith("s3://bucket/heatmaps/") and Path(uploads[0][0]).exists())
check("run ids are unique per container (prefix)", "-exp" in res["run_id"])
check("handler scores the true fix near the noise floor", res["mean"]["score"] < 4.0, f"score {res['mean']['score']:.2f}")
mob = lam.handler({"config": {"objective": "mobility", "n_ue": 10, "n_ticks": 20}, "seeds": [1, 2]}, None,
                  field_loader=lambda: field)
check("the Lambda handler also serves the mobility objective, JSON-safe", mob["objective"] == "mobility"
      and bool(json.dumps(mob)) and mob["mean"]["handovers_per_ue_min"] >= 0)
for bad, why in [({"cells": {"3": {"tilt": 40}}}, "tilt out of range"),
                 ({"cells": {"3": {"tilt": 8}, "2": {"tilt": 8}}}, "two changes")]:
    try:
        lam.handler({"config": bad, "seeds": [1]}, None, field_loader=lambda: field)
        check(f"handler enforces safety bounds ({why})", False)
    except ValueError:
        check(f"handler enforces safety bounds ({why})", True)

# 2. Gateway API calls validate against botocore's service model
import boto3  # noqa: E402
from botocore.stub import Stubber  # noqa: E402
import provision_gateway as pg  # noqa: E402

client = boto3.client("bedrock-agentcore-control", region_name="us-west-2",
                      aws_access_key_id="x", aws_secret_access_key="y")
with Stubber(client) as stub:
    stub.add_response("create_gateway", {"gatewayId": "gw-1", "gatewayUrl": "https://gw", "gatewayArn": "arn:x",
                                         "status": "CREATING", "createdAt": 0, "updatedAt": 0, "name": "n",
                                         "roleArn": "arn:role", "protocolType": "MCP", "authorizerType": "AWS_IAM"})
    stub.add_response("create_gateway_target", {"gatewayArn": "arn:x", "targetId": "t-1", "createdAt": 0,
                                                "updatedAt": 0, "status": "CREATING", "name": "twin",
                                                "targetConfiguration": {"mcp": {"lambda": {
                                                    "lambdaArn": "arn:lambda", "toolSchema": {"inlinePayload": []}}}},
                                                "credentialProviderConfigurations": [
                                                    {"credentialProviderType": "GATEWAY_IAM_ROLE"}]})
    try:
        pg.create_gateway(client, "telecom-researcher", "arn:aws:iam::123456789012:role/r")
        pg.create_target(client, "gw-1", "arn:aws:lambda:us-west-2:123456789012:function:f")
        check("create_gateway and create_gateway_target requests are valid for the installed SDK", True)
    except Exception as e:  # noqa: BLE001
        check("create_gateway and create_gateway_target requests are valid for the installed SDK", False, str(e)[:160])
check("tool schema sent to the gateway has no unsupported keywords",
      "minItems" not in json.dumps(pg.tool_schema()))

# 3. SigV4 signer
from botocore.credentials import Credentials  # noqa: E402
import httpx  # noqa: E402
from atr import aws_adapters as aws  # noqa: E402

auth = aws.SigV4HttpxAuth("us-west-2", session=type("S", (), {"get_credentials": lambda s: Credentials("AKIAEXAMPLE", "secret")})())
req = httpx.Request("POST", "https://gw.example.amazonaws.com/mcp", json={"jsonrpc": "2.0", "method": "tools/list"})
signed = next(auth.auth_flow(req))
check("SigV4 signer adds an AWS4-HMAC-SHA256 Authorization header",
      signed.headers.get("authorization", "").startswith("AWS4-HMAC-SHA256"))

# 4. Dockerfile COPY sources exist, deploy python compiles
for df in ("lambda_fn", "agentcore"):
    text = (ROOT / "deploy" / df / "Dockerfile").read_text()
    missing = []
    for line in text.splitlines():
        if line.startswith("COPY"):
            src = re.findall(r'"([^"]+)"', line)[0] if '"' in line else line.split()[1]
            if not (HACK / src).exists():
                missing.append(src)
    check(f"{df} Dockerfile: every COPY source exists", not missing, ", ".join(missing))
for f in ["deploy/agentcore_app.py", "deploy/provision_gateway.py", "deploy/cdk/app.py", "deploy/cdk/stack.py",
          "deploy/lambda_fn/handler.py"]:
    try:
        py_compile.compile(str(ROOT / f), doraise=True, cfile=str(ROOT / "runs" / "_c.pyc"))
        check(f"{f} compiles", True)
    except py_compile.PyCompileError as e:
        check(f"{f} compiles", False, str(e)[:120])
(ROOT / "runs" / "_c.pyc").unlink(missing_ok=True)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
