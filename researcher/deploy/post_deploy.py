"""After `cdk deploy` and provision_gateway.py: AgentCore Runtime for the agents, the review console on Amplify, and an
env file with every ATR_* setting for running the CLI against AWS.

    python deploy/post_deploy.py --outputs deploy/cdk/outputs.json --gateway deploy/cdk/gateway.txt --image <ECR URI>

Re-runnable: updates the runtime if it exists, redeploys the console, and registers (approves) the model and the tools in
the checkpoint registry. Model: ATR_MODEL_ID if set, else Claude Sonnet 5 if visible (its Marketplace subscription is on
the event allow-list), else the newest Claude Sonnet inference profile the account can see.
"""
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_NAME = "telecom_researcher"
LOCAL_FOR_AGENTS = ("kiro", "stream", "loglake", "features", "search", "graph", "protocol", "lab", "model_import", "evidence")
PREFERRED_MODEL = "us.anthropic.claude-sonnet-5"


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def pick_model(bedrock) -> str:
    if os.environ.get("ATR_MODEL_ID"):
        return os.environ["ATR_MODEL_ID"]
    profiles = bedrock.list_inference_profiles(typeEquals="SYSTEM_DEFINED").get("inferenceProfileSummaries", [])
    ids = [p["inferenceProfileId"] for p in profiles if p.get("status") == "ACTIVE"
           and "anthropic.claude" in p["inferenceProfileId"] and "sonnet" in p["inferenceProfileId"]]
    ids.sort(key=lambda i: (i.startswith("us."), i), reverse=True)
    if PREFERRED_MODEL in ids:
        return PREFERRED_MODEL
    if not ids:
        raise SystemExit("no Claude Sonnet inference profile visible: enable model access in Bedrock, or set ATR_MODEL_ID")
    return ids[0]


def runtime_env(o: dict, gateway_url: str, model: str, region: str) -> dict:
    env = {"ATR_MODE": "aws", "ATR_REGION": region, "ATR_MODEL_ID": model, "ATR_GATEWAY_URL": gateway_url,
           "ATR_DDB_TABLE": o["FindingsTable"], "ATR_AUDIT_TABLE": o["AuditTable"], "ATR_SQS_URL": o["JobsQueueUrl"],
           "ATR_FIELD_BUCKET": o["FieldBucket"], "ATR_POLICY_STORE_ID": o["PolicyStoreId"],
           "ATR_REGISTRY_TABLE": o["RegistryTable"], "ATR_GUARDRAIL_ID": o["GuardrailId"],
           "ATR_GUARDRAIL_VERSION": o["GuardrailVersion"]}
    env.update({f"ATR_{s.upper()}": "local" for s in LOCAL_FOR_AGENTS})
    if os.environ.get("ATR_NLM_ENDPOINT"):         # the OTel network language model on SageMaker (deploy/nlm_endpoint.py)
        env["ATR_NLM_ENDPOINT"] = os.environ["ATR_NLM_ENDPOINT"]
    else:
        env["ATR_NLM"] = "local"
    return env


def register(table, model: str) -> list:
    """Nothing is callable until it is approved: register the model and every tool the policies name."""
    import time
    sys.path.insert(0, str(ROOT))
    from atr.checkpoint import RESOURCE_TYPE, load_policies
    tools = sorted({a for p in load_policies() for a in (p.actions or ()) if a not in RESOURCE_TYPE and not a.startswith("ask_")})
    items = [("Model", model)] + [("Tool", t) for t in tools]
    if os.environ.get("ATR_NLM_ENDPOINT"):
        items.append(("Model", "otel-llm-1.7b-it"))
    for kind, name in items:
        table.put_item(Item={"id": f"{kind}#{name}", "kind": kind, "name": name, "status": "approved",
                             "by": "post_deploy", "ts": str(round(time.time(), 3))})
    return items


def upsert_runtime(ctl, image: str, role_arn: str, env: dict) -> dict:
    existing = next((r for r in ctl.list_agent_runtimes().get("agentRuntimes", []) if r["agentRuntimeName"] == RUNTIME_NAME), None)
    kw = {"agentRuntimeArtifact": {"containerConfiguration": {"containerUri": image}}, "roleArn": role_arn,
          "networkConfiguration": {"networkMode": "PUBLIC"}, "environmentVariables": env}
    if existing:
        return ctl.update_agent_runtime(agentRuntimeId=existing["agentRuntimeId"], **kw)
    return ctl.create_agent_runtime(agentRuntimeName=RUNTIME_NAME, **kw)


def console_zip(o: dict) -> bytes:
    cfg = {"apiBase": o["ConsoleApiUrl"].rstrip("/"),
           "cognito": {"domain": o["CognitoDomain"], "clientId": o["CognitoClientId"], "redirectUri": o["ConsoleUrl"] + "/"}}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("index.html", "app.js"):
            z.write(ROOT / "console" / f, f)
        z.writestr("config.json", json.dumps(cfg))
    return buf.getvalue()


def deploy_console(amplify, o: dict) -> str:
    d = amplify.create_deployment(appId=o["ConsoleAppId"], branchName=o["ConsoleBranch"])
    req = urllib.request.Request(d["zipUploadUrl"], data=console_zip(o), method="PUT", headers={"Content-Type": "application/zip"})
    urllib.request.urlopen(req).read()
    amplify.start_deployment(appId=o["ConsoleAppId"], branchName=o["ConsoleBranch"], jobId=d["jobId"])
    return d["jobId"]


def main():
    import boto3
    region = os.environ.get("ATR_REGION") or os.environ.get("AWS_REGION") or "us-east-2"
    o = next(iter(json.load(open(arg("--outputs"))).values()))
    raw = Path(arg("--gateway")).read_bytes()      # PowerShell redirection writes UTF-16
    text_in = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    gateway_url = next(l.split("=", 1)[1].strip() for l in text_in.splitlines() if l.startswith("ATR_GATEWAY_URL="))
    model = pick_model(boto3.client("bedrock", region_name=region))
    print("model:", model)
    env = runtime_env(o, gateway_url, model, region)
    reg = register(boto3.resource("dynamodb", region_name=region).Table(o["RegistryTable"]), model)
    print(f"registry: approved {len(reg)} items (model {model} and {len(reg) - 1} tools)")
    image = arg("--image")
    if image:
        r = upsert_runtime(boto3.client("bedrock-agentcore-control", region_name=region), image, o["AgentRoleArn"], env)
        print("agent runtime:", r.get("agentRuntimeArn"), r.get("status"))
        env["ATR_AGENT_RUNTIME_ARN"] = r.get("agentRuntimeArn", "")
    if "ConsoleAppId" in o:
        print("console deployment job:", deploy_console(boto3.client("amplify", region_name=region), o), "->", o["ConsoleUrl"])
    full = dict(env, ATR_TRUTH_TABLE=o["TruthTable"], ATR_LAKE_BUCKET=o["LakeBucket"])
    if os.environ.get("ATR_EVIDENCE_BUCKET"):      # the team data bucket the evidence chain writes to
        full["ATR_EVIDENCE_BUCKET"] = os.environ["ATR_EVIDENCE_BUCKET"]
    full.update({k: v for k, v in {
        "ATR_STATE_MACHINE_ARN": o.get("StateMachineArn"), "ATR_KINESIS_PREFIX": "atr" if "StateMachineArn" in o else None,
        "ATR_GLUE_DATABASE": "atr_features" if "AthenaWorkGroup" in o else None,
        "ATR_ATHENA_OUTPUT": f"s3://{o['LakeBucket']}/athena/" if "AthenaWorkGroup" in o else None,
        "ATR_OPENSEARCH_ENDPOINT": o.get("OpenSearchEndpoint"), "ATR_NEPTUNE_ENDPOINT": o.get("NeptuneEndpoint"),
        "ATR_LAB_GATEWAY_URL": gateway_url, "ATR_MODEL_BUCKET": o.get("ModelBucket"),
        "ATR_MODEL_IMPORT_ROLE": o.get("ModelImportRoleArn")}.items() if v})
    text = "".join(f"{k}={v}\n" for k, v in sorted(full.items()) if not (v == "local" and k.startswith("ATR_") and k != "ATR_MODE"))
    Path(arg("--env-out", "atr_aws.env")).write_text(text)
    boto3.client("s3", region_name=region).put_object(Bucket=o["FieldBucket"], Key="deploy/atr_aws.env", Body=text.encode())
    print("\n" + text)


if __name__ == "__main__":
    main()
