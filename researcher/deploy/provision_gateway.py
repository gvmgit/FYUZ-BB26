"""Create the AgentCore Gateway and register the run_experiment Lambda as its tool.

    set ATR_GATEWAY_ROLE_ARN=<role Gateway assumes to invoke the Lambda>  (the CDK stack outputs it)
    set ATR_LAMBDA_ARN=<run_experiment Lambda ARN>                         (the CDK stack outputs it)
    python deploy/provision_gateway.py

Prints the gateway URL to put in ATR_GATEWAY_URL. The gateway uses the AWS_IAM authorizer, so callers sign requests
with SigV4 (the agents do this when ATR_GATEWAY_TOKEN is not set). The request shapes were validated against the
installed botocore service model (scripts/check_deploy.py); creating the resources needs a real account.
"""
import copy
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atr.gateway import RUN_EXPERIMENT_SCHEMA  # noqa: E402

TOOL_DESCRIPTION = ("Run one MAVERIC twin experiment (change at most one setting) over several seeds and return "
                    "per-seed KPIs and the discrepancy score against the field log.")


GATEWAY_KEYS = ("type", "properties", "required", "items", "description")


def gateway_schema(schema: dict) -> dict:
    """AgentCore Gateway's tool-schema language allows only type/properties/required/items/description, and every node
    needs a type. Enums move into the description; untyped values become strings."""
    s = copy.deepcopy(schema)
    out = {k: v for k, v in s.items() if k in GATEWAY_KEYS}
    if "enum" in s:
        out["description"] = (out.get("description", "") + " One of: " + ", ".join(map(str, s["enum"]))).strip()
    out.setdefault("type", "object" if "properties" in s else "string")
    if "properties" in out:
        out["properties"] = {k: gateway_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = gateway_schema(out["items"])
    return out


def tool_schema() -> dict:
    return gateway_schema(RUN_EXPERIMENT_SCHEMA)


def create_gateway(client, name: str, role_arn: str) -> dict:
    return client.create_gateway(name=name, roleArn=role_arn, protocolType="MCP", authorizerType="AWS_IAM",
                                 description="Telecom Researcher: run_experiment on the MAVERIC digital twin")


def create_target(client, gateway_id: str, lambda_arn: str) -> dict:
    return client.create_gateway_target(
        gatewayIdentifier=gateway_id, name="twin",
        description="MAVERIC digital twin experiments",
        targetConfiguration={"mcp": {"lambda": {
            "lambdaArn": lambda_arn,
            "toolSchema": {"inlinePayload": [{"name": "run_experiment", "description": TOOL_DESCRIPTION,
                                              "inputSchema": tool_schema()}]}}}},
        credentialProviderConfigurations=[{"credentialProviderType": "GATEWAY_IAM_ROLE"}])


LAB_DESCRIPTION = ("Drive the lab through the test automation API: traffic profile, handover rule, parameters, campaigns and "
                   "clean-lab replications. Never touches the live network.")


def create_lab_target(client, gateway_id: str, lambda_arn: str) -> dict:
    from atr.gateway import LAB_SCHEMA
    return client.create_gateway_target(
        gatewayIdentifier=gateway_id, name="lab", description="Emulated RAN lab (test automation API)",
        targetConfiguration={"mcp": {"lambda": {"lambdaArn": lambda_arn, "toolSchema": {"inlinePayload": [
            {"name": "lab_execute", "description": LAB_DESCRIPTION, "inputSchema": gateway_schema(LAB_SCHEMA)}]}}}},
        credentialProviderConfigurations=[{"credentialProviderType": "GATEWAY_IAM_ROLE"}])


def find_or_create(client, name: str, role_arn: str) -> dict:
    """Re-runnable: reuse a gateway with this name, else create it; then wait until it is READY."""
    import time
    gw = next((g for g in client.list_gateways().get("items", []) if g.get("name") == name), None)
    gid = gw["gatewayId"] if gw else create_gateway(client, name, role_arn)["gatewayId"]
    if gw:
        g = client.get_gateway(gatewayIdentifier=gid)
        if g.get("roleArn") != role_arn:        # the stack replaced the gateway role (e.g. moved under /bb26/)
            client.update_gateway(gatewayIdentifier=gid, name=name, roleArn=role_arn, protocolType="MCP",
                                  authorizerType="AWS_IAM", description=g.get("description", ""))
            print(f"gateway {gid}: role updated to {role_arn}", file=sys.stderr)
            time.sleep(5)
    for _ in range(60):
        g = client.get_gateway(gatewayIdentifier=gid)
        if g["status"] == "READY":
            return g
        if g["status"] == "FAILED":
            raise RuntimeError(f"gateway {gid} failed: {g.get('statusReasons')}")
        time.sleep(5)
    raise TimeoutError(f"gateway {gid} not READY after 5 minutes")


def main():
    import boto3
    import json
    if "--outputs" in sys.argv:      # CDK outputs file written by `cdk deploy --outputs-file`
        o = next(iter(json.load(open(sys.argv[sys.argv.index("--outputs") + 1])).values()))
        os.environ.setdefault("ATR_GATEWAY_ROLE_ARN", o["GatewayRoleArn"])
        os.environ.setdefault("ATR_LAMBDA_ARN", o["RunExperimentLambdaArn"])
        os.environ.setdefault("ATR_LAB_LAMBDA_ARN", o.get("LabExecutorLambdaArn", ""))
    client = boto3.client("bedrock-agentcore-control", region_name=os.environ.get("ATR_REGION") or os.environ.get("AWS_REGION") or "us-east-2")
    gw = find_or_create(client, "telecom-researcher", os.environ["ATR_GATEWAY_ROLE_ARN"])
    have = {t["name"] for t in client.list_gateway_targets(gatewayIdentifier=gw["gatewayId"]).get("items", [])}
    if "twin" not in have:
        create_target(client, gw["gatewayId"], os.environ["ATR_LAMBDA_ARN"])
    if os.environ.get("ATR_LAB_LAMBDA_ARN") and "lab" not in have:
        create_lab_target(client, gw["gatewayId"], os.environ["ATR_LAB_LAMBDA_ARN"])
    print("ATR_GATEWAY_URL=" + gw["gatewayUrl"])
    print("ATR_LAB_GATEWAY_URL=" + gw["gatewayUrl"])


if __name__ == "__main__":
    main()
