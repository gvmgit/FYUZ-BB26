"""Create (or reuse) the S3 source bucket, the CodeBuild service role and the `atr-deploy` project, upload the package and
start a build. Re-runnable.  Needs AWS credentials in the environment (`. .\\deploy\\aws_env.ps1`).

    python deploy/codebuild/launch.py [--phase 3] [--no-start]
"""
import json
import os
import sys
import time
from pathlib import Path

import boto3

REGION = os.environ.get("ATR_DEPLOY_REGION") or os.environ.get("AWS_REGION") or "us-east-2"
PKG = Path(__file__).resolve().parents[1] / "atr_aws_package.zip"
PROJECT, ROLE = "atr-deploy", "atr-codebuild-deploy"
ROLE_PATH = "/bb26/"            # event account rule: service roles under /bb26/ with the bb26 permissions boundary
EVIDENCE_BUCKET = os.environ.get("ATR_EVIDENCE_BUCKET", "")
NLM_ENDPOINT = os.environ.get("ATR_NLM_ENDPOINT", "")      # OTel network language model (deploy/nlm_endpoint.py)
MODEL = os.environ.get("ATR_MODEL_ID", "us.anthropic.claude-sonnet-5")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def deploy_policy(account: str, bucket: str) -> dict:
    a, r = account, REGION
    return {"Version": "2012-10-17", "Statement": [
        {"Sid": "UseCdkBootstrapRoles", "Effect": "Allow", "Action": "sts:AssumeRole",
         "Resource": f"arn:aws:iam::{a}:role/cdk-hnb659fds-*-{a}-{r}"},
        {"Sid": "ReadBootstrapVersion", "Effect": "Allow", "Action": ["ssm:GetParameter", "ssm:GetParameters"],
         "Resource": f"arn:aws:ssm:{r}:{a}:parameter/cdk-bootstrap/*"},
        {"Sid": "BuildLogs", "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": [f"arn:aws:logs:{r}:{a}:log-group:/atr/codebuild", f"arn:aws:logs:{r}:{a}:log-group:/atr/codebuild:*"]},
        {"Sid": "SourceAndArtifacts", "Effect": "Allow", "Action": ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject",
                                                                    "s3:GetBucketLocation", "s3:ListBucket"],
         "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"]},
        {"Sid": "WriteEnvFile", "Effect": "Allow", "Action": "s3:PutObject",
         "Resource": "arn:aws:s3:::telecomresearcher-field*/deploy/*"},
        {"Sid": "ReadStackOutputs", "Effect": "Allow", "Action": ["cloudformation:DescribeStacks"],
         "Resource": f"arn:aws:cloudformation:{r}:{a}:stack/TelecomResearcher/*"},
        {"Sid": "AgentImage", "Effect": "Allow", "Action": ["ecr:CreateRepository", "ecr:DescribeRepositories",
                                                            "ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload",
                                                            "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage",
                                                            "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
         "Resource": f"arn:aws:ecr:{r}:{a}:repository/atr-agents"},
        {"Sid": "EcrLogin", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
        {"Sid": "AgentCoreGatewayAndRuntime", "Effect": "Allow", "Action": [
            "bedrock-agentcore:CreateGateway", "bedrock-agentcore:UpdateGateway","bedrock-agentcore:GetGateway", "bedrock-agentcore:ListGateways",
            "bedrock-agentcore:CreateGatewayTarget", "bedrock-agentcore:ListGatewayTargets", "bedrock-agentcore:GetGatewayTarget",
            "bedrock-agentcore:CreateAgentRuntime", "bedrock-agentcore:UpdateAgentRuntime", "bedrock-agentcore:GetAgentRuntime",
            "bedrock-agentcore:ListAgentRuntimes", "bedrock-agentcore:CreateWorkloadIdentity",
            "bedrock-agentcore:GetWorkloadIdentity", "bedrock-agentcore:CreateAgentRuntimeEndpoint",
            "bedrock-agentcore:UpdateAgentRuntimeEndpoint", "bedrock-agentcore:GetAgentRuntimeEndpoint",
            "bedrock-agentcore:ListAgentRuntimeEndpoints"], "Resource": "*"},
        {"Sid": "PassOnlyTheStackRoles", "Effect": "Allow", "Action": "iam:PassRole",
         "Resource": [f"arn:aws:iam::{a}:role/bb26/TelecomResearcher-GatewayRole*",
                      f"arn:aws:iam::{a}:role/bb26/TelecomResearcher-AgentRole*"]},
        {"Sid": "AgentCoreServiceLinkedRole", "Effect": "Allow", "Action": "iam:CreateServiceLinkedRole", "Resource": "*",
         "Condition": {"StringLike": {"iam:AWSServiceName": "*bedrock-agentcore.amazonaws.com"}}},
        {"Sid": "PickModel", "Effect": "Allow", "Action": ["bedrock:ListInferenceProfiles"], "Resource": "*"},
        {"Sid": "RegisterModelAndTools", "Effect": "Allow", "Action": "dynamodb:PutItem",
         "Resource": f"arn:aws:dynamodb:{r}:{a}:table/TelecomResearcher-Registry*"},
        {"Sid": "PublishConsole", "Effect": "Allow", "Action": ["amplify:CreateDeployment", "amplify:StartDeployment"],
         "Resource": f"arn:aws:amplify:{r}:{a}:apps/*"},
        {"Sid": "DetectIdentityCenter", "Effect": "Allow", "Action": "sso:ListInstances", "Resource": "*"},
    ]}


def main():
    account = boto3.client("sts").get_caller_identity()["Account"]
    bucket = f"atr-source-{account}-{REGION}"
    s3, iam, cb = boto3.client("s3", region_name=REGION), boto3.client("iam"), boto3.client("codebuild", region_name=REGION)

    try:
        s3.head_bucket(Bucket=bucket)
    except s3.exceptions.ClientError:
        s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": REGION})
        s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        print("created bucket", bucket)
    s3.upload_file(str(PKG), bucket, "atr_aws_package.zip")
    print(f"uploaded {PKG.name} ({PKG.stat().st_size / 1e6:.1f} MB) to s3://{bucket}/")

    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "codebuild.amazonaws.com"},
                                                      "Action": "sts:AssumeRole"}]}
    try:
        role_arn = iam.get_role(RoleName=ROLE)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        role_arn = iam.create_role(RoleName=ROLE, Path=ROLE_PATH, AssumeRolePolicyDocument=json.dumps(trust),
                                   PermissionsBoundary=f"arn:aws:iam::{account}:policy/bb26-boundary/bb26-Boundary",
                                   Description="CodeBuild deploys the Telecom Researcher CDK stack")["Role"]["Arn"]
        print("created role", ROLE)
        time.sleep(10)          # IAM propagation before CodeBuild uses the role
    # Least privilege: CDK itself deploys through its bootstrap roles (created once by `cdk bootstrap` from a person's
    # session), so this role only assumes those, plus what the gateway/runtime/console steps need.
    iam.put_role_policy(RoleName=ROLE, PolicyName="atr-deploy", PolicyDocument=json.dumps(deploy_policy(account, bucket)))

    spec = {"name": PROJECT, "source": {"type": "S3", "location": f"{bucket}/atr_aws_package.zip"},
            "artifacts": {"type": "S3", "location": bucket, "path": "build-output", "packaging": "NONE", "name": PROJECT},
            "environment": {"type": "LINUX_CONTAINER", "image": "aws/codebuild/standard:7.0", "computeType": "BUILD_GENERAL1_LARGE",
                            "privilegedMode": True, "environmentVariables": [
                                {"name": "PHASE", "value": arg("--phase", "3")}, {"name": "ATR_MODEL_ID", "value": MODEL},
                                {"name": "ATR_EVIDENCE_BUCKET", "value": EVIDENCE_BUCKET},
                                {"name": "ATR_NLM_ENDPOINT", "value": NLM_ENDPOINT}]},
            "serviceRole": role_arn, "timeoutInMinutes": 120,
            "logsConfig": {"cloudWatchLogs": {"status": "ENABLED", "groupName": "/atr/codebuild"}}}
    if cb.batch_get_projects(names=[PROJECT])["projects"]:
        cb.update_project(**spec)
    else:
        for i in range(6):         # a just-created role can take a moment before CodeBuild may assume it
            try:
                cb.create_project(**spec)
                break
            except cb.exceptions.InvalidInputException as e:
                if i == 5:
                    raise
                print("waiting for the role to propagate:", str(e)[:80])
                time.sleep(10)
    print("project", PROJECT, "ready")
    if "--no-start" not in sys.argv:
        kw = {}
        if "--agents-only" in sys.argv:     # rebuild the agents' image and update the runtime; the stack is not touched
            kw["buildspecOverride"] = (Path(__file__).parent / "buildspec_agents.yml").read_text()
        b = cb.start_build(projectName=PROJECT, **kw)["build"]
        print("BUILD_ID=" + b["id"])


if __name__ == "__main__":
    main()
