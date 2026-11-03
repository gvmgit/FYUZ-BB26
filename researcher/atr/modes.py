"""The local/AWS toggle.

    ATR_MODE=local        everything runs on this machine (default)
    ATR_MODE=aws          every service below uses its AWS implementation
    ATR_<SERVICE>=local|aws   overrides one service (for partial bring-up), e.g. ATR_STORE=aws

Services and what each needs when set to aws (environment variables):
    model             Bedrock model for the Strands agents          ATR_MODEL_ID  [ATR_REGION]
    gateway           AgentCore Gateway MCP endpoint for run_experiment  ATR_GATEWAY_URL  [ATR_GATEWAY_TOKEN]
    code_interpreter  AgentCore Code Interpreter for the Critic's statistics  [ATR_CODE_INTERPRETER_ID]
    store             DynamoDB finding-card table                   ATR_DDB_TABLE
    audit             DynamoDB audit table                          ATR_AUDIT_TABLE
    scheduler         SQS queue for queued/overnight investigations  ATR_SQS_URL
    nlm               network language model endpoint (SageMaker)   ATR_NLM_ENDPOINT   (optional second opinion)
    kiro              headless Kiro for infrastructure-as-code       ATR_KIRO_CMD       (deploy-time only)
    stream            Kinesis Data Streams for KPI windows and alarms  ATR_KINESIS_PREFIX
    loglake           S3 log lake for decoded interface logs        ATR_LAKE_BUCKET
    features          Glue tables queried with Athena               ATR_GLUE_DATABASE  ATR_ATHENA_OUTPUT
    search            OpenSearch Serverless for log text + vectors  ATR_OPENSEARCH_ENDPOINT
    graph             Amazon Neptune finding graph (openCypher)     ATR_NEPTUNE_ENDPOINT
    protocol          Step Functions experiment protocol            ATR_STATE_MACHINE_ARN
    lab               lab executor behind AgentCore Gateway         ATR_LAB_GATEWAY_URL
    model_import      Bedrock Custom Model Import for the tuned NLM  ATR_MODEL_IMPORT_ROLE  ATR_MODEL_BUCKET
    policy            Verified Permissions (Cedar) checkpoint + registry  ATR_POLICY_STORE_ID  ATR_REGISTRY_TABLE
    evidence          evidence chain in the team data bucket       ATR_EVIDENCE_BUCKET
"""
import os
from dataclasses import dataclass
from typing import Dict, List

SERVICES = ("model", "gateway", "code_interpreter", "store", "audit", "scheduler", "nlm", "kiro",
            "stream", "loglake", "features", "search", "graph", "protocol", "lab", "model_import", "policy", "evidence")

REQUIRED_ENV: Dict[str, List[str]] = {
    "model": ["ATR_MODEL_ID"],
    "gateway": ["ATR_GATEWAY_URL"],
    "code_interpreter": [],
    "store": ["ATR_DDB_TABLE"],
    "audit": ["ATR_AUDIT_TABLE"],
    "scheduler": ["ATR_SQS_URL"],
    "nlm": ["ATR_NLM_ENDPOINT"],
    "kiro": ["ATR_KIRO_CMD"],
    "stream": ["ATR_KINESIS_PREFIX"],
    "loglake": ["ATR_LAKE_BUCKET"],
    "features": ["ATR_GLUE_DATABASE", "ATR_ATHENA_OUTPUT"],
    "search": ["ATR_OPENSEARCH_ENDPOINT"],
    "graph": ["ATR_NEPTUNE_ENDPOINT"],
    "protocol": ["ATR_STATE_MACHINE_ARN"],
    "lab": ["ATR_LAB_GATEWAY_URL"],
    "model_import": ["ATR_MODEL_IMPORT_ROLE", "ATR_MODEL_BUCKET"],
    "policy": ["ATR_POLICY_STORE_ID", "ATR_REGISTRY_TABLE"],
    "evidence": ["ATR_EVIDENCE_BUCKET"],
}


@dataclass
class Services:
    mode: str
    choice: Dict[str, str]

    def aws(self, service: str) -> bool:
        return self.choice[service] == "aws"

    def missing_env(self) -> Dict[str, List[str]]:
        return {s: [v for v in REQUIRED_ENV[s] if not os.environ.get(v)]
                for s in SERVICES if self.aws(s) and any(not os.environ.get(v) for v in REQUIRED_ENV[s])}

    @property
    def region(self) -> str:
        return os.environ.get("ATR_REGION") or os.environ.get("AWS_REGION") or "us-east-2"


def resolve(mode: str = None) -> Services:
    mode = (mode or os.environ.get("ATR_MODE") or "local").lower()
    if mode not in ("local", "aws"):
        raise ValueError(f"ATR_MODE must be 'local' or 'aws', got '{mode}'")
    choice = {}
    for s in SERVICES:
        override = os.environ.get(f"ATR_{s.upper()}")
        if override and override.lower() not in ("local", "aws"):
            raise ValueError(f"ATR_{s.upper()} must be 'local' or 'aws', got '{override}'")
        choice[s] = (override or mode).lower()
    return Services(mode, choice)
