"""Shared setup for the evidence tools: import path, the deployment's settings, fresh AWS credentials."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENV_FILE = ROOT / "deploy" / "cdk" / "atr_aws.env"
TEAM_BUCKET = "bb26-p-team09-data-651343917891-use2"
REGION = "us-east-2"


def load_env(path: Path = ENV_FILE) -> None:
    """ATR_* settings written by deploy/post_deploy.py (no secrets in it). Values already set win."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    os.environ.setdefault("ATR_EVIDENCE_BUCKET", TEAM_BUCKET)
    os.environ.setdefault("ATR_REGION", REGION)
    os.environ.setdefault("AWS_DEFAULT_REGION", REGION)


def refresh_credentials() -> None:
    """On the team laptop (ATR_LOCAL_LOGIN=1) renew short-lived credentials from the `aws login` profile before each
    long step; elsewhere (CloudShell, a judge's session) boto3's own credential chain is used unchanged."""
    if os.environ.get("ATR_LOCAL_LOGIN") != "1":
        return
    import json
    import subprocess
    import boto3
    env = {k: v for k, v in os.environ.items() if k not in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")}
    out = subprocess.run([os.environ.get("ATR_AWS_CLI", r"C:\Program Files\Amazon\AWSCLIV2\aws.exe"), "configure",
                          "export-credentials", "--profile", os.environ.get("ATR_AWS_PROFILE", "atr"), "--format", "process"],
                         capture_output=True, text=True, env=env, check=True).stdout
    c = json.loads(out)
    os.environ.update(AWS_ACCESS_KEY_ID=c["AccessKeyId"], AWS_SECRET_ACCESS_KEY=c["SecretAccessKey"],
                      AWS_SESSION_TOKEN=c["SessionToken"])
    boto3.setup_default_session(region_name=os.environ["ATR_REGION"])
