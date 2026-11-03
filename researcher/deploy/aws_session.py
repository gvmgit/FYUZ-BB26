"""A boto3 session that keeps itself fresh from the `atr` CLI profile (`aws login` gives short-lived credentials; the
project's botocore cannot read that provider directly). Long-running tools (live view, dashboard) call `client()`/
`resource()` each time and get a session renewed a few minutes before expiry.
Region: ATR_DEPLOY_REGION, else AWS_REGION, else us-east-2.
"""
import json
import os
import subprocess
import time
from datetime import datetime

import truststore

truststore.inject_into_ssl()           # corporate TLS inspection: trust the Windows certificate store
import boto3  # noqa: E402

AWS_CLI = r"C:\Program Files\Amazon\AWSCLIV2\aws.exe"
REGION = os.environ.get("ATR_DEPLOY_REGION") or os.environ.get("AWS_REGION") or "us-east-2"
_state = {"session": None, "expires": 0.0}


def session() -> boto3.Session:
    if _state["session"] is None or time.time() > _state["expires"] - 300:
        env = dict(os.environ, AWS_CA_BUNDLE=os.path.expanduser(r"~\.aws\ca-bundle.pem"))
        for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            env.pop(k, None)               # let the CLI refresh from the login, not echo stale env credentials
        out = subprocess.run([AWS_CLI, "configure", "export-credentials", "--profile", "atr", "--format", "process"],
                             capture_output=True, text=True, env=env, check=True).stdout
        c = json.loads(out)
        _state["session"] = boto3.Session(aws_access_key_id=c["AccessKeyId"], aws_secret_access_key=c["SecretAccessKey"],
                                          aws_session_token=c["SessionToken"], region_name=REGION)
        exp = c.get("Expiration")
        _state["expires"] = (datetime.fromisoformat(exp.replace("Z", "+00:00")).timestamp() if exp
                             else time.time() + 900)
    return _state["session"]


def client(name: str):
    return session().client(name)


def resource(name: str):
    return session().resource(name)


def expire_now():
    """Call after an ExpiredToken error to force a refresh on the next request."""
    _state["expires"] = 0.0
