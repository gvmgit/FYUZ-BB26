"""run_experiment as a Lambda function: the target behind AgentCore Gateway.

AgentCore Gateway invokes a Lambda target with the tool arguments as the event, so `event` is exactly the
run_experiment input {"config": ..., "seeds": [...]}. The function loads the twin (baked into the image) and the
current field measurement log from S3, runs the experiment, uploads any heatmap to S3, and returns the result dict.
The same ExperimentRunner validation (tilt range, one setting at a time) applies here, so the safety bounds hold even
if a caller bypasses the agents.
"""
import io
import os
import uuid

import pandas as pd

_state = {}


def _default_field_loader() -> pd.DataFrame:
    import boto3
    body = boto3.client("s3").get_object(Bucket=os.environ["ATR_FIELD_BUCKET"], Key="field/current.csv")["Body"].read()
    return pd.read_csv(io.BytesIO(body))


def _default_uploader(path: str, key: str) -> str:
    import boto3
    bucket = os.environ["ATR_FIELD_BUCKET"]
    boto3.client("s3").upload_file(path, bucket, key)
    return f"s3://{bucket}/{key}"


def _field_version() -> str:
    """ETag of the current field log: a newly planted fault replaces it, and a warm Lambda must not keep the old one."""
    import boto3
    return boto3.client("s3").head_object(Bucket=os.environ["ATR_FIELD_BUCKET"], Key="field/current.csv")["ETag"]


def _runner(field_loader, version=None):
    if "runner" not in _state or _state.get("version") != version:
        from atr.experiment import ExperimentRunner
        from atr.twin import RFTwin, default_inventory
        if "twin" not in _state:
            _state["twin"] = RFTwin.load_or_build()          # models are baked into the image; never trains here
        runner = ExperimentRunner(_state["twin"], default_inventory(), field_loader())
        runner.id_prefix = uuid.uuid4().hex[:6] + "-"
        _state.update(runner=runner, version=version)
    return _state["runner"]


def handler(event, context=None, field_loader=None, uploader=None):
    runner = _runner(field_loader or _default_field_loader, None if field_loader else _field_version())
    config_, seeds = event["config"], [int(s) for s in event["seeds"]]
    runner.validate(config_)
    result = runner.run(config_, seeds)
    if result.get("heatmap"):
        key = f"heatmaps/{result['run_id']}.png"
        result["heatmap"] = (uploader or _default_uploader)(result["heatmap"], key)
    return result
