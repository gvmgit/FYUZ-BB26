"""The network language model on SageMaker: OTel-LLM-1.7B-IT (the event's OTel model, Qwen3-1.7B fine-tuned on telecom
data) served by the AWS LMI (vLLM) container, as endpoint `atr-otel-nlm`. The Scientist asks it for a context-grounded
second opinion through the checkpoint (InvokeModel on Model::otel-llm-1.7b-it).

    python deploy/nlm_endpoint.py create [--instance ml.g6.2xlarge]     copy is a prerequisite: models/otel-llm-1.7b-it/
    python deploy/nlm_endpoint.py status | test | delete                 in the team bucket (us-east-2)

The model is read from S3 as uncompressed files; nothing is downloaded to the laptop.
"""
import json
import os
import sys
import time

import boto3

REGION = os.environ.get("ATR_REGION", "us-east-2")
ACCOUNT = "651343917891"
NAME = "atr-otel-nlm"
MODEL_ID = "otel-llm-1.7b-it"
MODEL_S3 = f"s3://bb26-p-team09-data-{ACCOUNT}-use2/models/otel-llm-1.7b-it/"
IMAGE = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/djl-inference:0.36.0-lmi28.0.0-cu130"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/bb26-SageMakerExecution"
INSTANCES = ["ml.g6.2xlarge", "ml.g5.2xlarge", "ml.g6.xlarge", "ml.g5.xlarge"]


def prompt(context: str, question: str) -> str:
    """Qwen3 chat format with thinking off: a short, context-grounded answer."""
    return (f"<|im_start|>system\nYou are a radio network optimisation expert. Answer only from the context.<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def create(sm, instance=None):
    try:
        sm.create_model(ModelName=NAME, ExecutionRoleArn=ROLE, PrimaryContainer={
            "Image": IMAGE,
            "ModelDataSource": {"S3DataSource": {"S3Uri": MODEL_S3, "S3DataType": "S3Prefix", "CompressionType": "None"}},
            "Environment": {"OPTION_ROLLING_BATCH": "vllm", "OPTION_DTYPE": "bf16", "OPTION_MAX_MODEL_LEN": "8192",
                            "OPTION_GPU_MEMORY_UTILIZATION": "0.85", "TENSOR_PARALLEL_DEGREE": "1"}})
    except sm.exceptions.ClientError as e:
        if "Cannot create already existing model" not in str(e):
            raise
    last = None
    for inst in ([instance] if instance else INSTANCES):
        cfg = f"{NAME}-{inst.replace('.', '-')}"
        try:
            sm.create_endpoint_config(EndpointConfigName=cfg, ProductionVariants=[{
                "VariantName": "main", "ModelName": NAME, "InstanceType": inst, "InitialInstanceCount": 1,
                "ContainerStartupHealthCheckTimeoutInSeconds": 900, "ModelDataDownloadTimeoutInSeconds": 1200}])
        except sm.exceptions.ClientError as e:
            if "already existing" not in str(e):
                raise
        try:
            sm.create_endpoint(EndpointName=NAME, EndpointConfigName=cfg)
            print(f"creating endpoint {NAME} on {inst}")
            return inst
        except sm.exceptions.ClientError as e:
            last = e
            print(f"{inst}: {str(e)[:200]}")
    raise SystemExit(f"no instance type accepted: {last}")


def wait(sm):
    while True:
        d = sm.describe_endpoint(EndpointName=NAME)
        print(time.strftime("%H:%M:%S"), d["EndpointStatus"], d.get("FailureReason", ""), flush=True)
        if d["EndpointStatus"] in ("InService", "Failed"):
            return d["EndpointStatus"]
        time.sleep(30)


def test():
    rt = boto3.client("sagemaker-runtime", region_name=REGION)
    ctx = ("Zone Z5: field best-server RSRP averages -107 dBm where the inventory predicts -97 dBm. "
           "Per-cell anomaly scores: cell 2: 0.91, cell 4: 0.35, cell 1: 0.10. Cell 2 inventory: azimuth 100 deg, tilt 10 deg.")
    body = {"inputs": prompt(ctx, "Which cell and which setting (tilt, azimuth or cell off) is most likely wrong? One sentence."),
            "parameters": {"max_new_tokens": 80, "temperature": 0.2}}
    t0 = time.time()
    r = rt.invoke_endpoint(EndpointName=NAME, ContentType="application/json", Body=json.dumps(body))
    print(f"{time.time() - t0:.1f}s", r["Body"].read().decode()[:600])


def main():
    sm = boto3.client("sagemaker", region_name=REGION)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "create":
        inst = sys.argv[sys.argv.index("--instance") + 1] if "--instance" in sys.argv else None
        create(sm, inst)
        wait(sm)
    elif cmd == "status":
        wait(sm)
    elif cmd == "test":
        test()
    elif cmd == "delete":
        d = sm.describe_endpoint(EndpointName=NAME)
        sm.delete_endpoint(EndpointName=NAME)
        sm.delete_endpoint_config(EndpointConfigName=d["EndpointConfigName"])
        sm.delete_model(ModelName=NAME)
        print("deleted endpoint, config and model")


if __name__ == "__main__":
    main()
