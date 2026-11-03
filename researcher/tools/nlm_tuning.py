"""NLM tuning unit: LoRA fine-tune of the OTel network language model, base and tuned scored on one frozen test set.

    python tools/nlm_tuning.py start [--instance ml.g6.2xlarge]     freeze + declare (S3, before the run), launch the job
    python tools/nlm_tuning.py finish --run-id <run id>             fetch the predictions, grade, write items and record

The test set (finetune/data/test.jsonl, split by fault setting on 28 Oct, disjoint from train) is copied to
evidence/frozen/nlm_test_v1.jsonl and its sha256 goes into the declaration before training starts. The base model is read
from the team bucket (models/otel-llm-1.7b-it/, copied from the event bucket); the job (finetune/train_eval_lora.py)
answers every test item with the base model, trains the LoRA adapter, answers again with the tuned model, and saves both
answers per item. The rule-based grader (atr/grader.py grade_nlm_tuning) scores them; no model grades.
"""
import argparse
import io
import json
import shutil
import sys
import tarfile
import time

from _common import ROOT, load_env

load_env()

import boto3  # noqa: E402

from atr import grader  # noqa: E402
from atr.checkpoint import new_run_id  # noqa: E402
from atr.evidence import Evidence  # noqa: E402

REGION, ACCOUNT = "us-east-2", "651343917891"
BUCKET = f"bb26-p-team09-data-{ACCOUNT}-use2"
BASE_MODEL = f"s3://{BUCKET}/models/otel-llm-1.7b-it/"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/bb26-SageMakerExecution"
IMAGE = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker"
INSTANCES = ["ml.g6.2xlarge", "ml.g5.2xlarge", "ml.g5.4xlarge"]
TEST_SET = "nlm_test_v1.jsonl"
TARGET = {"metric": "joint_accuracy_gain", "value": 0.10}
REQUIREMENTS = "transformers>=4.51,<4.57\npeft>=0.14,<0.18\naccelerate>=1.0\n"
FT = ROOT / "finetune"


def job_name(run_id: str) -> str:
    return ("atr-" + run_id.replace("_", "-"))[:63]


def start(a):
    ev = Evidence()
    frozen = ev.root / "frozen" / TEST_SET
    if not frozen.exists():
        shutil.copyfile(FT / "data" / "test.jsonl", frozen)
    run_id = new_run_id("nlm_tuning")
    decl = ev.declare(run_id, "nlm_tuning", TARGET, grader.NAME, "otel-llm-1.7b-it", test_set=TEST_SET,
                      extra={"base_model": BASE_MODEL, "method": "LoRA r=16 on attention and MLP projections, 3 epochs",
                             "train": "finetune/data/train.jsonl (suspect_ranking + claim_triage)",
                             "scored": "suspect_ranking items of the frozen test set: kind and cell must both match"})
    print(f"declared {run_id}: {decl['target']} -> {decl['location']}")

    s3 = boto3.client("s3", region_name=REGION)
    prefix = f"atr-finetune/{job_name(run_id)}"
    s3.upload_file(str(FT / "data" / "train.jsonl"), BUCKET, f"{prefix}/data/train/train.jsonl")
    s3.upload_file(str(frozen), BUCKET, f"{prefix}/data/test/{TEST_SET}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(FT / "train_eval_lora.py", arcname="train_eval_lora.py")
        info = tarfile.TarInfo("requirements.txt"); body = REQUIREMENTS.encode(); info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/code/sourcedir.tar.gz", Body=buf.getvalue())

    def channel(name, uri):
        return {"ChannelName": name, "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": uri,
                                                                        "S3DataDistributionType": "FullyReplicated"}}}

    sm = boto3.client("sagemaker", region_name=REGION)
    last = None
    for inst in ([a.instance] if a.instance else INSTANCES):
        try:
            sm.create_training_job(
                TrainingJobName=job_name(run_id), RoleArn=ROLE,
                AlgorithmSpecification={"TrainingImage": IMAGE, "TrainingInputMode": "File"},
                InputDataConfig=[channel("model", BASE_MODEL), channel("train", f"s3://{BUCKET}/{prefix}/data/train/"),
                                 channel("test", f"s3://{BUCKET}/{prefix}/data/test/")],
                OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{prefix}/output/"},
                ResourceConfig={"InstanceType": inst, "InstanceCount": 1, "VolumeSizeInGB": 100},
                StoppingCondition={"MaxRuntimeInSeconds": 3 * 3600},
                HyperParameters={"sagemaker_program": "train_eval_lora.py",
                                 "sagemaker_submit_directory": f"s3://{BUCKET}/{prefix}/code/sourcedir.tar.gz",
                                 "sagemaker_region": REGION, "epochs": "3"},
                Tags=[{"Key": "run_id", "Value": run_id}])
            print(f"training job {job_name(run_id)} on {inst}")
            return
        except sm.exceptions.ClientError as e:
            last = e
            print(f"{inst}: {str(e)[:220]}")
    raise SystemExit(f"could not start the job: {last}")


def finish(a):
    ev = Evidence()
    run_id = a.run_id
    decl = ev.load_declared(run_id)
    sm = boto3.client("sagemaker", region_name=REGION)
    while True:
        j = sm.describe_training_job(TrainingJobName=job_name(run_id))
        status = j["TrainingJobStatus"]
        print(time.strftime("%H:%M:%S"), status, j.get("SecondaryStatus"), flush=True)
        if status in ("Completed", "Failed", "Stopped") or not a.wait:
            break
        time.sleep(60)
    if status in ("Failed", "Stopped"):            # a declared run keeps its record, failed or not
        ev.items(run_id, [])
        result = grader.grade_nlm_tuning([], decl["target"])
        ev.record(run_id, {"unit": "nlm_tuning", "target": decl["target"], "grader": grader.NAME, "result": result,
                           "pass": False, "training_job": job_name(run_id), "job_status": status,
                           "failure_reason": j.get("FailureReason", "")[:1000]})
        raise SystemExit(f"job {status}, recorded as FAIL: {j.get('FailureReason', '')[:500]}")
    if status != "Completed":
        raise SystemExit(f"job still {status}; run finish again (or with --wait)")
    uri = j["ModelArtifacts"]["S3ModelArtifacts"]
    bucket, key = uri[5:].split("/", 1)
    blob = boto3.client("s3", region_name=REGION).get_object(Bucket=bucket, Key=key)["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        preds = [json.loads(x) for x in tar.extractfile("predictions.jsonl").read().decode().splitlines() if x.strip()]
    rows = ev.items(run_id, preds)
    result = grader.grade_nlm_tuning(rows, decl["target"])
    rec = ev.record(run_id, {"unit": "nlm_tuning", "target": decl["target"], "grader": grader.NAME, "result": result,
                             "pass": result["pass"], "training_job": job_name(run_id), "adapter": uri})
    print(json.dumps({k: rec["result"][k] for k in ("metric", "value", "base_joint_accuracy", "tuned_joint_accuracy", "n", "pass")}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start"); s.add_argument("--instance")
    f = sub.add_parser("finish"); f.add_argument("--run-id", required=True); f.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    {"start": start, "finish": finish}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
