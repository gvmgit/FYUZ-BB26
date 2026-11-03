"""Launch the LoRA fine-tuning job on SageMaker (script mode).

    set ATR_TRAIN_ROLE_ARN=<SageMaker execution role>
    set ATR_TRAIN_IMAGE=<Hugging Face PyTorch training container image URI for your region>
    set ATR_TRAIN_BUCKET=<S3 bucket for data, code and output>
    python finetune/launch_sagemaker.py --model-id <base model> --data finetune/data

Uploads the dataset and this folder's code to S3, then creates the training job. The `create_training_job` request is
checked against botocore's service model (scripts/check_finetune.py); the job itself needs your account, a GPU quota and a
base model you can download.
"""
import argparse
import os
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def package_code(dest: Path) -> Path:
    """tar.gz of the training code (train_lora.py, data_utils.py, requirements.txt) for SageMaker script mode."""
    with tarfile.open(dest, "w:gz") as tar:
        for name in ("train_lora.py", "data_utils.py", "requirements.txt"):
            tar.add(HERE / name, arcname=name)
    return dest


def build_request(job_name, role_arn, image, bucket, model_id, instance_type="ml.g5.2xlarge", epochs=3,
                  with_validation=True) -> dict:
    prefix = f"s3://{bucket}/atr-finetune/{job_name}"

    def channel(name, uri):
        return {"ChannelName": name, "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": uri, "S3DataDistributionType": "FullyReplicated"}}}

    channels = [channel("train", f"{prefix}/data/train/")]
    if with_validation:
        channels.append(channel("validation", f"{prefix}/data/validation/"))
    return {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {"TrainingImage": image, "TrainingInputMode": "File"},
        "InputDataConfig": channels,
        "OutputDataConfig": {"S3OutputPath": f"{prefix}/output/"},
        "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 200},
        "StoppingCondition": {"MaxRuntimeInSeconds": 4 * 3600},
        "HyperParameters": {"sagemaker_program": "train_lora.py",
                            "sagemaker_submit_directory": f"{prefix}/code/sourcedir.tar.gz",
                            "model-id": model_id, "epochs": str(epochs)},
    }


def main():
    import boto3
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--instance-type", default="ml.g5.2xlarge")
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    bucket = os.environ["ATR_TRAIN_BUCKET"]
    name = "atr-lora-" + time.strftime("%Y%m%d-%H%M%S")
    s3 = boto3.client("s3")
    prefix = f"atr-finetune/{name}"
    s3.upload_file(str(Path(a.data) / "train.jsonl"), bucket, f"{prefix}/data/train/train.jsonl")
    has_val = (Path(a.data) / "val.jsonl").exists()
    if has_val:
        s3.upload_file(str(Path(a.data) / "val.jsonl"), bucket, f"{prefix}/data/validation/val.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        s3.upload_file(str(package_code(Path(tmp) / "sourcedir.tar.gz")), bucket, f"{prefix}/code/sourcedir.tar.gz")
    req = build_request(name, os.environ["ATR_TRAIN_ROLE_ARN"], os.environ["ATR_TRAIN_IMAGE"], bucket, a.model_id,
                        a.instance_type, a.epochs, has_val)
    boto3.client("sagemaker").create_training_job(**req)
    print(f"started training job {name}; adapter will be at {req['OutputDataConfig']['S3OutputPath']}")


if __name__ == "__main__":
    main()
