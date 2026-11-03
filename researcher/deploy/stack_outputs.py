"""Write the deployed stack's outputs in the same shape `cdk deploy --outputs-file` produces.

    python deploy/stack_outputs.py deploy/cdk/outputs.json
"""
import json
import sys

import boto3

out = sys.argv[1] if len(sys.argv) > 1 else "deploy/cdk/outputs.json"
stack = boto3.client("cloudformation").describe_stacks(StackName="TelecomResearcher")["Stacks"][0]
data = {"TelecomResearcher": {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}}
json.dump(data, open(out, "w"), indent=1)
print(f"{stack['StackStatus']}: {len(data['TelecomResearcher'])} outputs -> {out}")
