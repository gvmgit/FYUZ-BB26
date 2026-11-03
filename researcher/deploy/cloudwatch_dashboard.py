"""Create (or update) the CloudWatch dashboard `TelecomResearcher` for the deployed stack.

    python deploy/cloudwatch_dashboard.py          -> prints the console URL
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import aws_session  # noqa: E402

R = aws_session.REGION
o = json.load(open(Path(__file__).parent / "cdk" / "outputs.json"))["TelecomResearcher"]
fn = lambda arn: arn.split(":")[-1]          # noqa: E731
twin, lab = fn(o["RunExperimentLambdaArn"]), fn(o["LabExecutorLambdaArn"])
runtime_logs = next(g["logGroupName"] for g in aws_session.client("logs").describe_log_groups(
    logGroupNamePrefix="/aws/bedrock-agentcore/runtimes/telecom_researcher")["logGroups"])


def metric(title, metrics, x, y, w=8, h=6, stat="Sum", period=60, view="timeSeries"):
    return {"type": "metric", "x": x, "y": y, "width": w, "height": h,
            "properties": {"title": title, "region": R, "metrics": metrics, "stat": stat, "period": period, "view": view,
                           "stacked": False}}


widgets = [
    {"type": "text", "x": 0, "y": 0, "width": 24, "height": 2, "properties": {"markdown":
        "# Autonomous Telecom Researcher - live on AWS\nStrands agents in **Bedrock AgentCore Runtime** (Claude Sonnet 5 on Bedrock) - "
        "experiments through **AgentCore Gateway** to the MAVERIC twin **Lambda** - findings and audit in **DynamoDB** - "
        "protocol in **Step Functions**"}},
    metric("Experiments: twin Lambda invocations", [["AWS/Lambda", "Invocations", "FunctionName", twin, {"label": "run_experiment"}],
                                                    ["AWS/Lambda", "Invocations", "FunctionName", lab, {"label": "lab_execute"}]], 0, 2),
    metric("Twin Lambda duration (ms)", [["AWS/Lambda", "Duration", "FunctionName", twin, {"stat": "Average", "label": "average"}],
                                         ["AWS/Lambda", "Duration", "FunctionName", twin, {"stat": "p90", "label": "p90"}]], 8, 2),
    metric("Lambda errors and throttles", [["AWS/Lambda", "Errors", "FunctionName", twin, {"label": "twin errors"}],
                                           ["AWS/Lambda", "Errors", "FunctionName", lab, {"label": "lab errors"}],
                                           ["AWS/Lambda", "Throttles", "FunctionName", twin, {"label": "twin throttles"}]], 16, 2),
    metric("Bedrock: model invocations", [[{"expression": "SEARCH('{AWS/Bedrock,ModelId} MetricName=\"Invocations\"', 'Sum', 60)",
                                            "id": "inv", "label": ""}]], 0, 8),
    metric("Bedrock: tokens", [[{"expression": "SEARCH('{AWS/Bedrock,ModelId} MetricName=\"InputTokenCount\"', 'Sum', 60)", "id": "tin", "label": "input"}],
                               [{"expression": "SEARCH('{AWS/Bedrock,ModelId} MetricName=\"OutputTokenCount\"', 'Sum', 60)", "id": "tout", "label": "output"}]], 8, 8),
    metric("AgentCore (Runtime and Gateway)", [[{"expression": "SEARCH('{AWS/Bedrock-AgentCore} Invocations', 'Sum', 60)", "id": "ac", "label": ""}]], 16, 8),
    metric("DynamoDB writes: findings and audit", [["AWS/DynamoDB", "ConsumedWriteCapacityUnits", "TableName", o["FindingsTable"], {"label": "findings"}],
                                                   ["AWS/DynamoDB", "ConsumedWriteCapacityUnits", "TableName", o["AuditTable"], {"label": "audit trail"}]], 0, 14),
    metric("Step Functions: experiment protocol", [["AWS/States", "ExecutionsStarted", "StateMachineArn", o["StateMachineArn"], {"label": "started"}],
                                                   ["AWS/States", "ExecutionsSucceeded", "StateMachineArn", o["StateMachineArn"], {"label": "succeeded"}],
                                                   ["AWS/States", "ExecutionsFailed", "StateMachineArn", o["StateMachineArn"], {"label": "failed"}]], 8, 14),
    metric("Experiments today", [["AWS/Lambda", "Invocations", "FunctionName", twin, {"label": "run_experiment calls"}]], 16, 14,
           stat="Sum", period=86400, view="singleValue"),
    {"type": "log", "x": 0, "y": 20, "width": 24, "height": 6, "properties": {
        "title": "AgentCore Runtime: latest agent log lines", "region": R, "view": "table",
        "query": f"SOURCE '{runtime_logs}' | fields @timestamp, @message | sort @timestamp desc | limit 25"}},
]
aws_session.client("cloudwatch").put_dashboard(DashboardName="TelecomResearcher", DashboardBody=json.dumps({"widgets": widgets}))
print(f"https://{R}.console.aws.amazon.com/cloudwatch/home?region={R}#dashboards/dashboard/TelecomResearcher")
