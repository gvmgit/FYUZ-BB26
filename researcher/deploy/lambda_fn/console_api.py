"""Review console API (API Gateway REST, Cognito authorizer -> this Lambda). Same routes as the local server in
atr/serve.py, so console/app.js works against either:

  GET  /api/findings            finding cards (DynamoDB)
  GET  /api/graph?kind=tilt     finding-graph summary and the recommended test strategy for a fault class (Neptune)
  POST /api/verdict             {"card_id", "verdict": accept|reject|request_experiment, "comment", "request"}
                                the reviewer's name comes from the Cognito token, not the request body

STATUS: written; the route logic is shared with the local server and tested there; NOT deployed.
"""
import json
import os

os.environ.setdefault("ATR_RUNS_DIR", "/tmp/runs")

from atr import aws_adapters as aws  # noqa: E402
from atr.graph import make_graph  # noqa: E402
from atr.serve import api_graph, api_verdict  # noqa: E402


def _resp(code, body):
    return {"statusCode": code, "headers": {"Content-Type": "application/json",
                                            "Access-Control-Allow-Origin": os.environ.get("ATR_CONSOLE_ORIGIN", "*")},
            "body": json.dumps(body, default=str)}


def handler(event, context=None):
    store, path, method = aws.DynamoDBStore(), event.get("path", ""), event.get("httpMethod", "GET")
    try:
        if method == "GET" and path.endswith("/api/findings"):
            return _resp(200, store.list())
        if method == "GET" and path.endswith("/api/graph"):
            kind = (event.get("queryStringParameters") or {}).get("kind", "tilt")
            return _resp(200, api_graph(make_graph(), kind))
        if method == "POST" and path.endswith("/api/verdict"):
            claims = event.get("requestContext", {}).get("authorizer", {}).get("claims", {})
            who = claims.get("email") or claims.get("cognito:username")
            if not who:
                return _resp(401, {"error": "no reviewer identity in the token"})
            body = json.loads(event.get("body") or "{}")
            return _resp(200, api_verdict(store, body, who, scheduler=aws.SqsScheduler(), audit=aws.DynamoAudit()))
    except (KeyError, ValueError, PermissionError) as e:
        return _resp(400, {"error": str(e)})
    return _resp(404, {"error": f"no route {method} {path}"})
