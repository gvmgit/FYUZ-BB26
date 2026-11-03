"""AWS implementations of every swappable service. Imported only when a service is set to 'aws'.

STATUS: written against the SDKs installed in this project's venv (strands-agents 1.62, bedrock-agentcore 1.25,
boto3), but NOT executed against a real AWS account from this machine. Each adapter takes its AWS client as an
optional argument, so the logic around the calls is unit-tested with fakes (scripts/check_adapters.py); the calls
themselves need verifying in your account (the README's step-by-step deployment).
"""
import json
import os
import subprocess
import time
import uuid
from decimal import Decimal
from typing import Callable, List, Optional

from . import stats
from .gateway import ToolError
from .store import FindingStore


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"AWS mode needs environment variable {name}")
    return v


def _to_ddb(obj):
    """DynamoDB rejects floats; round-trip through Decimal."""
    return json.loads(json.dumps(obj, default=str), parse_float=Decimal)


def _from_ddb(obj):
    return json.loads(json.dumps(obj, default=lambda o: float(o) if isinstance(o, Decimal) else str(o)))


# --- Bedrock (Claude) via Strands ---------------------------------------------------------------------------------
def bedrock_model(region: str, role: Optional[str] = None, checkpoint=None):
    """Claude on Bedrock with Amazon Bedrock Guardrails on input and output (ATR_GUARDRAIL_ID / _VERSION), behind the
    checkpoint when one is given (every call is an InvokeModel decision for `role`)."""
    from strands.models import BedrockModel
    kw = {}
    if os.environ.get("ATR_MODEL_TEMPERATURE"):     # newer Claude models on Bedrock reject `temperature`; older ones accept it
        kw["temperature"] = float(os.environ["ATR_MODEL_TEMPERATURE"])
    if os.environ.get("ATR_GUARDRAIL_ID"):
        kw.update(guardrail_id=os.environ["ATR_GUARDRAIL_ID"], guardrail_version=os.environ.get("ATR_GUARDRAIL_VERSION", "DRAFT"),
                  guardrail_trace="enabled", guardrail_redact_input=False, guardrail_latest_message=True)
    model = BedrockModel(model_id=_env("ATR_MODEL_ID"), region_name=region, **kw)
    if checkpoint is not None:
        from .checkpoint import guarded
        model = guarded(model, role or "agent", checkpoint, _env("ATR_MODEL_ID"))
    return model


# --- Multimodal: the model reads the coverage heatmap ---------------------------------------------------------------
class HeatmapReader:
    """Asks a vision-capable model (Claude on Bedrock) to describe the best-server RSRP heatmap next to the KPI summary,
    so the Scientist's claims can use what the picture shows (weak lobes, coverage holes, mis-pointed sectors)."""

    def __init__(self, model_factory: Callable[[], object], s3=None):
        self.model_factory, self.s3 = model_factory, s3

    def _bytes(self, path: str) -> bytes:
        if str(path).startswith("s3://"):
            import boto3
            bucket, key = str(path)[5:].split("/", 1)
            return (self.s3 or boto3.client("s3")).get_object(Bucket=bucket, Key=key)["Body"].read()
        return open(path, "rb").read()

    def describe(self, heatmap_path: str, symptom_text: str) -> str:
        from strands import Agent
        agent = Agent(model=self.model_factory(), callback_handler=None, system_prompt=(
            "You inspect a mobile-network coverage heatmap (best-server RSRP, green strong, red weak; triangles are "
            "cell sites with the number of pixels they serve). Describe in 3 sentences what looks abnormal, which "
            "sites look responsible, and nothing you cannot see."))
        result = agent([{"text": f"KPI summary: {symptom_text}. What does the heatmap show?"},
                        {"image": {"format": "png", "source": {"bytes": self._bytes(heatmap_path)}}}])
        return str(result)


# --- AgentCore Gateway: run_experiment over MCP ---------------------------------------------------------------------
def _sigv4_auth_class():
    import httpx

    class SigV4HttpxAuth(httpx.Auth):
        """Signs each MCP request with SigV4 for a gateway that uses the AWS_IAM authorizer."""
        requires_request_body = True

        def __init__(self, region: str, service: str = "bedrock-agentcore", session=None):
            import boto3
            self.region, self.service = region, service
            self.creds = (session or boto3.Session()).get_credentials()

        def auth_flow(self, request):
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            headers = {k: v for k, v in request.headers.items() if k.lower() not in ("connection", "content-length")}
            aws_req = AWSRequest(method=request.method, url=str(request.url), data=request.content, headers=headers)
            SigV4Auth(self.creds.get_frozen_credentials(), self.service, self.region).add_auth(aws_req)
            request.headers.update(dict(aws_req.headers))
            yield request

    return SigV4HttpxAuth


def SigV4HttpxAuth(region: str, **kw):   # noqa: N802 - factory with a class-like name
    return _sigv4_auth_class()(region, **kw)


class RemoteGateway:
    """Same interface as gateway.Gateway, but run_experiment executes behind an AgentCore Gateway MCP endpoint
    (a Lambda target running the twin). Policy and audit still apply locally before the call leaves."""

    tool = "run_experiment"

    def __init__(self, url: str, token: Optional[str], audit, policy: Optional[Callable] = None, mcp_client=None):
        self.audit, self.policy, self.count = audit, policy, 0
        if mcp_client is None:
            from strands.tools.mcp import MCPClient
            if token:   # gateway created with a JWT authorizer
                mcp_client = MCPClient(url=url, headers={"Authorization": f"Bearer {token}"})
            else:       # gateway created with the AWS_IAM authorizer: sign requests with SigV4
                mcp_client = MCPClient(url=url, auth_provider=SigV4HttpxAuth(os.environ.get("ATR_REGION")
                                       or os.environ.get("AWS_REGION") or "us-east-2"))
        self.mcp = mcp_client
        self._tool_name = None
        self._started = False

    def _session(self):
        """Strands' MCPClient only talks inside a started session; start it once, on first use."""
        if not self._started:
            if hasattr(self.mcp, "start"):
                self.mcp.start()
            self._started = True
        return self.mcp

    def close(self):
        if self._started and hasattr(self.mcp, "stop"):
            self.mcp.stop(None, None, None)
        self._started = False

    def _resolve_tool(self) -> str:
        """Gateway prefixes tool names with the target name (e.g. 'twin___run_experiment'); find ours."""
        self._session()
        if self._tool_name is None:
            names = [t.tool_name for t in self.mcp.list_tools_sync()]
            match = [n for n in names if n.endswith(self.tool)]
            if not match:
                raise ToolError(f"gateway exposes no {self.tool} tool (has: {names})")
            self._tool_name = match[0]
        return self._tool_name

    def invoke(self, name: str, args: dict, agent: str):
        if name != self.tool:
            raise ToolError(f"unknown tool '{name}'")
        t0, error, result = time.time(), None, None
        try:
            if self.policy:
                self.policy(agent, name)
            res = self.mcp.call_tool_sync(tool_use_id=uuid.uuid4().hex, name=self._resolve_tool(), arguments=args)
            text = "".join(c.get("text", "") for c in res.get("content", []))
            if res.get("status") == "error":
                raise ToolError(text)
            # JSON turns integer dict keys (cell ids) into strings; the local gateway hands back ints. Restore them so
            # the agents see the same shapes either way (inventory and diagnostics are keyed by int cell id).
            result = json.loads(text, object_hook=lambda d: {int(k) if isinstance(k, str) and k.isdigit() else k: v
                                                             for k, v in d.items()})
            self.count += 1
            return result
        except Exception as e:  # noqa: BLE001
            error = str(e)
            raise ToolError(error) from e
        finally:
            self.audit.write({"kind": "tool_call", "agent": agent, "tool": name, "args": args, "ok": error is None,
                              "error": error, "run_id": (result or {}).get("run_id"),
                              "seconds": round(time.time() - t0, 3), "via": "agentcore-gateway"})


# --- AgentCore Code Interpreter: the Critic's statistics ---------------------------------------------------------
class CodeInterpreterStats:
    def __init__(self, region: str, identifier: Optional[str] = None, session_factory=None):
        self.region, self.identifier = region, identifier
        self._factory = session_factory
        self._cm = self._client = None

    def _session(self):
        if self._client is None:
            if self._factory is None:
                from bedrock_agentcore.tools.code_interpreter_client import code_session
                self._factory = lambda: code_session(self.region, identifier=self.identifier)
            self._cm = self._factory()
            self._client = self._cm.__enter__()
        return self._client

    def __call__(self, baseline: List[float], treatment: List[float]) -> dict:
        res = self._session().execute_code(stats.code_interpreter_script(baseline, treatment))
        return parse_interpreter_output(res)

    def close(self):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = self._client = None


def parse_interpreter_output(res) -> dict:
    """execute_code returns a response whose structuredContent/content carries stdout; find the JSON line."""
    texts = []
    stream = res.get("stream") if isinstance(res, dict) else None
    for event in (stream if stream is not None else [res]):
        result = event.get("result", event) if isinstance(event, dict) else {}
        if result.get("isError"):
            raise ToolError("code interpreter error: " + json.dumps(result.get("structuredContent") or result)[:500])
        if result.get("structuredContent", {}).get("stdout"):
            texts.append(result["structuredContent"]["stdout"])
        texts.extend(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))
    for line in reversed("\n".join(texts).splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ToolError("code interpreter returned no JSON result")


# --- DynamoDB: finding cards and audit trail -----------------------------------------------------------------------
class DynamoDBStore(FindingStore):
    def __init__(self, table=None, table_name: Optional[str] = None, region: Optional[str] = None):
        if table is None:
            import boto3
            table = boto3.resource("dynamodb", region_name=region).Table(table_name or _env("ATR_DDB_TABLE"))
        self.table = table

    def put(self, card: dict) -> None:
        self.table.put_item(Item=_to_ddb(card))

    def list(self) -> List[dict]:
        items, resp = [], self.table.scan()
        items += resp.get("Items", [])
        while "LastEvaluatedKey" in resp:
            resp = self.table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
            items += resp.get("Items", [])
        return sorted((_from_ddb(i) for i in items), key=lambda c: c.get("created", 0), reverse=True)

    def get(self, card_id: str) -> Optional[dict]:
        item = self.table.get_item(Key={"id": card_id}).get("Item")
        return _from_ddb(item) if item else None


class DynamoAudit:
    """Audit sink: partition key 'id' (uuid) so concurrent writers never collide."""

    def __init__(self, table=None, table_name: Optional[str] = None, region: Optional[str] = None):
        if table is None:
            import boto3
            table = boto3.resource("dynamodb", region_name=region).Table(table_name or _env("ATR_AUDIT_TABLE"))
        self.table = table

    def write(self, record: dict) -> None:
        self.table.put_item(Item=_to_ddb({"id": uuid.uuid4().hex, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "epoch": time.time(), **record}))


class CloudWatchAudit:
    """Audit sink: one JSON event per record in the /atr/audit log group, one stream per trace id. CloudWatch stamps
    each event's ingestion time, so a judge can compare it with the record's own time (Logs Insights, read-only)."""

    GROUP = "/atr/audit"

    def __init__(self, client=None, group: Optional[str] = None, region: Optional[str] = None):
        if client is None:
            import boto3
            client = boto3.client("logs", region_name=region)
        self.client, self.group, self._streams = client, group or self.GROUP, set()

    def _stream(self, name: str) -> str:
        name = name.replace(":", "-").replace("*", "-")[:500]
        if name not in self._streams:
            for create, kw in ((self.client.create_log_group, {"logGroupName": self.group}),
                               (self.client.create_log_stream, {"logGroupName": self.group, "logStreamName": name})):
                try:
                    create(**kw)
                except self.client.exceptions.ResourceAlreadyExistsException:
                    pass
            self._streams.add(name)
        return name

    def write(self, record: dict) -> None:
        stream = self._stream(str(record.get("trace_id") or record.get("run_id") or "untraced"))
        self.client.put_log_events(logGroupName=self.group, logStreamName=stream, logEvents=[
            {"timestamp": int(time.time() * 1000), "message": json.dumps(record, default=str)}])


class TeeAudit:
    def __init__(self, *sinks):
        self.sinks = sinks

    def write(self, record: dict) -> None:
        for s in self.sinks:
            s.write(record)


# --- Scheduler: queued / overnight investigations ------------------------------------------------------------------
class SqsScheduler:
    """Queue side of the scheduler. EventBridge (nightly rule in the CDK stack) and `atr.cli schedule` enqueue;
    an AgentCore Runtime worker consumes. `priority` maps to message group via a delay-free attribute."""

    def __init__(self, queue_url: Optional[str] = None, client=None, region: Optional[str] = None):
        if client is None:
            import boto3
            client = boto3.client("sqs", region_name=region)
        self.client, self.url = client, queue_url or _env("ATR_SQS_URL")

    def submit(self, job: dict, priority: int = 5) -> str:
        r = self.client.send_message(QueueUrl=self.url, MessageBody=json.dumps(job),
                                     MessageAttributes={"priority": {"DataType": "Number", "StringValue": str(priority)}})
        return r["MessageId"]


# --- Blind test in AWS: the teammate's hidden fault ------------------------------------------------------------------
def plant_fault(field_csv: str, truth: dict, s3=None, table=None, region: Optional[str] = None) -> dict:
    """Upload the field measurement log (what the run_experiment Lambda compares against) to S3 and store the truth in a
    separate table the agents' IAM roles cannot read. Only `reveal` reads it."""
    import boto3
    s3 = s3 or boto3.client("s3", region_name=region)
    table = table or boto3.resource("dynamodb", region_name=region).Table(_env("ATR_TRUTH_TABLE"))
    bucket = _env("ATR_FIELD_BUCKET")
    s3.put_object(Bucket=bucket, Key="field/current.csv", Body=field_csv.encode("utf-8"))
    table.put_item(Item=_to_ddb({"id": "current", "planted": time.time(), "truth": truth}))
    return {"bucket": bucket, "key": "field/current.csv"}


def read_truth(table=None, region: Optional[str] = None) -> Optional[dict]:
    import boto3
    table = table or boto3.resource("dynamodb", region_name=region).Table(_env("ATR_TRUTH_TABLE"))
    item = table.get_item(Key={"id": "current"}).get("Item")
    return _from_ddb(item)["truth"] if item else None


# --- Network language model (OTel-LLM-8.3B) as a second opinion ------------------------------------------------------
class NetworkLM:
    """Calls a SageMaker endpoint hosting the network language model. The request/response schema depends on how the
    model is deployed, so the payload builder and parser are overridable; defaults assume a text-generation container."""

    def __init__(self, endpoint: Optional[str] = None, client=None, region: Optional[str] = None):
        if client is None:
            import boto3
            client = boto3.client("sagemaker-runtime", region_name=region)
        self.client, self.endpoint = client, endpoint or _env("ATR_NLM_ENDPOINT")

    model_id = "otel-llm-1.7b-it"

    def second_opinion(self, symptom_text: str, suspicion: dict, inventory: Optional[dict] = None) -> str:
        """Context-grounded question to OTel-LLM-1.7B-IT (Qwen3 chat format, thinking off), served by the LMI container."""
        ctx = f"Field symptom: {symptom_text}\nPer-cell anomaly scores (higher = more suspicious): {suspicion}"
        if inventory:
            ctx += "\nInventory: " + "; ".join(f"cell {c}: tilt {v.get('tilt')} deg, azimuth {v.get('azimuth')} deg"
                                               for c, v in sorted(inventory.items()))
        prompt = (f"<|im_start|>system\nYou are a radio network optimisation expert. Answer only from the context.<|im_end|>\n"
                  f"<|im_start|>user\nContext:\n{ctx}\n\nQuestion: Which cell and which setting (tilt, azimuth or cell off) is "
                  f"most likely wrong, and why? Answer in two sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        resp = self.client.invoke_endpoint(EndpointName=self.endpoint, ContentType="application/json",
                                           Body=json.dumps({"inputs": prompt, "parameters": {"max_new_tokens": 120, "temperature": 0.2}}))
        body = json.loads(resp["Body"].read())
        text = body[0]["generated_text"] if isinstance(body, list) else body.get("generated_text", str(body)) if isinstance(body, dict) else str(body)
        return text.split("</think>")[-1].replace("<|im_end|>", "").strip()


# --- Headless Kiro: infrastructure-as-code generation (deploy time) ---------------------------------------------------
def kiro_generate(spec_path: str, out_dir: str, run: Callable = subprocess.run) -> dict:
    """Run headless Kiro on the infrastructure spec. The command comes from ATR_KIRO_CMD (e.g. the Kiro CLI invocation
    your environment uses); the spec path and output dir are exported as KIRO_SPEC / KIRO_OUT_DIR. Nothing about Kiro's
    flags is assumed here - verify the command against your Kiro installation."""
    cmd = _env("ATR_KIRO_CMD")
    env = {**os.environ, "KIRO_SPEC": spec_path, "KIRO_OUT_DIR": out_dir}
    p = run(cmd, shell=True, env=env, capture_output=True, text=True)
    return {"returncode": p.returncode, "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}


# --- Lab executor behind AgentCore Gateway (the HCLTech test automation API's slot) ---------------------------------
class RemoteLabGateway(RemoteGateway):
    """The lab_execute tool on AgentCore Gateway (a Lambda target running lab.LabExecutor, which ships its telemetry to
    Kinesis and S3 itself). Same policy and audit as locally."""
    tool = "lab_execute"


# --- Multimodal ingestion ------------------------------------------------------------------------------------------
def _batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


class KinesisStreams:
    """ingest.StreamBus on Kinesis Data Streams: one stream per record type, named <prefix>-kpi, <prefix>-alarms."""

    def __init__(self, prefix: Optional[str] = None, client=None, region: Optional[str] = None):
        if client is None:
            import boto3
            client = boto3.client("kinesis", region_name=region)
        self.client, self.prefix = client, prefix or _env("ATR_KINESIS_PREFIX")

    def put_records(self, stream: str, records) -> int:
        records, n = list(records), 0
        for b in _batches(records, 500):
            r = self.client.put_records(StreamName=f"{self.prefix}-{stream}", Records=[
                {"Data": json.dumps(x).encode(), "PartitionKey": str(x.get("cell", 0))} for x in b])
            if r.get("FailedRecordCount"):
                raise RuntimeError(f"Kinesis rejected {r['FailedRecordCount']} records on {stream}")
            n += len(b)
        return n

    def read(self, stream: str, after: int = 0) -> List[dict]:
        name, out = f"{self.prefix}-{stream}", []
        for shard in self.client.list_shards(StreamName=name)["Shards"]:
            it = self.client.get_shard_iterator(StreamName=name, ShardId=shard["ShardId"],
                                                ShardIteratorType="TRIM_HORIZON")["ShardIterator"]
            while it:
                r = self.client.get_records(ShardIterator=it, Limit=1000)
                out += [json.loads(x["Data"]) for x in r["Records"]]
                it = r.get("NextShardIterator") if r["Records"] else None
        return out[after:]


class S3LogLake:
    def __init__(self, bucket: Optional[str] = None, client=None, region: Optional[str] = None):
        if client is None:
            import boto3
            client = boto3.client("s3", region_name=region)
        self.client, self.bucket = client, bucket or _env("ATR_LAKE_BUCKET")

    def put(self, key: str, text: str) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=text.encode("utf-8"))
        return f"s3://{self.bucket}/{key}"

    def get(self, key: str) -> str:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read().decode("utf-8")


def _sql_literal(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def _typed(v: Optional[str]):
    if v is None:
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


class AthenaFeatureStore:
    """ingest.FeatureStore on Glue tables (JSON lines under s3://<lake>/features/<table>/campaign=<id>/) queried with
    Athena. The Glue tables are created by the CDK stack. `?` placeholders become Athena execution parameters."""

    def __init__(self, database: Optional[str] = None, output: Optional[str] = None, bucket: Optional[str] = None,
                 athena=None, s3=None, region: Optional[str] = None, poll: float = 1.0):
        import boto3
        self.athena = athena or boto3.client("athena", region_name=region)
        self.s3 = s3 or boto3.client("s3", region_name=region)
        self.database, self.output = database or _env("ATR_GLUE_DATABASE"), output or _env("ATR_ATHENA_OUTPUT")
        self.bucket, self.poll = bucket or _env("ATR_LAKE_BUCKET"), poll

    def _put(self, table: str, rows):
        rows = list(rows)
        if not rows:
            return
        camp = rows[0]["campaign"]
        body = "".join(json.dumps({k: v for k, v in r.items() if k != "fields"}) + "\n" for r in rows)
        self.s3.put_object(Bucket=self.bucket, Key=f"features/{table}/campaign={camp}/part-0.json", Body=body.encode())

    def load_kpi(self, rows): self._put("kpi_window", rows)
    def load_alarms(self, rows): self._put("alarm", rows)
    def load_events(self, rows): self._put("iface_event", rows)

    def drop_campaign(self, campaign: str):
        for t in ("kpi_window", "alarm", "iface_event"):
            self.s3.delete_object(Bucket=self.bucket, Key=f"features/{t}/campaign={campaign}/part-0.json")

    def query(self, sql: str, params: tuple = ()) -> List[dict]:
        kw = {"QueryString": sql, "QueryExecutionContext": {"Database": self.database},
              "ResultConfiguration": {"OutputLocation": self.output}}
        if params:
            kw["ExecutionParameters"] = [_sql_literal(p) for p in params]
        qid = self.athena.start_query_execution(**kw)["QueryExecutionId"]
        while True:
            st = self.athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
            if st["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(self.poll)
        if st["State"] != "SUCCEEDED":
            raise RuntimeError(f"Athena query {st['State']}: {st.get('StateChangeReason', '')}")
        rows, cols, token = [], None, None
        while True:
            r = self.athena.get_query_results(QueryExecutionId=qid, **({"NextToken": token} if token else {}))
            data = r["ResultSet"]["Rows"]
            if cols is None:
                cols = [c.get("VarCharValue") for c in data[0]["Data"]]
                data = data[1:]
            rows += [{c: _typed(d.get("VarCharValue")) for c, d in zip(cols, row["Data"])} for row in data]
            token = r.get("NextToken")
            if not token:
                return rows


def _signed_request(method: str, url: str, service: str, region: str, body: Optional[bytes] = None,
                    headers: Optional[dict] = None, session=None, opener=None) -> dict:
    """SigV4-signed HTTPS request (OpenSearch Serverless 'aoss', Neptune 'neptune-db'). `opener` is injectable for tests."""
    import urllib.request
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    headers = dict(headers or {})
    req = AWSRequest(method=method, url=url, data=body, headers=headers)
    if session is not False:
        import boto3
        creds = (session or boto3.Session()).get_credentials().get_frozen_credentials()
        SigV4Auth(creds, service, region).add_auth(req)
    r = urllib.request.Request(url, data=body, method=method, headers=dict(req.headers))
    with (opener or urllib.request.urlopen)(r) as resp:
        return json.loads(resp.read().decode() or "{}")


class OpenSearchLogSearch:
    """ingest.LogSearch on OpenSearch Serverless: text plus a Bedrock Titan embedding per document; search combines a
    k-NN query on the embedding with a text match, filtered by campaign/cell."""

    def __init__(self, endpoint: Optional[str] = None, index: str = "atr-logs", region: Optional[str] = None,
                 request=None, embed=None):
        self.endpoint = (endpoint or _env("ATR_OPENSEARCH_ENDPOINT")).rstrip("/")
        self.index_name, self.region = index, region or os.environ.get("ATR_REGION") or os.environ.get("AWS_REGION") or "us-east-2"
        self.request = request or (lambda m, u, b=None: _signed_request(m, u, "aoss", self.region, b,
                                                                        {"Content-Type": "application/json"}))
        self.embed = embed or self._titan
        self.docs = []

    def _titan(self, text: str) -> List[float]:
        import boto3
        r = boto3.client("bedrock-runtime", region_name=self.region).invoke_model(
            modelId=os.environ.get("ATR_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"), body=json.dumps({"inputText": text}))
        return json.loads(r["body"].read())["embedding"]

    def ensure_index(self, dim: int = 1024):
        mapping = {"settings": {"index.knn": True}, "mappings": {"properties": {
            "text": {"type": "text"}, "campaign": {"type": "keyword"}, "cell": {"type": "integer"},
            "kind": {"type": "keyword"}, "vec": {"type": "knn_vector", "dimension": dim}}}}
        try:
            self.request("PUT", f"{self.endpoint}/{self.index_name}", json.dumps(mapping).encode())
        except Exception as e:  # noqa: BLE001 - already exists
            if "resource_already_exists" not in str(e):
                raise

    def index(self, docs):
        docs = list(docs)
        if not getattr(self, "_index_ready", False):   # first write: make sure the k-NN mapping exists
            self.ensure_index()
            self._index_ready = True
        lines = []
        for d in docs:
            lines.append(json.dumps({"index": {"_index": self.index_name}}))
            lines.append(json.dumps({**{k: v for k, v in d.items() if k != "id"}, "doc_id": d["id"], "vec": self.embed(d["text"])}))
        for b in _batches(lines, 1000):
            self.request("POST", f"{self.endpoint}/_bulk", ("\n".join(b) + "\n").encode())
        self.docs.extend(docs)

    def search(self, query: str, k: int = 5, where: Optional[dict] = None) -> List[dict]:
        filt = [{"term": {key: v}} for key, v in (where or {}).items()]
        body = {"size": k, "query": {"bool": {"filter": filt, "should": [
            {"knn": {"vec": {"vector": self.embed(query), "k": k}}}, {"match": {"text": query}}]}}}
        r = self.request("POST", f"{self.endpoint}/{self.index_name}/_search", json.dumps(body).encode())
        return [{**h["_source"], "id": h["_source"].get("doc_id"), "score": round(float(h["_score"]), 3)}
                for h in r.get("hits", {}).get("hits", [])]


# --- Amazon Neptune: the finding graph -----------------------------------------------------------------------------
class NeptuneGraph:
    """graph.FindingGraph on Neptune, through the openCypher HTTPS endpoint (IAM auth, SigV4 'neptune-db')."""

    def __init__(self, endpoint: Optional[str] = None, region: Optional[str] = None, request=None):
        self.endpoint = (endpoint or _env("ATR_NEPTUNE_ENDPOINT")).rstrip("/")
        if not self.endpoint.startswith("https://"):
            self.endpoint = f"https://{self.endpoint}:8182"
        self.region = region or os.environ.get("ATR_REGION") or os.environ.get("AWS_REGION") or "us-east-2"
        self.request = request or (lambda body: _signed_request(
            "POST", f"{self.endpoint}/openCypher", "neptune-db", self.region, body,
            {"Content-Type": "application/x-www-form-urlencoded"}))

    def cypher(self, query: str, params: Optional[dict] = None) -> List[dict]:
        from urllib.parse import urlencode
        body = urlencode({"query": query, "parameters": json.dumps(params or {})}).encode()
        return self.request(body).get("results", [])

    def add_finding(self, card: dict) -> dict:
        from .graph import cypher_for, finding_facts
        for q, p in cypher_for(finding_facts(card)):
            self.cypher(q, p)
        return self.summary()

    def test_strategy(self, kind: str) -> dict:
        from .graph import STRATEGY_QUERY, strategy_from_rows
        return strategy_from_rows(kind, self.cypher(STRATEGY_QUERY, {"kind": kind}))

    def summary(self) -> dict:
        return {r["label"]: r["n"] for r in self.cypher("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n")}


# --- Step Functions: the experiment protocol -------------------------------------------------------------------------
class StepFunctionsProtocol:
    def __init__(self, arn: Optional[str] = None, client=None, region: Optional[str] = None, poll: float = 5.0):
        if client is None:
            import boto3
            client = boto3.client("stepfunctions", region_name=region)
        self.client, self.arn, self.poll = client, arn or _env("ATR_STATE_MACHINE_ARN"), poll

    def start(self, payload: dict) -> str:
        return self.client.start_execution(stateMachineArn=self.arn, name=payload["run_id"],
                                           input=json.dumps(payload))["executionArn"]

    def wait(self, execution_arn: str, timeout: float = 3600) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            d = self.client.describe_execution(executionArn=execution_arn)
            if d["status"] != "RUNNING":
                return {"status": d["status"], "output": json.loads(d.get("output") or "{}"), "error": d.get("error"),
                        "cause": d.get("cause")}
            time.sleep(self.poll)
        raise TimeoutError(execution_arn)

    def history(self, execution_arn: str) -> List[dict]:
        events = self.client.get_execution_history(executionArn=execution_arn, maxResults=1000)["events"]
        return [{"type": e["type"], "state": (e.get("stateEnteredEventDetails") or {}).get("name"), "ts": str(e["timestamp"])}
                for e in events if e["type"].endswith(("StateEntered", "Failed", "Succeeded"))]


class S3Context:
    """Per-run protocol context for the task Lambda (Step Functions passes only small JSON between states)."""

    def __init__(self, bucket: Optional[str] = None, client=None, region: Optional[str] = None):
        if client is None:
            import boto3
            client = boto3.client("s3", region_name=region)
        self.client, self.bucket = client, bucket or _env("ATR_LAKE_BUCKET")

    def load(self, run_id: str) -> dict:
        try:
            return json.loads(self.client.get_object(Bucket=self.bucket, Key=f"protocol/{run_id}/ctx.json")["Body"].read())
        except self.client.exceptions.NoSuchKey:
            return {}

    def save(self, run_id: str, ctx: dict):
        self.client.put_object(Bucket=self.bucket, Key=f"protocol/{run_id}/ctx.json",
                               Body=json.dumps(ctx, default=str).encode())


# --- Bedrock Custom Model Import: serve the research-tuned NLM ------------------------------------------------------
class BedrockModelImport:
    """Import merged fine-tuned weights (Hugging Face format in S3) as a Bedrock custom model, then invoke it."""

    def __init__(self, bedrock=None, runtime=None, region: Optional[str] = None):
        import boto3
        self.bedrock = bedrock or boto3.client("bedrock", region_name=region)
        self.runtime = runtime or boto3.client("bedrock-runtime", region_name=region)

    def start_import(self, job_name: str, model_name: str, s3_uri: str, role_arn: Optional[str] = None) -> str:
        return self.bedrock.create_model_import_job(
            jobName=job_name, importedModelName=model_name, roleArn=role_arn or _env("ATR_MODEL_IMPORT_ROLE"),
            modelDataSource={"s3DataSource": {"s3Uri": s3_uri}})["jobArn"]

    def status(self, job_arn: str) -> dict:
        r = self.bedrock.get_model_import_job(jobIdentifier=job_arn)
        return {"status": r["status"], "model_arn": r.get("importedModelArn"), "failure": r.get("failureMessage")}

    def invoke(self, model_arn: str, prompt: str, max_tokens: int = 200) -> str:
        r = self.runtime.invoke_model(modelId=model_arn, body=json.dumps(
            {"prompt": prompt, "max_gen_len": max_tokens, "temperature": 0.0}))
        body = json.loads(r["body"].read())
        return body.get("generation") or body.get("outputs", [{}])[0].get("text", "") or json.dumps(body)
