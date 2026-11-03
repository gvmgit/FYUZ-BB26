"""Checks the logic of every AWS adapter and the toggle against in-memory fakes (no AWS account needed).

This proves the code around each AWS call - serialisation, policy, audit, parsing, config errors. It does NOT prove
the AWS calls themselves; that needs a real account (see the README's step-by-step deployment).

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_adapters.py
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atr import aws_adapters as aws, modes, stats  # noqa: E402
from atr.anf import AgentMesh, PolicyViolation  # noqa: E402
from atr.audit import FileAudit  # noqa: E402
from atr.gateway import ToolError  # noqa: E402
from atr.runtime import ConfigError, build  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


class FakeTable:
    def __init__(self):
        self.items = {}

    def put_item(self, Item):
        assert not any(isinstance(v, float) for v in _walk(Item)), "DynamoDB would reject a float"
        self.items[Item["id"]] = Item

    def get_item(self, Key):
        return {"Item": self.items[Key["id"]]} if Key["id"] in self.items else {}

    def scan(self, **kw):
        return {"Items": list(self.items.values())}


def _walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)
    else:
        yield o


# 1. DynamoDB store round trip with floats, ordering and get
t = FakeTable()
store = aws.DynamoDBStore(table=t)
store.put({"id": "a", "created": 1.5, "score": 3.14159, "nested": {"x": [1.25, 2]}})
store.put({"id": "b", "created": 2.5})
check("DynamoDBStore stores floats as Decimal and returns them as floats",
      store.get("a")["nested"]["x"] == [1.25, 2] and store.get("a")["score"] == 3.14159)
check("DynamoDBStore.list is newest first", [c["id"] for c in store.list()] == ["b", "a"])
check("DynamoDBStore.get of a missing id is None", store.get("zzz") is None)

# 2. DynamoDB audit
at = FakeTable()
aws.DynamoAudit(table=at).write({"kind": "tool_call", "seconds": 0.25})
check("DynamoAudit writes a unique, float-safe record", len(at.items) == 1)

# 3. SQS scheduler
sent = []
sched = aws.SqsScheduler("https://sqs/q", client=SimpleNamespace(
    send_message=lambda **kw: sent.append(kw) or {"MessageId": "m1"}))
mid = sched.submit({"random_seed": 3}, priority=1)
check("SqsScheduler enqueues JSON with a priority attribute",
      mid == "m1" and json.loads(sent[0]["MessageBody"]) == {"random_seed": 3}
      and sent[0]["MessageAttributes"]["priority"]["StringValue"] == "1")

# 4. Code Interpreter: run the exact program that would be sent, return it in the service's response shape
base, treat = [8.0, 8.2, 7.9, 8.1], [3.0, 3.1, 2.9, 3.2]


class FakeSession:
    def execute_code(self, code):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        return {"stream": [{"result": {"isError": out.returncode != 0,
                                       "content": [{"type": "text", "text": out.stdout}],
                                       "structuredContent": {"stdout": out.stdout, "stderr": out.stderr}}}]}


class FakeCM:
    def __enter__(self):
        return FakeSession()

    def __exit__(self, *a):
        pass


ci = aws.CodeInterpreterStats("us-west-2", session_factory=lambda: FakeCM())
remote, local = ci(base, treat), stats._paired_effect_impl(base, treat)
check("Code Interpreter program runs standalone and matches the local result",
      remote["mean_improvement"] == local["mean_improvement"] and remote["ci95"] == local["ci95"])
stats.set_executor(ci)
n0 = stats.calls["remote"]
stats.paired_effect(base, treat)
stats.set_executor(None)
check("stats.paired_effect routes through the executor when set", stats.calls["remote"] == n0 + 1)
try:
    aws.parse_interpreter_output({"stream": [{"result": {"isError": True, "structuredContent": {"stderr": "boom"}}}]})
    check("interpreter errors raise", False)
except ToolError:
    check("interpreter errors raise", True)

# 5. Remote gateway over (fake) MCP: tool-name prefix, policy, audit, counting
audit_path = Path(__file__).resolve().parents[1] / "runs" / "check_adapters_audit.jsonl"
audit_path.unlink(missing_ok=True)
mesh = AgentMesh(audit=FileAudit(audit_path))
calls = []


class FakeMCP:
    def list_tools_sync(self):
        return [SimpleNamespace(tool_name="twin___run_experiment")]

    def call_tool_sync(self, tool_use_id, name, arguments):
        calls.append((name, arguments))
        return {"status": "success", "content": [{"text": json.dumps({"run_id": "exp001", "mean": {"score": 3.0}})}]}


gw = aws.RemoteGateway("https://gw", None, mesh.audit, policy=mesh.authorize, mcp_client=FakeMCP())
r = gw.invoke("run_experiment", {"config": {"cells": {}}, "seeds": [1]}, agent="experimenter")
check("RemoteGateway resolves the prefixed tool name and parses the result",
      calls[0][0] == "twin___run_experiment" and r["run_id"] == "exp001" and gw.count == 1)
try:
    gw.invoke("run_experiment", {"config": {}, "seeds": [1]}, agent="critic")
    check("RemoteGateway enforces mesh policy before the call leaves", False)
except ToolError:
    check("RemoteGateway enforces mesh policy before the call leaves", len(calls) == 1)
check("RemoteGateway audits both calls",
      sum(1 for _ in open(audit_path, encoding="utf-8") if '"tool_call"' in _) == 2)

# 5b. Multimodal: the heatmap bytes reach a Strands agent as an image content block
from atr import playbook as pb  # noqa: E402
png = Path(__file__).resolve().parents[1] / "runs" / "check_heatmap.png"
png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
seen_blocks = []


def vision_script(prompt):
    return "the heatmap shows a weak lobe near cell 3"
    yield  # pragma: no cover


class SpyModel(pb.PlaybookModel):
    async def stream(self, messages, *a, **kw):
        seen_blocks.extend(messages[-1]["content"])
        async for ev in super().stream(messages, *a, **kw):
            yield ev


text = aws.HeatmapReader(lambda: SpyModel(vision_script, "vision")).describe(str(png), "zone Z7 is weak")
check("HeatmapReader sends PNG bytes as an image block with the KPI text",
      any("image" in b and b["image"]["source"]["bytes"].startswith(b"\x89PNG") for b in seen_blocks)
      and any("text" in b and "zone Z7" in b["text"] for b in seen_blocks) and "weak lobe" in text)
s3_reads = []
aws.HeatmapReader(None, s3=SimpleNamespace(get_object=lambda **kw: s3_reads.append(kw) or {"Body": io.BytesIO(b"x")}))._bytes("s3://b/heatmaps/e.png")
check("HeatmapReader fetches s3:// heatmaps", s3_reads == [{"Bucket": "b", "Key": "heatmaps/e.png"}])
png.unlink(missing_ok=True)

# 6. Network LM, plant/reveal, Kiro
nlm = aws.NetworkLM("ep", client=SimpleNamespace(invoke_endpoint=lambda **kw: {
    "Body": io.BytesIO(json.dumps([{"generated_text": "cell 3 is misconfigured"}]).encode())}))
check("NetworkLM parses a text-generation response", "cell 3" in nlm.second_opinion("symptom", {3: 2.0}))
os.environ.update(ATR_FIELD_BUCKET="b", ATR_TRUTH_TABLE="t")
puts, tt = [], FakeTable()
aws.plant_fault("a,b\n1,2\n", {"kind": "tilt", "cell": 3, "value": 16.0},
                s3=SimpleNamespace(put_object=lambda **kw: puts.append(kw)), table=tt)
check("plant_fault uploads the field log and stores the truth separately",
      puts[0]["Key"] == "field/current.csv" and aws.read_truth(table=tt)["cell"] == 3)
os.environ["ATR_KIRO_CMD"] = "echo kiro"
k = aws.kiro_generate("spec.md", "out", run=lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=cmd, stderr=""))
check("kiro_generate passes spec and output dir via environment", k["returncode"] == 0 and k["stdout"] == "echo kiro")

# 7. The toggle
for k_ in [k_ for k_ in os.environ if k_.startswith("ATR_") and k_ not in ()]:
    os.environ.pop(k_)
check("default mode is local with every service local",
      modes.resolve().mode == "local" and not any(modes.resolve().aws(s) for s in modes.SERVICES))
os.environ["ATR_STORE"] = "aws"
sv = modes.resolve("local")
check("a per-service override flips only that service", sv.aws("store") and not sv.aws("model"))
os.environ.pop("ATR_STORE")
check("aws mode flips every service and reports missing configuration",
      all(modes.resolve("aws").aws(s) for s in modes.SERVICES) and "model" in modes.resolve("aws").missing_env())
try:
    build("aws")
    check("building in aws mode without configuration fails loudly", False)
except ConfigError as e:
    check("building in aws mode without configuration fails loudly", "ATR_MODEL_ID" in str(e))
try:
    modes.resolve("hybrid")
    check("an invalid mode is rejected", False)
except ValueError:
    check("an invalid mode is rejected", True)

audit_path.unlink(missing_ok=True)
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
