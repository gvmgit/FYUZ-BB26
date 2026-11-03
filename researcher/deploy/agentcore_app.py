"""AgentCore Runtime entrypoint: runs a full investigation with the Strands agents in AWS mode.

Invoke with {"mode": "aws"} (default) or {} ; a queued SQS job such as {"random_seed": 3} is accepted and ignored here
(the field log is whatever `plant-fault` last uploaded). Returns a short summary; the finding cards are in DynamoDB.

    agentcore launch      (or build deploy/agentcore/Dockerfile and create the runtime with the container URI)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bedrock_agentcore.runtime import BedrockAgentCoreApp  # noqa: E402

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload):
    from atr import runtime
    from atr.strands_agents import run_strands_investigation
    payload = payload or {}
    rt = runtime.build(payload.get("mode", os.environ.get("ATR_MODE", "aws")), emit=lambda m: None,
                       run_id=payload.get("run_id"), trace_id=payload.get("trace_id"))
    try:
        out = run_strands_investigation(rt.researcher, rt.mesh, rt.model_for)
    except Exception as e:  # noqa: BLE001  (a failure keeps its record: the run id and the error are returned and logged)
        rt.mesh.checkpoint.record({"kind": "failed", "error": f"{type(e).__name__}: {e}"[:500]})
        return {"run_id": rt.mesh.run_id, "trace_id": rt.mesh.checkpoint.trace_id, "error": f"{type(e).__name__}: {e}"[:500]}
    finally:
        rt.close()
    head = out["headline"]
    return {"run_id": rt.mesh.run_id, "trace_id": rt.mesh.checkpoint.trace_id, "headline": head["title"] if head else None,
            "claim": {k: head["claim"].get(k) for k in ("kind", "cell", "estimate")} if head else None,
            "experiments": out["experiments"], "tokens": out["tokens"], "cards": [c["id"] for c in out["cards"]]}


if __name__ == "__main__":
    app.run()
