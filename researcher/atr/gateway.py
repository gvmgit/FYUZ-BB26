"""Tool gateway: local stand-in for AgentCore Gateway.

Agents never call MAVERIC directly. They call named tools with JSON arguments; the gateway enforces the mesh
policy, validates arguments, executes, and writes an audit record (access control + audit trail). On AWS the
same tool is registered on AgentCore Gateway (see aws_adapters.py) and the agents are unchanged.
"""
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .audit import AuditSink, FileAudit

RUN_EXPERIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "config": {
            "type": "object",
            "description": "Settings to change, one at a time, e.g. {'name': 'cell3 tilt 12', "
                           "'cells': {'3': {'tilt': 12}}}. Optional flags: render (heatmap), diagnostics.",
        },
        "seeds": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
    },
    "required": ["config", "seeds"],
}


LAB_SCHEMA = {"type": "object", "properties": {      # lab_execute: the emulated lab's test automation API
    "action": {"type": "string", "enum": ["reset", "set_traffic", "set_handover_rule", "set_parameter", "run_campaign", "replicate"]},
    "windows": {"type": "integer"}, "seed0": {"type": "integer"}, "campaign_id": {"type": "string"},
    "profile": {"type": "string"}, "fault": {"type": "object"}, "with_logs": {"type": "boolean"},
    "cell": {"type": "integer"}, "name": {"type": "string"}, "value": {}, "hyst": {"type": "number"}, "ttt": {"type": "integer"}},
    "required": ["action"]}


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable
    validator: Optional[Callable] = None


class ToolError(Exception):
    pass


class Gateway:
    def __init__(self, audit: Optional[AuditSink] = None, policy: Optional[Callable[[str, str], None]] = None):
        self._tools: Dict[str, Tool] = {}
        self.audit = audit or FileAudit()
        self.policy = policy   # e.g. AgentMesh.authorize; raises if the agent may not call the tool
        self.count = 0         # successful run_experiment calls (the demo's "experiments needed" figure)

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def specs(self):
        return [{"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                for t in self._tools.values()]

    def invoke(self, name: str, args: dict, agent: str):
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool '{name}'")
        t0, error, result = time.time(), None, None
        try:
            if self.policy:
                self.policy(agent, name)
            if tool.validator:
                tool.validator(args)
            result = tool.fn(**args)
            self.count += 1
            return result
        except Exception as e:  # noqa: BLE001 - audit then re-raise as ToolError
            error = str(e)
            raise ToolError(error) from e
        finally:
            self.audit.write({"kind": "tool_call", "agent": agent, "tool": name, "args": args,
                              "ok": error is None, "error": error, "run_id": (result or {}).get("run_id"),
                              "seconds": round(time.time() - t0, 3)})


def build_gateway(runner, audit: Optional[AuditSink] = None, policy=None) -> Gateway:
    gw = Gateway(audit=audit, policy=policy)
    gw.register(Tool(
        name="run_experiment",
        description="Run one MAVERIC twin experiment (change at most one setting) over several seeds and "
                    "return per-seed KPIs and the discrepancy score against the field log.",
        input_schema=RUN_EXPERIMENT_SCHEMA,
        fn=lambda config, seeds: runner.run(config, seeds),   # noqa: A006
        validator=lambda a: runner.validate(a["config"]),
    ))
    return gw
