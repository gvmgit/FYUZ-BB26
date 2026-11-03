"""Offline model for Strands agents (local mode).

`PlaybookModel` implements the Strands `Model` interface, so the agent loop, tool calling, tool results and
message history are all real Strands. What it replaces is only the LLM's choice of the next tool: that choice is a
deterministic script. In AWS mode the same agents get `BedrockModel` (Claude) instead and decide for themselves.

A script is a generator: it yields (tool_name, args) and receives the tool result text, and its return value is the
agent's final answer. The model rebuilds script state each turn by replaying the tool results found in the
message history, so it needs no hidden state.
"""
import json
import re
import uuid
from typing import AsyncIterable, Callable, Generator

from strands.models import Model

Script = Callable[[str], Generator]


def history_of(messages):
    """Prompt text of the latest user turn plus [(tool, args, result_text)] executed since."""
    last = max(i for i, m in enumerate(messages)
               if m["role"] == "user" and any("text" in b for b in m["content"]))
    prompt = " ".join(b["text"] for b in messages[last]["content"] if "text" in b)
    pending, calls = {}, []
    for m in messages[last + 1:]:
        for b in m["content"]:
            if "toolUse" in b:
                pending[b["toolUse"]["toolUseId"]] = (b["toolUse"]["name"], b["toolUse"]["input"])
            if "toolResult" in b:
                tr = b["toolResult"]
                name, args = pending[tr["toolUseId"]]
                calls.append((name, args, "".join(c.get("text", "") for c in tr["content"] if "text" in c)))
    return prompt, calls


class PlaybookModel(Model):
    def __init__(self, script: Script, name: str = "playbook"):
        self.script, self.name = script, name
        self._config = {"model_id": f"offline-playbook:{name}"}

    def update_config(self, **model_config) -> None:
        self._config.update(model_config)

    def get_config(self):
        return self._config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError("PlaybookModel does not produce structured output")
        yield  # pragma: no cover  (makes this an async generator)

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs) -> AsyncIterable[dict]:
        prompt, calls = history_of(messages)
        gen = self.script(prompt)
        action, final = None, None
        try:
            action = next(gen)
            for _name, _args, result in calls:
                action = gen.send(result)
        except StopIteration as stop:
            action, final = None, stop.value
        yield {"messageStart": {"role": "assistant"}}
        if action is not None:
            name, args = action
            yield {"contentBlockStart": {"contentBlockIndex": 0,
                                         "start": {"toolUse": {"name": name, "toolUseId": uuid.uuid4().hex[:12]}}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": json.dumps(args)}}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": final or ""}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                            "metrics": {"latencyMs": 0}}}


# --- the four playbooks ---------------------------------------------------------------------------------------

def scientist_script(prompt: str):
    yield ("observe_symptom", {})
    claims = yield ("propose_claims", {})
    return claims


def _single(tool_by_verb: dict, pattern: str, arg: str = "claim_id"):
    def script(prompt: str):
        m = re.search(pattern, prompt)
        if not m:
            return "could not parse the request"
        verb, target = m.group(1), m.group(2)
        result = yield (tool_by_verb[verb], {arg: target})
        return result
    return script


experimenter_script = _single({"test": "test_claim", "control": "run_control", "null": "run_null",
                               "holdout": "run_holdout", "replicate": "run_replication"},
                              r"\b(test|control|null|holdout|replicate)\s+(\S+)")
critic_script = _single({"review": "review_claim"}, r"\b(review)\s+(\S+)")


# --- orchestration playbooks ---------------------------------------------------------------------------------------

def _json(text: str, who: str) -> dict:
    """Parse a specialist's reply. A tool error comes back as plain text; fail loudly with it instead of carrying on."""
    try:
        return json.loads(text)
    except ValueError:
        raise RuntimeError(f"{who} returned an error instead of a result: {text[:400]}") from None


def _step(reply: str, who: str) -> dict:
    return _json(reply, who)


# --- exploration playbooks (no planted fault: look for a change worth making) ---------------------------------------

def explore_scientist_script(prompt: str):
    result = yield ("scan_network", {})
    return result


explore_experimenter_script = _single({"test": "test_insight", "replicate": "replicate_insight",
                                       "holdout": "holdout_insight"}, r"\b(test|replicate|holdout)\s+(\S+)",
                                      arg="hypothesis_id")
explore_critic_script = _single({"review": "review_insight"}, r"\b(review)\s+(\S+)", arg="hypothesis_id")


def explore_orchestrator_script(prompt: str):
    plan = _json((yield ("ask_scientist", {"request": "scan the network for changes worth making"})), "scientist")
    for hid in plan["hypothesis_ids"]:
        _step((yield ("ask_experimenter", {"task": f"test {hid}"})), "experimenter")
        for _ in range(4):
            verdict = _json((yield ("ask_critic", {"task": f"review {hid}"})), "critic")
            if verdict["status"] == "needs_replication":
                _step((yield ("ask_experimenter", {"task": f"replicate {hid}"})), "experimenter")
            elif verdict["status"] == "needs_holdout":
                _step((yield ("ask_experimenter", {"task": f"holdout {hid}"})), "experimenter")
            else:
                break
    summary = yield ("publish_insights", {})
    return summary


def orchestrator_script(prompt: str):
    plan = _json((yield ("ask_scientist", {"request": "observe the symptom and propose testable claims"})), "scientist")
    for claim_id in plan["claim_ids"]:
        _step((yield ("ask_experimenter", {"task": f"test {claim_id}"})), "experimenter")
        for _ in range(6):
            verdict = _json((yield ("ask_critic", {"task": f"review {claim_id}"})), "critic")
            if verdict["status"] == "needs_control":
                _step((yield ("ask_experimenter", {"task": f"control {claim_id}"})), "experimenter")
            elif verdict["status"] == "needs_null":
                _step((yield ("ask_experimenter", {"task": f"null {claim_id}"})), "experimenter")
            elif verdict["status"] == "needs_replication":
                _step((yield ("ask_experimenter", {"task": f"replicate {claim_id}"})), "experimenter")
            elif verdict["status"] == "needs_holdout":
                _step((yield ("ask_experimenter", {"task": f"holdout {claim_id}"})), "experimenter")
            else:
                break
    summary = yield ("publish_findings", {})
    return summary
