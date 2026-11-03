"""Dependency-free helpers for the training script (kept apart so they can be tested without transformers/peft)."""
import json


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_chat_examples(records, tasks=("suspect_ranking", "claim_triage")):
    """Training examples: just the chat messages. `meta` (ground truth, the rule pipeline's answer) is never given to
    the model, so it cannot be learned by copying."""
    out = []
    for r in records:
        if r["task"] not in tasks:
            continue
        roles = [m["role"] for m in r["messages"]]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"unexpected message roles {roles} in {r['meta']['case_id']}")
        json.loads(r["messages"][2]["content"])      # the target must be valid JSON
        out.append({"messages": r["messages"]})
    return out
