"""Score a language model against the rule-based system on held-out cases.

The yardstick a fine-tuned model has to beat. Two tasks (see finetune_data.py):

  suspect_ranking  - from diagnostics alone (0 experiments), name the fault kind and cell.
                     Baselines: majority class; the rule-based top suspect (cell only); and the full rule-based pipeline
                     (the researcher's own conclusion after ~70-95 experiments), stored per case so nothing is re-run.
  claim_triage     - for each claim, will it survive the Critic? Metric: mean reciprocal rank of the confirmed claim
                     versus the order the Scientist currently proposes them in (so a good model could cut experiments).

A predictor is any callable `record -> dict` (ranking) or `record -> float` (triage score). `EndpointPredictor` wraps a
SageMaker endpoint hosting the tuned model.
"""
import collections
import json
import re
from typing import Callable, Dict, List, Optional


def parse_json_answer(text: str) -> dict:
    """Pull the first JSON object out of a model's reply; {} if there is none."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except ValueError:
        return {}


class EndpointPredictor:
    """SageMaker endpoint hosting the fine-tuned model. The request/response schema depends on the serving container
    (this assumes a text-generation container taking {"inputs": ...} and returning generated_text); override
    `format_prompt` / `extract_text` to match yours. Not run against a real endpoint from this machine."""

    def __init__(self, endpoint: str, client=None, region: Optional[str] = None,
                 format_prompt: Optional[Callable[[dict], str]] = None):
        if client is None:
            import boto3
            client = boto3.client("sagemaker-runtime", region_name=region)
        self.client, self.endpoint = client, endpoint
        self.format_prompt = format_prompt or (lambda rec: rec["messages"][0]["content"] + "\n\n"
                                               + rec["messages"][1]["content"] + "\n\nAnswer:")

    def extract_text(self, body) -> str:
        return body[0]["generated_text"] if isinstance(body, list) else str(body)

    def raw(self, record: dict) -> str:
        resp = self.client.invoke_endpoint(EndpointName=self.endpoint, ContentType="application/json",
                                           Body=json.dumps({"inputs": self.format_prompt(record),
                                                            "parameters": {"max_new_tokens": 80, "temperature": 0.0}}))
        return self.extract_text(json.loads(resp["Body"].read()))

    def __call__(self, record: dict):
        answer = parse_json_answer(self.raw(record))
        if record["task"] == "claim_triage":
            return 1.0 if answer.get("survives_critic") else 0.0
        return answer


# --- baselines ------------------------------------------------------------------------------------------------------
def majority_predictor(train_records: List[dict]) -> Callable[[dict], dict]:
    labels = [json.dumps(r["meta"]["truth"], sort_keys=True) for r in train_records if r["task"] == "suspect_ranking"]
    top = json.loads(collections.Counter(labels).most_common(1)[0][0])
    return lambda rec: top


def rule_pipeline(rec: dict) -> dict:
    """The researcher's own conclusion after running its experiments."""
    return rec["meta"]["system_outcome"]


def rule_top_suspect(rec: dict) -> dict:
    """Cell with the highest rule-based suspicion score; it says nothing about the kind of fault."""
    return {"kind": None, "cell": rec["meta"]["rule_top_suspect_cell"]}


# --- metrics --------------------------------------------------------------------------------------------------------
def evaluate_ranking(records: List[dict], predict: Callable[[dict], dict]) -> dict:
    rows = [r for r in records if r["task"] == "suspect_ranking"]
    n = len(rows)
    kind_ok = cell_ok = joint = 0
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        truth, pred = r["meta"]["truth"], predict(r) or {}
        k = pred.get("kind") == truth["kind"]
        c = pred.get("cell") == truth["cell"]
        kind_ok += k
        cell_ok += c
        joint += k and c
        per[truth["kind"]][1] += 1
        per[truth["kind"]][0] += k and c
    return {"n": n, "kind_acc": kind_ok / n if n else None, "cell_acc": cell_ok / n if n else None,
            "joint_acc": joint / n if n else None,
            "by_class": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per.items())}}


def evaluate_triage(records: List[dict], score: Callable[[dict], float]) -> dict:
    """Mean reciprocal rank of the confirmed claim, over investigations that confirmed one."""
    cases = collections.defaultdict(list)
    for r in records:
        if r["task"] == "claim_triage":
            cases[r["meta"]["case_id"]].append(r)
    rr, top1, used = [], 0, 0
    for recs in cases.values():
        if not any(r["meta"]["label"]["survives_critic"] for r in recs):
            continue
        used += 1
        ranked = sorted(recs, key=lambda r: -score(r))
        pos = next(i for i, r in enumerate(ranked, 1) if r["meta"]["label"]["survives_critic"])
        rr.append(1 / pos)
        top1 += pos == 1
    return {"cases_with_confirmed_claim": used, "mrr": sum(rr) / used if used else None,
            "top1": top1 / used if used else None}


def scientist_order_score(rec: dict) -> float:
    return -rec["meta"]["claim_order"]


def baseline_report(train: List[dict], test: List[dict], extra: Optional[Dict[str, Callable]] = None) -> dict:
    """Everything a tuned model is compared with, on the test split."""
    out = {"ranking": {"majority class": evaluate_ranking(test, majority_predictor(train)),
                       "rule top suspect (cell only, 0 experiments)": evaluate_ranking(test, rule_top_suspect),
                       "full rule-based pipeline (runs experiments)": evaluate_ranking(test, rule_pipeline)},
           "triage": {"current Scientist order": evaluate_triage(test, scientist_order_score)}}
    for name, predictor in (extra or {}).items():
        out["ranking"][name] = evaluate_ranking(test, predictor)
        out["triage"][name] = evaluate_triage(test, predictor)
    n_exp = [r["meta"]["experiments"] for r in test if r["task"] == "suspect_ranking"]
    out["pipeline_experiments_per_case"] = sum(n_exp) / len(n_exp) if n_exp else None
    return out


def format_report(rep: dict) -> str:
    lines = ["suspect_ranking on the test split (higher is better)",
             f"  {'predictor':<48}{'kind':>7}{'cell':>7}{'both':>7}"]
    for name, m in rep["ranking"].items():
        f = lambda x: "  n/a " if x is None else f"{x:6.0%} "  # noqa: E731
        lines.append(f"  {name:<48}{f(m['kind_acc'])}{f(m['cell_acc'])}{f(m['joint_acc'])}")
    lines.append("\nclaim_triage: rank of the confirmed claim (MRR / top-1, higher is better)")
    for name, m in rep["triage"].items():
        lines.append(f"  {name:<48}{m['mrr']:.2f} / {m['top1']:.0%}  ({m['cases_with_confirmed_claim']} cases)")
    lines.append(f"\nthe rule-based pipeline needs {rep['pipeline_experiments_per_case']:.0f} experiments per case on average; "
                 "a model reading diagnostics needs 0")
    return "\n".join(lines)
