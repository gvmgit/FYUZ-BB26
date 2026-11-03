"""The research-tuned network language model: LoRA fine-tuning on validated findings, and its evaluation.

Training data, all produced by the researcher itself:
  suspect_ranking      blind-test investigations (finetune/data/train.jsonl): diagnostics -> the fault kind and cell
  finding_explanation  research-cycle findings (finetune/data/findings_corpus.jsonl): cross-modal observation -> finding;
                       only findings an engineer ACCEPTED are used, unless include_unreviewed (demo)

Locally the base is a small open model (default SmolLM2-135M-Instruct) so the full loop - train, evaluate, register, serve -
runs on a CPU. The same code is the SageMaker entry point for the OTel-LLM-8.3B-IT once its weights are available
(finetune/launch_sagemaker.py), and `export_merged` writes the Hugging Face folder Bedrock Custom Model Import takes.

Evaluation is on the held-out test split (cases the model never saw), against the same baselines as before: majority class,
the rule-based top suspect, and the full rule-based pipeline that runs ~90 experiments. The untuned base model is scored too,
so the effect of fine-tuning is measured, not assumed.
"""
import json
import time
from pathlib import Path
from typing import List, Optional

from . import config

REGISTRY = config.MODELS / "nlm_registry.json"
STATE = config.RUNS / "nlm_state.json"
HF_CACHE = config.MODELS / "hf"
MAX_PROMPT_TOKENS = 1024


def _load(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def build_dataset(data_dir: str = "finetune/data", include_unreviewed: bool = False, store=None) -> dict:
    from .artefacts import load_corpus
    d = config.ROOT / data_dir
    train = [r for r in _load(d / "train.jsonl") if r["task"] == "suspect_ranking"]
    val = [r for r in _load(d / "val.jsonl") if r["task"] == "suspect_ranking"]
    test = [r for r in _load(d / "test.jsonl") if r["task"] == "suspect_ranking"]
    corpus, used = load_corpus(), []
    if corpus:
        if store is None:
            from .serve import make_store
            store = make_store()
        for s in corpus:
            card = store.get(s["card_id"])
            ok = card is not None and (any(v["verdict"] == "accept" for v in card.get("engineer_verdicts", []))
                                       or (include_unreviewed and card["verdict"] == "confirmed"))
            if ok:
                used.append({"task": s["task"], "messages": s["messages"], "meta": {"case_id": s["card_id"]}})
    return {"train": train + used, "val": val, "test": test, "findings_used": len(used), "findings_in_corpus": len(corpus)}


def _hf():
    import os
    import truststore
    truststore.inject_into_ssl()          # corporate TLS inspection: trust the OS store
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return torch, AutoModelForCausalLM, AutoTokenizer


def _encode(tok, messages, with_answer: bool):
    prompt = tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"][-MAX_PROMPT_TOKENS:]
    if not with_answer:
        return p_ids, None
    a_ids = tok(messages[2]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    return p_ids + a_ids, [-100] * len(p_ids) + a_ids        # loss only on the answer


class Predictor:
    """record -> {"kind", "cell"} by greedy decoding; usable with finetune_eval.evaluate_ranking."""

    def __init__(self, model, tok, torch):
        self.model, self.tok, self.torch = model, tok, torch
        self.raw_answers = []

    def __call__(self, rec: dict) -> dict:
        from .finetune_eval import parse_json_answer
        ids, _ = _encode(self.tok, rec["messages"], with_answer=False)
        with self.torch.no_grad():
            out = self.model.generate(self.torch.tensor([ids]), max_new_tokens=24, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][len(ids):], skip_special_tokens=True)
        self.raw_answers.append(text)
        ans = parse_json_answer(text)
        try:
            return {"kind": ans.get("kind"), "cell": int(ans["cell"]) if "cell" in ans else None}
        except (TypeError, ValueError):
            return {"kind": ans.get("kind"), "cell": None}


def train(base: str, data: dict, epochs: int = 3, lr: float = 3e-4, out_dir: Optional[Path] = None, log=print) -> dict:
    torch, AutoModelForCausalLM, AutoTokenizer = _hf()
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    examples = [_encode(tok, r["messages"], with_answer=True) for r in data["train"]]
    steps, losses, t0 = 0, [], time.time()
    model.train()
    for ep in range(epochs):
        order = torch.randperm(len(examples)).tolist()
        ep_loss = []
        for i in order:
            ids, labels = examples[i]
            loss = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels])).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss.append(float(loss))
            steps += 1
        losses.append(round(sum(ep_loss) / len(ep_loss), 4))
        log(f"[nlm] epoch {ep + 1}/{epochs}: mean answer loss {losses[-1]}  ({time.time() - t0:.0f}s)")
    model.eval()
    out_dir = out_dir or config.MODELS / f"nlm-adapter-{time.strftime('%Y%m%d-%H%M%S')}"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {"model": model, "tok": tok, "torch": torch, "adapter_dir": str(out_dir), "trainable_params": trainable,
            "steps": steps, "epoch_loss": losses, "train_seconds": round(time.time() - t0, 1)}


def evaluate(predict, data: dict) -> dict:
    from . import finetune_eval as fe
    return fe.evaluate_ranking(data["test"], predict)


def train_and_evaluate(base: str = "HuggingFaceTB/SmolLM2-135M-Instruct", epochs: int = 3, include_unreviewed: bool = False,
                       data_dir: str = "finetune/data", log=print) -> dict:
    from . import finetune_eval as fe
    data = build_dataset(data_dir, include_unreviewed)
    log(f"[nlm] training examples: {len(data['train'])} ({data['findings_used']} from accepted research findings of "
        f"{data['findings_in_corpus']} in the corpus); held-out test cases: {len(data['test'])}")
    torch, AutoModelForCausalLM, AutoTokenizer = _hf()
    try:
        tok = AutoTokenizer.from_pretrained(base)
    except OSError as e:
        raise SystemExit(
            f"cannot load base model '{base}': {str(e).splitlines()[0]}\n"
            "This network blocks huggingface.co (HTTP 403). Get the model folder through an approved route (a machine "
            "or proxy your IT allows, or an internal model registry), copy it under models/, and run e.g.\n"
            "  python -m atr.cli train-nlm --base models/SmolLM2-135M-Instruct\n"
            "or run the same training on SageMaker (finetune/launch_sagemaker.py), which downloads from AWS.") from None
    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32).eval()
    before = evaluate(Predictor(base_model, tok, torch), data)
    log(f"[nlm] untuned base model on the test split: kind {before['kind_acc']:.0%}, cell {before['cell_acc']:.0%}, "
        f"both {before['joint_acc']:.0%}")
    del base_model
    t = train(base, data, epochs, log=log)
    pred = Predictor(t["model"], t["tok"], t["torch"])
    after = evaluate(pred, data)
    log(f"[nlm] fine-tuned model on the test split: kind {after['kind_acc']:.0%}, cell {after['cell_acc']:.0%}, "
        f"both {after['joint_acc']:.0%}")
    baselines = {k: v for k, v in fe.baseline_report(data["train"], data["test"])["ranking"].items()}
    entry = {"version": Path(t["adapter_dir"]).name, "base": base, "adapter_dir": t["adapter_dir"],
             "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_examples": len(data["train"]),
             "findings_used": data["findings_used"], "epochs": epochs, "trainable_params": t["trainable_params"],
             "epoch_loss": t["epoch_loss"], "train_seconds": t["train_seconds"],
             "test": {"untuned_base": before, "fine_tuned": after, "baselines": baselines},
             "sample_answers": pred.raw_answers[:3]}
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    reg.append(entry)
    REGISTRY.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    from .artefacts import load_corpus
    STATE.write_text(json.dumps({"trained_on": len(load_corpus()), "version": entry["version"]}), encoding="utf-8")
    return {k: entry[k] for k in ("version", "train_examples", "findings_used", "epoch_loss", "train_seconds", "test")}


def export_merged(adapter_dir: str, out_dir: Optional[str] = None) -> str:
    """Merge the LoRA adapter into the base weights and save a Hugging Face folder (safetensors + config + tokenizer):
    the input Bedrock Custom Model Import expects in S3 (aws_adapters.BedrockModelImport)."""
    torch, AutoModelForCausalLM, AutoTokenizer = _hf()
    from peft import PeftModel
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model_name_or_path"], torch_dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out = out_dir or str(Path(adapter_dir).with_name(Path(adapter_dir).name.replace("adapter", "merged")))
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out)
    return out


def load_latest():
    """The newest registered adapter as a Predictor (the researcher's local 'NLM second opinion')."""
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    if not reg:
        return None
    torch, AutoModelForCausalLM, AutoTokenizer = _hf()
    from peft import PeftModel
    e = reg[-1]
    tok = AutoTokenizer.from_pretrained(e["adapter_dir"])
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(e["base"], torch_dtype=torch.float32),
                                      e["adapter_dir"]).eval()
    return Predictor(model, tok, torch)
