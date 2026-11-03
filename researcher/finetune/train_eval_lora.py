"""LoRA fine-tune of the OTel network language model, with base and tuned scored on the same frozen test set
(SageMaker script mode). One job, so both models see the same items, the same prompt and the same decoding.

Channels (SageMaker mounts them under /opt/ml/input/data/):
    model   the base model as Hugging Face files (OTel-LLM-1.7B-IT, copied from the event bucket to the team bucket)
    train   train.jsonl from `python -m atr.cli export-dataset` (suspect_ranking and claim_triage records)
    test    the frozen test set (evidence/frozen/nlm_test_v1.jsonl); only its suspect_ranking records are scored

Output (SM_MODEL_DIR, packed by SageMaker into model.tar.gz): the LoRA adapter, predictions.jsonl with each test item's
raw base and tuned answers, and metrics.json. The reported result is graded outside the job, from predictions.jsonl, by
the rule-based grader (atr/grader.py grade_nlm_tuning); the job's own numbers are a progress check only.
"""
import argparse
import json
import math
import os
import random
import re
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--tasks", default="suspect_ranking,claim_triage")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_file(folder, suffix):
    return next(os.path.join(folder, n) for n in sorted(os.listdir(folder)) if n.endswith(suffix))


def parse_json_answer(text):
    """Same rule as atr/finetune_eval.parse_json_answer: the first JSON object in the reply, {} if none."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except ValueError:
        return {}


def main():
    args = parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = "/opt/ml/input/data"
    out = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(f"{data}/model")
    model = AutoModelForCausalLM.from_pretrained(f"{data}/model", torch_dtype=torch.bfloat16).cuda()
    stop = [t for t in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if t is not None]

    def prompt_text(messages):
        return tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    tasks = set(args.tasks.split(","))
    train = [r for r in load_jsonl(first_file(f"{data}/train", ".jsonl")) if r["task"] in tasks]
    test = [r for r in load_jsonl(first_file(f"{data}/test", ".jsonl")) if r["task"] == "suspect_ranking"]
    print(f"train records {len(train)}  test items {len(test)}", flush=True)

    @torch.no_grad()
    def answer_all(m, label):
        m.eval()
        answers, t0 = [], time.time()
        for r in test:
            ids = tok(prompt_text(r["messages"]), return_tensors="pt").input_ids.cuda()
            gen = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens,
                             do_sample=False, eos_token_id=stop, pad_token_id=tok.pad_token_id or stop[0])
            answers.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip())
        joint = sum(parse_json_answer(a).get("kind") == r["meta"]["truth"]["kind"] and
                    parse_json_answer(a).get("cell") == r["meta"]["truth"]["cell"] for a, r in zip(answers, test))
        print(f"[{label}] joint {joint}/{len(test)} in {time.time() - t0:.0f}s", flush=True)
        return answers

    base_answers = answer_all(model, "base")

    # training examples: the same prompt as at test time, loss on the answer only
    examples = []
    for r in train:
        p = tok(prompt_text(r["messages"]), add_special_tokens=False).input_ids
        a = tok(r["messages"][2]["content"] + "<|im_end|>\n", add_special_tokens=False).input_ids
        if len(p) + len(a) <= args.max_length:
            examples.append((p, a))
    print(f"training examples {len(examples)} (dropped {len(train) - len(examples)} over {args.max_length} tokens)", flush=True)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=args.target_modules.split(","), task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.0)
    steps_total = math.ceil(len(examples) * args.epochs / args.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / max(1, steps_total)))
    model.train()
    seen, step, t0, running = 0, 0, time.time(), 0.0
    order = []
    while len(order) < len(examples) * args.epochs:
        batch = list(range(len(examples)))
        random.shuffle(batch)
        order += batch
    order = order[:int(len(examples) * args.epochs)]
    for n, i in enumerate(order, 1):
        p, a = examples[i]
        ids = torch.tensor([p + a]).cuda()
        labels = torch.tensor([[-100] * len(p) + a]).cuda()
        loss = model(input_ids=ids, labels=labels).loss / args.grad_accum
        loss.backward()
        running += loss.item()
        if n % args.grad_accum == 0 or n == len(order):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0 or n == len(order):
                print(f"step {step}/{steps_total}  loss {running / 10 if step % 10 == 0 else running:.4f}  {time.time() - t0:.0f}s", flush=True)
                running = 0.0

    model.gradient_checkpointing_disable()
    tuned_answers = answer_all(model, "tuned")

    model.save_pretrained(f"{out}/adapter")
    tok.save_pretrained(f"{out}/adapter")
    with open(f"{out}/predictions.jsonl", "w", encoding="utf-8") as f:
        for r, b, t in zip(test, base_answers, tuned_answers):
            f.write(json.dumps({"case_id": r["meta"].get("case_id"), "truth": r["meta"]["truth"],
                                "base_answer": b, "tuned_answer": t}) + "\n")
    json.dump({"train_examples": len(examples), "test_items": len(test), "optimizer_steps": step,
               "hyperparameters": vars(args)}, open(f"{out}/metrics.json", "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
