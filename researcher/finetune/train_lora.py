"""LoRA supervised fine-tuning of a causal language model on the Telecom Researcher dataset (SageMaker script mode).

The run that produced our reported result uses finetune/train_eval_lora.py instead (base and tuned scored in one job on
the frozen test set; launched by tools/nlm_tuning.py). This TRL-based variant is kept for longer training runs.

Inputs (SageMaker sets these): channels `train` and `validation` holding train.jsonl / val.jsonl from
`python -m atr.cli export-dataset`; output goes to SM_MODEL_DIR (LoRA adapter + tokenizer).
"""
import argparse
import os

from data_utils import load_jsonl, to_chat_examples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="Hugging Face id or local path of the base model")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj",
                   help="comma-separated attention projections to adapt; names differ by architecture")
    p.add_argument("--tasks", default="suspect_ranking,claim_triage")
    return p.parse_args()


def main():
    args = parse_args()
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_dir = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    tasks = tuple(args.tasks.split(","))

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train = Dataset.from_list(to_chat_examples(load_jsonl(os.path.join(train_dir, "train.jsonl")), tasks))
    val_path = os.path.join(val_dir, "val.jsonl")
    val = Dataset.from_list(to_chat_examples(load_jsonl(val_path), tasks)) if os.path.exists(val_path) else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype="auto")

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             target_modules=args.target_modules.split(","), task_type="CAUSAL_LM")
    config = SFTConfig(output_dir=model_dir, num_train_epochs=args.epochs, learning_rate=args.learning_rate,
                       per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
                       max_length=args.max_length, logging_steps=10, save_strategy="no",
                       eval_strategy="epoch" if val is not None else "no", bf16=True,
                       assistant_only_loss=False)   # set True if your chat template marks assistant turns
    trainer = SFTTrainer(model=model, args=config, train_dataset=train, eval_dataset=val,
                         processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
