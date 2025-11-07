"""
Train a yes/no relevance judge via GRPO (TRL) with QLoRA on Qwen2.5-1.5B-Instruct.

Preferred inputs (consistent with baselines):
- A splits directory with train.csv, val.csv, test.csv, each with columns: query_id,query,abstract,label (1 yes, 0 no)

Legacy input (still supported):
- A single CSV passed via --data; the script will create an internal train/val/test split.

Logging:
- TensorBoard scalars and CSV (via TRL logging + simple CSV writer)

Example (WSL/Linux):
python -m scripts.train_judge_grpo \
    --splits ./data/splits \
    --out ./runs/qwen15b \
    --steps 3000 \
    --batch-size 64 \
    --micro-batch 4 \
    --eval-interval 250
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# Avoid tokenizers parallelism + fork warning/deadlocks in DataLoader workers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainerCallback
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from transformers import set_seed

import sys


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_ROOT = _project_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.rl.prompts import build_prompt, parse_answer  # type: ignore
from scripts.rl.reward_fn import compute_reward  # type: ignore


@dataclass
class Example:
    query: str
    abstract: str
    label: int


def load_pairs_csv(path: str) -> List[Example]:
    out: List[Example] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("query") or "").strip()
            a = (row.get("abstract") or "").strip()
            lab = row.get("label")
            if not q or not a or lab is None:
                continue
            try:
                y = int(lab)
            except Exception:
                continue
            if y not in (0, 1):
                continue
            out.append(Example(q, a, y))
    return out


def split_dataset(examples: List[Example], val_size: int, test_size: int, seed: int) -> Tuple[List[Example], List[Example], List[Example]]:
    rnd = random.Random(seed)
    idx = list(range(len(examples)))
    rnd.shuffle(idx)
    val = idx[:val_size]
    test = idx[val_size : val_size + test_size]
    train = idx[val_size + test_size :]
    def take(ix):
        return [examples[i] for i in ix]
    return take(train), take(val), take(test)


def main():
    ap = argparse.ArgumentParser(description="PPO train relevance judge (yes/no)")
    ap.add_argument("--splits", default=None, help="Directory with train.csv, val.csv, test.csv")
    ap.add_argument("--data", default=None, help="(Legacy) Single CSV with query,abstract,label; will be split internally if --splits is not provided")
    ap.add_argument("--out", required=True, help="Output directory for checkpoints/logs")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64, help="Effective batch size")
    ap.add_argument("--micro-batch", type=int, default=4, help="Micro batch size (per device)")
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--val-size", type=int, default=500, help="(Legacy) Used only when --data is provided")
    ap.add_argument("--test-size", type=int, default=500, help="(Legacy) Used only when --data is provided")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-abstract-chars", type=int, default=2000)
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024, help="Max tokens from prompt fed to model (truncate)")
    ap.add_argument("--num-generations", type=int, default=2, help="Number of sampled generations per prompt (>=2 for GRPO)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-beta", type=float, default=0.05)
    ap.add_argument("--fmt-penalty", type=float, default=1.0, help="Penalty applied when format is violated (1.0 => zero reward for correct-but-bad format)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--optim", default="adamw_torch_fused", help="Optimizer for Trainer (e.g., adamw_torch, adamw_torch_fused)")
    ap.add_argument("--dataloader-workers", type=int, default=2)
    ap.add_argument("--log-dir", default=None, help="TensorBoard logging directory (defaults to <out>/tb)")
    ap.add_argument("--log-steps", type=int, default=20, help="How often to log scalars (steps)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    # Prefer maximum matmul precision for speed on Ampere+ GPUs
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # Load data: prefer explicit splits if provided, else fallback to legacy --data
    train: List[Example]
    val: List[Example]
    test: List[Example]
    if args.splits:
        def _load_split_csv(dir_path: str, name: str) -> List[Example]:
            path = os.path.join(dir_path, f"{name}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing split file: {path}")
            return load_pairs_csv(path)
        train = _load_split_csv(args.splits, "train")
        val = _load_split_csv(args.splits, "val")
        test = _load_split_csv(args.splits, "test")
        if not train or not val:
            raise RuntimeError("Empty splits detected; ensure train.csv and val.csv contain labeled rows.")
    else:
        if not args.data:
            raise ValueError("Provide either --splits <dir> or --data <csv>.")
        pairs = load_pairs_csv(args.data)
        if len(pairs) < args.val_size + args.test_size + 10:
            raise RuntimeError("Not enough data for requested split sizes.")
        train, val, test = split_dataset(pairs, args.val_size, args.test_size, seed=args.seed)

    # Model + tokenizer with QLoRA
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        # Use bfloat16 for compute to match Qwen defaults and avoid dtype mismatches
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # For GRPO, use a base CausalLM model (no value head)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quant_cfg,
    )
    # Disable cache during training to avoid retaining large KV tensors and wasting VRAM
    try:
        model.config.use_cache = False
    except Exception:
        pass
    # Some TRL components expect this attribute on the model; ensure it exists
    if not hasattr(model, "warnings_issued"):
        try:
            model.warnings_issued = {}
        except Exception:
            pass
    # Guard against dtype mismatches at the final projection by casting inputs to lm_head's dtype
    try:
        def _cast_lm_head_input(module, inputs):
            if not inputs:
                return inputs
            x = inputs[0]
            try:
                target_dtype = module.weight.dtype
                if hasattr(x, "dtype") and x.dtype != target_dtype:
                    x = x.to(target_dtype)
                if len(inputs) == 1:
                    return (x,)
                return (x, *inputs[1:])
            except Exception:
                return inputs
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "register_forward_pre_hook"):
            model.lm_head.register_forward_pre_hook(_cast_lm_head_input)
    except Exception:
        pass

    # Ensure LoRA adapters use bf16 to avoid float32 upcasts in forward
    try:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            lora_dtype=torch.bfloat16,
        )
    except TypeError:
        # Older PEFT versions may not support lora_dtype; fallback without it
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # TRL 0.24: remove unsupported fields like model_name/optimize_cuda_cache
    # Configure GRPO (supports callable reward functions in TRL 0.24)
    # Match effective batch via per_device_train_batch_size * gradient_accumulation_steps ~= args.batch_size
    grad_accum = max(1, args.batch_size // max(1, args.micro_batch))
    tb_dir = args.log_dir or os.path.join(args.out, "tb")
    grpo_config = GRPOConfig(
        output_dir=args.out,
        logging_dir=tb_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=max(1, args.micro_batch),
        gradient_accumulation_steps=grad_accum,
        max_steps=args.steps,
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        logging_strategy="steps",
        logging_steps=max(1, args.log_steps),
        save_strategy="steps",
        save_steps=args.eval_interval,
        remove_unused_columns=False,
        max_prompt_length=args.max_prompt_tokens,
        max_completion_length=args.max_new_tokens,
        num_generations=max(2, args.num_generations),
        # GRPO samples multiple generations; temperature must be > 0
        temperature=0.7,
        top_p=0.95,
        report_to=["tensorboard"],
        optim=args.optim,
        dataloader_num_workers=max(0, args.dataloader_workers),
        dataloader_pin_memory=True,
        # KL/regularization strength for GRPO (TRL 0.24 uses `beta`)
        beta=args.kl_beta,
    )

    # Build training dataset of prompts
    train_prompts = [build_prompt(ex.query, ex.abstract, args.max_abstract_chars) for ex in train]
    train_labels = [ex.label for ex in train]
    prompt_to_label = {p: int(y) for p, y in zip(train_prompts, train_labels)}
    ds = Dataset.from_dict({"prompt": train_prompts})

    # Reward function compatible with GRPO (prompts, completions) -> rewards
    # TRL 0.24 GRPO passes extra keyword args like completion_ids; accept and ignore them
    def reward_fn(prompts: list, completions: list, completion_ids=None, prompt_ids=None, **kwargs) -> list[float]:
        rewards = []
        for p, comp in zip(prompts, completions):
            gold = prompt_to_label.get(p, 0)
            ans, fmt_ok = parse_answer(comp)
            rew = compute_reward(comp, gold, fmt_ok, fmt_penalty=args.fmt_penalty)
            rewards.append(rew)
        return rewards

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Simple evaluation helper
    def quick_eval(split: List[Example], n: int = 200) -> Tuple[float, float, float, float, float]:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        import numpy as np
        # Stratified sample up to n examples to avoid single-class AUROC issues
        pos = [ex for ex in split if int(ex.label) == 1]
        neg = [ex for ex in split if int(ex.label) == 0]
        k_pos = min(len(pos), max(1, n // 2))
        k_neg = min(len(neg), max(1, n - k_pos))
        rng = random.Random(args.seed)
        xs = []
        if pos:
            xs.extend(rng.sample(pos, k_pos))
        if neg:
            xs.extend(rng.sample(neg, k_neg))
        # If still short (e.g., highly imbalanced), top-up randomly
        if len(xs) < min(n, len(split)):
            remaining = [ex for ex in split if ex not in xs]
            take = min(min(n, len(split)) - len(xs), len(remaining))
            if take > 0:
                xs.extend(rng.sample(remaining, take))
        preds = []
        probs = []
        golds = []
        fmt_ok_count = 0
        model.eval()
        for ex in xs:
            prompt = build_prompt(ex.query, ex.abstract, args.max_abstract_chars)
            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen = tokenizer.decode(out[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
            ans, fmt_ok = parse_answer(gen)
            fmt_ok_count += 1 if fmt_ok else 0
            pred = 1 if ans == "yes" else 0
            preds.append(pred)
            golds.append(int(ex.label))
            # crude prob via next-token logits
            from scripts.rl.reward_fn import yes_probability
            p_yes = yes_probability(model, tokenizer, prompt, device=str(model.device))
            probs.append(p_yes)
        acc = float(accuracy_score(golds, preds))
        prec = float(precision_score(golds, preds, zero_division=0))
        rec = float(recall_score(golds, preds, zero_division=0))
        try:
            # AUROC is undefined if only one class is present in golds; guard explicitly
            if len(set(golds)) < 2:
                auroc = float('nan')
            else:
                auroc = float(roc_auc_score(golds, probs))
        except Exception:
            auroc = float('nan')
        fmt_ok_rate = float(fmt_ok_count / max(1, len(preds)))
        fmt_violation_rate = 1.0 - fmt_ok_rate
        return acc, prec, rec, auroc, fmt_violation_rate

    # Lightweight CSV logger for eval metrics
    metrics_csv = os.path.join(args.out, "metrics.csv")
    if not os.path.exists(metrics_csv):
        try:
            with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["step", "split", "acc", "prec", "rec", "auroc", "fmt_violation_rate"])
        except Exception:
            pass

    class EvalCallback(TrainerCallback):
        def __init__(self, eval_every: int, val_split: List[Example]):
            self.eval_every = max(1, eval_every)
            self.val_split = val_split
            self.writer: SummaryWriter | None = None

        def on_train_begin(self, args_, state, control, **kwargs):
            # Baseline metrics at step 0 for immediate visibility in TensorBoard
            try:
                if self.writer is None:
                    self.writer = SummaryWriter(log_dir=tb_dir)
                acc, prec, rec, auroc, fmt_v = quick_eval(self.val_split, n=min(len(self.val_split), 200))
                trainer.log({
                    "acc_val": acc,
                    "prec_val": prec,
                    "rec_val": rec,
                    "auroc_val": auroc,
                    "fmt_violation_val": fmt_v,
                })
                # Write under eval/* so they appear clearly separate from train/*
                try:
                    self.writer.add_scalar("eval/accuracy", acc, 0)
                    self.writer.add_scalar("eval/precision", prec, 0)
                    self.writer.add_scalar("eval/recall", rec, 0)
                    self.writer.add_scalar("eval/auroc", auroc, 0)
                    self.writer.add_scalar("eval/format_violation_rate", fmt_v, 0)
                    self.writer.flush()
                except Exception:
                    pass
                with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([0, "val", acc, prec, rec, auroc, fmt_v])
            except Exception:
                pass

        def on_step_end(self, args_, state, control, **kwargs):
            step = int(state.global_step or 0)
            if step > 0 and (step % self.eval_every == 0):
                try:
                    acc, prec, rec, auroc, fmt_v = quick_eval(self.val_split, n=min(len(self.val_split), 200))
                    # Log to trainer for console/TensorBoard
                    trainer.log({
                        "acc_val": acc,
                        "prec_val": prec,
                        "rec_val": rec,
                        "auroc_val": auroc,
                        "fmt_violation_val": fmt_v,
                    })
                    # Also write to TensorBoard manually under eval/*
                    try:
                        if self.writer is None:
                            self.writer = SummaryWriter(log_dir=tb_dir)
                        self.writer.add_scalar("eval/accuracy", acc, step)
                        self.writer.add_scalar("eval/precision", prec, step)
                        self.writer.add_scalar("eval/recall", rec, step)
                        self.writer.add_scalar("eval/auroc", auroc, step)
                        self.writer.add_scalar("eval/format_violation_rate", fmt_v, step)
                        self.writer.flush()
                    except Exception:
                        pass
                    # Append to CSV
                    try:
                        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            w.writerow([step, "val", acc, prec, rec, auroc, fmt_v])
                    except Exception:
                        pass
                except Exception:
                    pass
            return control

        def on_train_end(self, args_, state, control, **kwargs):
            try:
                if self.writer is not None:
                    self.writer.flush()
                    self.writer.close()
            except Exception:
                pass

    # Register periodic evaluation
    try:
        trainer.add_callback(EvalCallback(args.eval_interval, val))
    except Exception:
        pass

    # Run training using GRPOTrainer's built-in loop
    trainer.train()
    # Save final checkpoint
    final_dir = os.path.join(args.out, "checkpoints", "final")
    os.makedirs(final_dir, exist_ok=True)
    # TRL GRPOTrainer may not expose save_pretrained; prefer saving the underlying model/tokenizer
    try:
        if hasattr(trainer, "save_model"):
            # If available, this saves model weights and, in some implementations, tokenizer/config
            trainer.save_model(final_dir)
        else:
            trainer.model.save_pretrained(final_dir)
    except Exception:
        # Fallback: save just the wrapped model (PEFT adapters will be saved if present)
        try:
            trainer.model.save_pretrained(final_dir)
        except Exception:
            pass
    # Always try to save tokenizer alongside the model for easier reload
    try:
        tokenizer.save_pretrained(final_dir)
    except Exception:
        pass
    # Quick end-of-run eval (val + test if available)
    try:
        acc, prec, rec, auroc, _ = quick_eval(val, n=min(len(val), 200))
        print({"acc_val": acc, "prec_val": prec, "rec_val": rec, "auroc_val": auroc})
        if test:
            acc_t, prec_t, rec_t, auroc_t, _ = quick_eval(test, n=min(len(test), 200))
            print({"acc_test": acc_t, "prec_test": prec_t, "rec_test": rec_t, "auroc_test": auroc_t})
    except Exception:
        pass
    print(f"Training complete: out={args.out}")


if __name__ == "__main__":
    main()
