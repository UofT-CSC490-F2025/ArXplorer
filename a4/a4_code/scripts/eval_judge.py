r"""
Evaluate a trained relevance judge that outputs strictly: yes or no.

Reads a CSV with columns: query,abstract,label
Loads a base model (optionally with LoRA checkpoint) and runs greedy decoding.
Metrics: accuracy, precision, recall, AUROC (from yes-prob), format violation rate.

Example (PowerShell):
python .\scripts\eval_judge.py `
  --data .\data\query_abstract_pairs_synth.csv `
  --ckpt .\runs\ppo_qwen15b\checkpoints\step_3000 `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --out .\runs\eval_step_3000
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead

import sys
import os


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_ROOT = _project_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.rl.prompts import build_prompt, parse_answer  # type: ignore
from scripts.rl.reward_fn import yes_probability  # type: ignore


class Example:
    def __init__(self, query: str, abstract: str, label: int):
        self.query = query
        self.abstract = abstract
        self.label = label


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


def main():
    ap = argparse.ArgumentParser(description="Evaluate relevance judge (yes/no)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=False, help="LoRA/TRL checkpoint dir; if omitted, uses base model only")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-abstract-chars", type=int, default=2000)
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    exs = load_pairs_csv(args.data)
    if args.limit and args.limit > 0:
        exs = exs[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    if args.ckpt and os.path.isdir(args.ckpt):
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.ckpt,
            device_map="auto",
            dtype=torch.float16,
            quantization_config=quant_cfg,
        )
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model,
            device_map="auto",
            dtype=torch.float16,
            quantization_config=quant_cfg,
        )

    preds = []
    probs = []
    golds = []
    fmt_total = 0
    fmt_viol = 0

    model.eval()
    # Robust device detection (works for PEFT/ValueHead wrappers)
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ex in exs:
        prompt = build_prompt(ex.query, ex.abstract, args.max_abstract_chars)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        ans, fmt_ok = parse_answer(gen)
        fmt_total += 1
        if not fmt_ok:
            fmt_viol += 1
        pred = 1 if ans == "yes" else 0
        preds.append(pred)
        golds.append(int(ex.label))
        p_yes = yes_probability(model, tokenizer, prompt, device=str(device))
        probs.append(p_yes)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    acc = float(accuracy_score(golds, preds))
    prec = float(precision_score(golds, preds, zero_division=0))
    rec = float(recall_score(golds, preds, zero_division=0))
    f1 = float(f1_score(golds, preds, zero_division=0))
    # Confusion counts
    tp = sum(1 for y, p in zip(golds, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(golds, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(golds, preds) if y == 1 and p == 0)
    try:
        auroc = float(roc_auc_score(golds, probs))
    except Exception:
        auroc = float("nan")
    fmt_violation_rate = fmt_viol / max(1, fmt_total)

    # Write report
    with open(os.path.join(args.out, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"accuracy\t{acc}\n")
        f.write(f"precision\t{prec}\n")
        f.write(f"recall\t{rec}\n")
        f.write(f"f1\t{f1}\n")
        f.write(f"auroc\t{auroc}\n")
        f.write(f"format_violation_rate\t{fmt_violation_rate}\n")
        f.write(f"false_positives\t{fp}\n")
        f.write(f"false_negatives\t{fn}\n")

    print("Evaluation complete.")
    print({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auroc,
        "format_violation_rate": fmt_violation_rate,
        "fp": fp,
        "fn": fn,
    })


if __name__ == "__main__":
    main()
