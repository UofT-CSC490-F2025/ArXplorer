"""
Verifiable reward function and scoring helpers for the judge.

Reward:
- 1.0 if output == gold (yes/no), 0.0 otherwise.
- Optional penalties for format violations and long outputs.

Also includes a helper to estimate P(yes) for AUROC by scoring the
next-token probabilities of 'yes' vs 'no'. This is an approximation.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch

from .prompts import parse_answer


def compute_reward(
    model_answer: str,
    gold_label: int,
    format_ok: bool,
    fmt_penalty: float = 1.0,
    length_penalty: float = 0.0,
) -> float:
    """
    gold_label: 1 for 'yes', 0 for 'no'
    format_ok: True if strictly 'yes' or 'no' per contract
    """
    gold = "yes" if gold_label == 1 else "no"
    parsed, _ = parse_answer(model_answer)
    base = 1.0 if parsed == gold else 0.0
    penalty = 0.0
    # Enforce strict formatting: a correct answer with bad format gets zero reward by default
    # (fmt_penalty=1.0 ensures base - penalty == 0). Increase fmt_penalty > 1.0 to make it negative.
    if not format_ok:
        penalty += fmt_penalty
    if length_penalty > 0 and len((model_answer or "").strip()) > 8:
        penalty += length_penalty
    return max(0.0, base - penalty)


@torch.no_grad()
def yes_probability(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    Approximate P(yes) by computing next-token probabilities of the first token
    of 'yes' vs 'no' given the prompt. This ignores multi-token sequences and
    assumes first-token discrimination is indicative.
    """
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model(**enc)
    logits = out.logits  # [B, T, V]
    last = logits[:, -1, :]
    # Get first-token ids for 'yes' and 'no'
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    if not yes_ids or not no_ids:
        return 0.5
    yes_id = yes_ids[0]
    no_id = no_ids[0]
    probs = torch.softmax(last, dim=-1)
    p_yes = float(probs[0, yes_id].item())
    p_no = float(probs[0, no_id].item())
    denom = p_yes + p_no
    if denom <= 0:
        return 0.5
    return p_yes / denom
