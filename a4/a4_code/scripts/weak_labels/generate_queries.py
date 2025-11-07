r"""
Generate diverse ML queries using a local Ollama model (default llama3:8b).
Writes a CSV with columns: query_id,query
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from typing import List
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.weak_labels.lib.llm_vote import (
    ollama_generate
)

TOPICS = [
    "computer vision","natural language processing","reinforcement learning","theory","optimization",
    "robotics","multimodal","graphs","fairness","privacy","federated learning","time series",
    "audio","speech","compilers for ML","distributed training","efficient inference","transformers",
]

PROMPT = (
    "You generate short search queries for machine learning literature.\n"
    "- 14 words max, 6 words min.\n"
    "- No punctuation except spaces.\n"
    "- No numbering. One query per line.\n"
    "- Diverse across subfields.\n"
    "Example queries:\n"
    "- Recent progress in Vision Transformers (ViT) for instance segmentation on medical images.\n"
    "- Meta-learning approaches for few-shot image classification with limited training data.\n"
    "- Bayesian Optimization techniques for tuning hyperparameters of large sequential models.\n"
    "Produce {n} queries.\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--model", default="llama3:8b")
    args = ap.parse_args()

    topic = random.choice(TOPICS)
    # Request multiline output: increase num_predict and disable newline stop tokens
    text = ollama_generate(
        args.ollama,
        args.model,
        PROMPT.format(n=args.n) + f"\nSubfield to cover: {topic}.",
        temperature=0.7,
        num_predict=min(2048, max(64, args.n * 12)),
        stop=[],  # allow newlines so model can emit many lines
    )
    # split lines, dedupe, filter lengths
    qs: List[str] = []
    seen = set()
    for line in text.splitlines():
        q = " ".join(line.strip().split())
        # Remove leading bullets/numbering like "1.", "-", "*", "•"
        q = re.sub(r"^\s*(?:[-*•]+|\d+[\.)]\s*)", "", q)
        if not q:
            continue
        wc = len(q.split())
        if wc < 4 or wc > 12:
            continue
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        qs.append(q)
    # downselect to the requested count
    random.shuffle(qs)
    qs = qs[: args.n]

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id","query"])
        for i, q in enumerate(qs):
            w.writerow([i, q])
    print(f"wrote {len(qs)} queries -> {args.out}")


if __name__ == "__main__":
    main()
