"""
Local LLM baseline (Ollama) for (query, abstract) pair relevance classification.

Requires Ollama running locally and the model pulled (default: llama3:8b).
Sends a concise prompt and expects strictly "yes" or "no".
"""

import argparse
import json
import os
import time
from typing import Dict, List

import pandas as pd
import requests


PROMPT_TEMPLATE = (
    "You are a strict binary classifier for academic search relevance.\n"
    "Given a user query and an academic abstract, decide if the abstract is relevant to the query.\n"
    "Answer with a single token: 'yes' or 'no' (lowercase). No other text.\n\n"
    "Query:\n{query}\n\n"
    "Abstract:\n{abstract}\n\n"
    "Answer:"
)


def call_ollama(model: str, prompt: str, timeout: float = 60.0, base_url: str = "http://localhost:11434") -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    text = (data.get("response") or "").strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    # Fallback: keep only first token
    toks = text.split()
    return toks[0] if toks else "no"


def evaluate_split(df: pd.DataFrame, model: str, rate_limit: float = 0.0, base_url: str = "http://localhost:11434") -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    pred_pos = 0
    preds: List[Dict] = []
    for i in range(len(df)):
        q = str(df.iloc[i]["query"])[:2000]
        a = str(df.iloc[i]["abstract"])[:4000]
        y = int(df.iloc[i]["label"])
        prompt = PROMPT_TEMPLATE.format(query=q, abstract=a)
        try:
            ans = call_ollama(model, prompt, base_url=base_url)
        except Exception:
            # If the local server fails, default to 'no'
            ans = "no"
        pred = 1 if ans == "yes" else 0
        if pred == 1:
            pred_pos += 1
            if y == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y == 1:
                fn += 1
            else:
                tn += 1
        preds.append({
            "query_id": str(df.iloc[i]["query_id"]),
            "pred": pred,
            "label": y,
            "raw": ans,
        })
        if rate_limit > 0:
            time.sleep(rate_limit)
    precision = tp / max(1, pred_pos)
    pos = int((df["label"] == 1).sum())
    recall = tp / max(1, pos)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    total = len(df)
    accuracy = (tp + tn) / max(1, total)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "pos": int(pos),
        "pred_pos": int(pred_pos),
    }, preds


def main():
    ap = argparse.ArgumentParser(description="Local LLM (Ollama) baseline for P@1")
    ap.add_argument("--splits", required=True, help="Directory with val.csv and test.csv")
    ap.add_argument("--model", default="llama3:8b")
    ap.add_argument("--outdir", default="experiments/llm_local")
    ap.add_argument("--rate-limit", type=float, default=0.0, help="Seconds to sleep between requests")
    ap.add_argument(
        "--ollama-base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Base URL for Ollama API (e.g., http://localhost:11434 or http://<windows-host-ip>:11434)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    val = pd.read_csv(os.path.join(args.splits, "val.csv"))
    test = pd.read_csv(os.path.join(args.splits, "test.csv"))

    val_stats, val_preds = evaluate_split(val, args.model, rate_limit=args.rate_limit, base_url=args.ollama_base_url)
    test_stats, test_preds = evaluate_split(test, args.model, rate_limit=args.rate_limit, base_url=args.ollama_base_url)

    with open(os.path.join(args.outdir, "preds_val.json"), "w", encoding="utf-8") as f:
        json.dump({"preds": val_preds}, f)
    with open(os.path.join(args.outdir, "preds_test.json"), "w", encoding="utf-8") as f:
        json.dump({"preds": test_preds}, f)

    metrics = {"model": args.model, "ollama_base_url": args.ollama_base_url, "val": val_stats, "test": test_stats}
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
