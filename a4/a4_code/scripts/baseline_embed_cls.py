"""
Embedding similarity classifier baseline for (query, abstract) pairs.

Encodes queries (titles) and abstracts with a sentence-transformer on GPU if available.
Scores via cosine similarity. Chooses a threshold on the validation set to maximize precision (P@1) among predicted positives.
Reports P@1 on test and saves per-pair predictions.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def load_split(splits_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(splits_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64, device: str = "cuda") -> np.ndarray:
    # sentence-transformers respects model.max_seq_length; ensure truncation to bound VRAM/time
    model.max_seq_length = getattr(model, "max_seq_length", 512) or 512
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, device=device, show_progress_bar=True, normalize_embeddings=False)
    return emb.astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.sum(a_norm * b_norm, axis=1)


def best_precision_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Try thresholds at unique score values (plus small epsilon paddings)
    uniq = np.unique(scores)
    if len(uniq) > 512:
        # Subsample to 512 candidate thresholds for speed
        idx = np.linspace(0, len(uniq) - 1, 512, dtype=int)
        uniq = uniq[idx]
    best_tau = 1.0
    best_prec = -1.0
    best_stats = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "tp": 0, "fp": 0, "fn": 0, "pos": int(labels.sum()), "pred_pos": 0}
    for tau in uniq:
        pred = scores >= tau
        pred_pos = int(pred.sum())
        if pred_pos == 0:
            continue
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = pred_pos - tp
        fn = int((labels == 1).sum()) - tp
        prec = tp / max(1, pred_pos)
        rec = tp / max(1, int((labels == 1).sum()))
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        total = int(len(labels))
        tn = total - tp - fp - fn
        acc = (tp + tn) / max(1, total)
        # Maximize precision; tie-break by more predicted positives to avoid trivial solutions
        if prec > best_prec or (abs(prec - best_prec) < 1e-9 and pred_pos > best_stats["pred_pos"]):
            best_prec = prec
            best_tau = float(tau)
            best_stats = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "accuracy": float(acc),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pos": int((labels == 1).sum()),
                "pred_pos": pred_pos,
            }
    # Fallback: if no positives ever predicted, set tau above max score
    if best_prec < 0:
        best_tau = float(scores.max() + 1e-3)
        best_stats = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "tp": 0, "fp": 0, "fn": 0, "pos": int((labels == 1).sum()), "pred_pos": 0}
    return best_tau, best_stats


def precision_at_1(scores: np.ndarray, labels: np.ndarray, tau: float) -> Dict[str, float]:
    pred = scores >= tau
    pred_pos = int(pred.sum())
    if pred_pos == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "tp": 0, "fp": 0, "fn": 0, "pos": int((labels == 1).sum()), "pred_pos": 0}
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = pred_pos - tp
    fn = int((labels == 1).sum()) - tp
    prec = tp / max(1, pred_pos)
    rec = tp / max(1, int((labels == 1).sum()))
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    total = int(len(labels))
    tn = total - tp - fp - fn
    acc = (tp + tn) / max(1, total)
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pos": int((labels == 1).sum()),
        "pred_pos": pred_pos,
    }


def main():
    ap = argparse.ArgumentParser(description="Embedding similarity classifier baseline (P@1)")
    ap.add_argument("--splits", required=True, help="Directory with train.csv, val.csv, test.csv")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformer model name")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="experiments/embed_cls", help="Directory to write predictions and metrics")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train = load_split(args.splits, "train")
    val = load_split(args.splits, "val")
    test = load_split(args.splits, "test")

    # Load model
    model = SentenceTransformer(args.model, device=args.device)
    # Truncate to manageable length via model.max_seq_length
    model.max_seq_length = getattr(model, "max_seq_length", 512) or 512

    def compute_scores(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        queries = df["query"].astype(str).tolist()
        abstracts = df["abstract"].astype(str).tolist()
        q_emb = encode_texts(model, queries, batch_size=args.batch_size, device=args.device)
        a_emb = encode_texts(model, abstracts, batch_size=args.batch_size, device=args.device)
        scores = cosine_sim(q_emb, a_emb)
        labels = df["label"].astype(int).to_numpy()
        meta = [
            {
                "query_id": str(df.iloc[i]["query_id"]),
                "score": float(scores[i]),
                "label": int(labels[i]),
            }
            for i in range(len(df))
        ]
        return scores, labels, meta

    _, _, _ = compute_scores(train)  # embeddings warm-up
    val_scores, val_labels, val_meta = compute_scores(val)
    tau, val_stats = best_precision_threshold(val_scores, val_labels)

    test_scores, test_labels, test_meta = compute_scores(test)
    test_stats = precision_at_1(test_scores, test_labels, tau)

    # Save predictions
    def save_preds(split: str, meta: List[Dict], scores: np.ndarray, tau: float):
        preds = []
        for i, m in enumerate(meta):
            pred_label = int(scores[i] >= tau)
            preds.append({**m, "pred": pred_label})
        with open(os.path.join(args.outdir, f"preds_{split}.json"), "w", encoding="utf-8") as f:
            json.dump({"threshold": tau, "preds": preds}, f)

    save_preds("val", val_meta, val_scores, tau)
    save_preds("test", test_meta, test_scores, tau)

    # Save metrics
    metrics = {
        "model": args.model,
        "device": args.device,
        "val": {"threshold": tau, **val_stats},
        "test": {"threshold": tau, **test_stats},
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
