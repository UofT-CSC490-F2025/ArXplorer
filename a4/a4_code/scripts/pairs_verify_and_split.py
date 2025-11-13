"""
Verify and split query_abstract_pairs.csv into train/test/val by query_id.

Defaults to 60/20/20 (normalized if custom ratios don't sum to 1).

Writes: data/splits/train.csv, test.csv, val.csv and a summary JSON.
"""

import argparse
import json
import os
import random
from typing import List, Tuple

import pandas as pd


def normalize_ratios(r: List[float]) -> List[float]:
    s = sum(r)
    if s <= 0:
        return [0.6, 0.2, 0.2]
    return [x / s for x in r]


def split_by_query(df: pd.DataFrame, ratios: List[float], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    qids = list(df["query_id"].astype(str).unique())
    random.shuffle(qids)
    n = len(qids)
    r = normalize_ratios(ratios)
    n_train = int(r[0] * n)
    n_test = int(r[1] * n)
    n_val = n - n_train - n_test
    train_ids = set(qids[:n_train])
    test_ids = set(qids[n_train:n_train + n_test])
    val_ids = set(qids[n_train + n_test:])
    train = df[df["query_id"].astype(str).isin(train_ids)].copy()
    test = df[df["query_id"].astype(str).isin(test_ids)].copy()
    val = df[df["query_id"].astype(str).isin(val_ids)].copy()
    return train, test, val


def main():
    ap = argparse.ArgumentParser(description="Verify and split query_abstract_pairs.csv into train/test/val")
    ap.add_argument("--pairs", required=True, help="Path to query_abstract_pairs.csv")
    ap.add_argument("--out", required=True, help="Output directory for splits")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.6, 0.2, 0.2], help="Ratios for train test val (will be normalized)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.pairs)

    # Basic checks and cleanup
    before = len(df)
    df = df.drop_duplicates(subset=["query_id", "query", "abstract", "label"]).reset_index(drop=True)
    dropped_dupes = before - len(df)
    df = df.dropna(subset=["query", "abstract"]).reset_index(drop=True)
    df = df[df["abstract"].str.len() >= 50].reset_index(drop=True)

    # Split by query_id
    train, test, val = split_by_query(df, args.ratios, args.seed)

    # Save
    train.to_csv(os.path.join(args.out, "train.csv"), index=False)
    test.to_csv(os.path.join(args.out, "test.csv"), index=False)
    val.to_csv(os.path.join(args.out, "val.csv"), index=False)

    summary = {
        "total_rows": int(len(df)),
        "dropped_duplicates": int(dropped_dupes),
        "n_queries": int(df["query_id"].nunique()),
        "splits": {
            "train": {"rows": int(len(train)), "queries": int(train["query_id"].nunique())},
            "test": {"rows": int(len(test)), "queries": int(test["query_id"].nunique())},
            "val": {"rows": int(len(val)), "queries": int(val["query_id"].nunique())},
        },
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
