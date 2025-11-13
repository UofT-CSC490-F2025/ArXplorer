r"""
Make a clean dataset from the weak-label output.

Reads: data/weak_labels/query_abstract_pairs_weak.csv (default)
Writes: CSV with columns: query_id, query, abstract, label

Also prints label distribution counts (and percentages).

Usage (from repo root):
  python -m scripts.weak_labels.make_clean_dataset \
    --in ./data/weak_labels/query_abstract_pairs_weak.csv \
    --out ./data/weak_labels/query_abstract_pairs_clean.csv

Options:
    --drop-uncertain   Drop rows with label == -1 (defaults to keep them).
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from typing import Dict, List


def parse_label(x: str) -> int:
    try:
        return int(x)
    except Exception:
        # Map anything unexpected to uncertain
        return -1


def main():
    ap = argparse.ArgumentParser(description="Make a clean (query_id, query, abstract, label) CSV and report label balance")
    ap.add_argument("--in", dest="inp", default="./data/weak_labels/query_abstract_pairs_weak.csv", help="Input CSV path")
    ap.add_argument("--out", dest="out", default="./data/weak_labels/query_abstract_pairs_clean.csv", help="Output CSV path")
    ap.add_argument("--drop-uncertain", action="store_true", help="Drop rows with label == -1")
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    with open(args.inp, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        # Basic header sanity check
        required = {"query", "abstract", "label"}
        missing = [c for c in ["query", "abstract", "label"] if c not in rdr.fieldnames or rdr.fieldnames is None]
        if missing:
            # We accept the schema emitted by combine_and_label.py and then select columns
            pass
        for r in rdr:
            rows.append(r)

    clean_rows: List[Dict[str, str]] = []
    # Ensure stable query_ids: prefer input 'query_id' if present; else assign per unique query
    qid_map: Dict[str, str] = {}
    next_qid = 1
    counts: Counter[int] = Counter()
    for r in rows:
        q = (r.get("query") or "").strip()
        a = (r.get("abstract") or "").strip()
        lab = parse_label(str(r.get("label", "-1")))
        if args.drop_uncertain and lab == -1:
            continue
        # Keep only non-empty entries
        if not q or not a:
            continue
        # determine query_id
        qid_val = (r.get("query_id") or "").strip()
        if not qid_val:
            # assign deterministic id per unique query
            if q not in qid_map:
                qid_map[q] = str(next_qid)
                next_qid += 1
            qid_val = qid_map[q]
        clean_rows.append({"query_id": qid_val, "query": q, "abstract": a, "label": str(lab)})
        counts[lab] += 1

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query", "abstract", "label"])  # header
        for r in clean_rows:
            w.writerow([r["query_id"], r["query"], r["abstract"], r["label"]])

    total = sum(counts.values()) or 1
    def pct(n: int) -> float:
        return 100.0 * n / total
    print(f"Wrote {len(clean_rows)} rows -> {args.out}")
    print("Label distribution:")
    for k in sorted(counts.keys()):
        print(f"  label={k}: {counts[k]} ({pct(counts[k]):.2f}%)")


if __name__ == "__main__":
    main()
