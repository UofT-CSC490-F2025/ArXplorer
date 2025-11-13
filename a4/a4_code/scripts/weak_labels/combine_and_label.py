r"""
Combine weak signals into a relevance label and write the final dataset.
Inputs: scores.csv from score_signals.py
Outputs: query_abstract_pairs_weak.csv with columns:
  query_id,query,abstract_id,abstract,label,confidence,llm_votes,cosine,bm25,xenc_score,overlap_idf,num_terms_overlap
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, List


essential_in = ["query_id","query","abstract_id","abstract","bm25","cosine","llm_votes","llm_score","overlap_idf","num_terms_overlap","xenc_score"]


def parse_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pos-th', type=float, default=0.8)
    ap.add_argument('--neg-th', type=float, default=0.2)
    ap.add_argument('--balance', action='store_true')
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    with open(args.scores, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Weighted sum (precision-oriented). Adjust as needed.
    out_rows: List[Dict[str, str]] = []
    for r in rows:
        xenc = parse_float(r.get('xenc_score','0'))
        llm = parse_float(r.get('llm_score','0'))
        cos = parse_float(r.get('cosine','0'))
        bm25 = parse_float(r.get('bm25','0'))
        # keyword features as small bump
        overlap = parse_float(r.get('overlap_idf','0'))
        ov_bump = 0.03 if overlap > 0 else 0.0
        score = 0.50 * xenc + 0.30 * llm + 0.15 * cos + 0.05 * bm25 + ov_bump
        label = 1 if score >= args.pos_th else (0 if score <= args.neg_th else -1)
        conf = abs(score - 0.5)
        out = {
            'query_id': r['query_id'],
            'query': r['query'],
            'abstract_id': r['abstract_id'],
            'abstract': r['abstract'],
            'label': str(label if label in (0,1) else -1),
            'confidence': f"{conf:.6f}",
            'llm_votes': r.get('llm_votes','0'),
            'cosine': r.get('cosine','0'),
            'bm25': r.get('bm25','0'),
            'xenc_score': r.get('xenc_score','0'),
            'overlap_idf': r.get('overlap_idf','0'),
            'num_terms_overlap': r.get('num_terms_overlap','0'),
        }
        out_rows.append(out)

    # Balance classes if requested (downsample negatives to match positives)
    if args.balance:
        pos = [r for r in out_rows if r['label'] == '1']
        neg = [r for r in out_rows if r['label'] == '0']
        other = [r for r in out_rows if r['label'] == '-1']
        k = min(len(pos), len(neg))
        neg = neg[:k]
        pos = pos[:k]
        out_rows = pos + neg  # drop uncertain; you can append 'other' if desired

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['query_id','query','abstract_id','abstract','label','confidence','llm_votes','cosine','bm25','xenc_score','overlap_idf','num_terms_overlap'])
        for r in out_rows:
            w.writerow([r['query_id'], r['query'], r['abstract_id'], r['abstract'], r['label'], r['confidence'], r['llm_votes'], r['cosine'], r['bm25'], r['xenc_score'], r['overlap_idf'], r['num_terms_overlap']])
    print(f"wrote {len(out_rows)} labeled pairs -> {args.out}")


if __name__ == '__main__':
    main()
