r"""
Score (query, abstract) pairs with keyword overlap, LLM votes, and optional cross-encoder.
Inputs: candidates.csv with columns: query_id,query,abstract_id,abstract,bm25,cosine
Outputs: scores.csv with added columns: llm_votes,llm_score,overlap_idf,num_terms_overlap,xenc_score
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.weak_labels.lib.text_utils import doc_tokens, compute_idf, overlap_idf, term_overlap_count
from scripts.weak_labels.lib.llm_vote import strict_vote

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # type: ignore


def read_candidates(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


essential_cols = ["query_id","query","abstract_id","abstract","bm25","cosine"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ollama', default='http://localhost:11434')
    ap.add_argument('--llm-model', default='llama3:8b')
    ap.add_argument('--llm-votes', type=int, default=3)
    ap.add_argument('--xenc', action='store_true', help='Enable cross-encoder scoring')
    ap.add_argument('--xenc-model', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    ap.add_argument('--xenc-top', type=int, default=1000, help='Score this many top pairs by max(bm25, cosine)')
    args = ap.parse_args()

    rows = read_candidates(args.candidates)

    # Build IDF over abstracts (tokenized) for overlap features
    abs_tokens: Dict[str, List[str]] = {}
    docs = []
    for r in rows:
        aid = r['abstract_id']
        if aid not in abs_tokens:
            toks = doc_tokens(r['abstract'])
            abs_tokens[aid] = toks
            docs.append(toks)
    idf = compute_idf(docs)

    # Identify top-N for cross-encoder if enabled
    top_idx = set()
    if args.xenc and CrossEncoder is not None:
        # Build a score = max(bm25, cosine)
        scored = []
        for i, r in enumerate(rows):
            try:
                bm = float(r['bm25'])
            except Exception:
                bm = 0.0
            try:
                co = float(r['cosine'])
            except Exception:
                co = 0.0
            scored.append((i, max(bm, co)))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_idx = {i for i, _ in scored[: args.xenc_top]}
        xenc = CrossEncoder(args.xenc_model) if CrossEncoder is not None else None
    else:
        xenc = None

    # Score rows
    out_rows: List[Dict[str, str]] = []
    for i, r in enumerate(tqdm(rows, desc='scoring')):
        q = r['query']
        a = r['abstract']
        qt = doc_tokens(q)
        at = abs_tokens[r['abstract_id']]
        r['overlap_idf'] = f"{overlap_idf(qt, at, idf):.6f}"
        r['num_terms_overlap'] = str(term_overlap_count(qt, at))
        # LLM votes
        try:
            yes, score = strict_vote(
                prompt=f"You are a relevance judge. Answer only with 'yes' or 'no' in lowercase.\nQuestion: Is the following abstract relevant to the query?\n\nQuery: {q}\nAbstract: {a}\n\nAnswer (yes/no only):",
                base_url=args.ollama,
                model=args.llm_model,
                votes=max(1, args.llm_votes),
                temperature=0.7,
            )
            r['llm_votes'] = str(int(yes))
            r['llm_score'] = f"{float(score):.6f}"
        except Exception:
            r['llm_votes'] = '0'
            r['llm_score'] = '0.000000'
        # Cross-encoder (optional, top-N only)
        if xenc is not None and i in top_idx:
            try:
                s = float(xenc.predict([(q, a)])[0])
                r['xenc_score'] = f"{s:.6f}"
            except Exception:
                r['xenc_score'] = '0.000000'
        else:
            r['xenc_score'] = r.get('xenc_score', '0.000000')
        out_rows.append(r)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=essential_cols + ['llm_votes','llm_score','overlap_idf','num_terms_overlap','xenc_score'])
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, '') for k in essential_cols + ['llm_votes','llm_score','overlap_idf','num_terms_overlap','xenc_score']})
    print(f'wrote {len(out_rows)} scored rows -> {args.out}')


if __name__ == '__main__':
    main()
