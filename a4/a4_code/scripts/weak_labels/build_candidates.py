r"""
Build candidate (query, abstract) pairs using BM25 and embeddings.
Inputs:
- --queries CSV: query_id,query
- --corpus CSV: id,title,abstract
Outputs:
- --out CSV with columns: query_id,query,abstract_id,abstract,bm25,cosine
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from typing import List, Tuple
import sys

import numpy as np
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.weak_labels.lib.bm25_utils import BM25Index
from scripts.weak_labels.lib.embed_utils import Embedder


def load_queries(path: str) -> List[Tuple[int, str]]:
    out = []
    with open(path, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            qid = int(row.get('query_id') or i)
            q = (row.get('query') or '').strip()
            if q:
                out.append((qid, q))
    return out


def load_corpus(path: str) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    abs_list: List[str] = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pid = (row.get('id') or row.get('paper_id') or row.get('arxiv_id') or '').strip()
            a = (row.get('abstract') or row.get('summary') or '').strip()
            if pid and a:
                ids.append(pid)
                abs_list.append(a)
    return ids, abs_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', required=True)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k-bm25', type=int, default=50)
    ap.add_argument('--k-emb', type=int, default=50)
    ap.add_argument('--random-neg', type=int, default=10)
    ap.add_argument('--embed-model', default='all-MiniLM-L6-v2')
    args = ap.parse_args()

    q_list = load_queries(args.queries)
    ids, abstracts = load_corpus(args.corpus)

    print('Building BM25 index...')
    bm25 = BM25Index(abstracts)

    print('Computing embeddings for abstracts...')
    emb = Embedder(args.embed_model)
    abs_emb = emb.encode(abstracts, batch_size=256, normalize=True)

    rng = random.Random(42)

    rows: List[Tuple[int,str,str,str,float,float]] = []
    for qid, q in tqdm(q_list, desc='candidates'):
        # BM25 top-k
        bm = bm25.get_top_k(q, k=args.k_bm25)
        bm_norm = {i: s for i, s in bm25.minmax_normalize(bm)}
        # Embedding top-k
        q_vec = emb.encode([q], batch_size=8, normalize=True)[0]
        top_emb = Embedder.cosine_top_k(q_vec, abs_emb, k=args.k_emb)
        cos = {i: s for i, s in top_emb}
        # Union + random negatives
        idx_set = set(list(bm_norm.keys()) + list(cos.keys()))
        # Add random negatives
        while len(idx_set) < (args.k_bm25 + args.k_emb + args.random_neg) and len(idx_set) < len(ids):
            idx_set.add(rng.randrange(0, len(ids)))
        for i in idx_set:
            rows.append((qid, q, ids[i], abstracts[i], bm_norm.get(i, 0.0), cos.get(i, 0.0)))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['query_id','query','abstract_id','abstract','bm25','cosine'])
        for r in rows:
            w.writerow(list(r))
    print(f'wrote {len(rows)} candidates -> {args.out}')


if __name__ == '__main__':
    main()
