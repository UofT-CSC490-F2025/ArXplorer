r"""
BM25 retrieval helpers using rank_bm25.
"""
from __future__ import annotations

from typing import List, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

from .text_utils import doc_tokens


class BM25Index:
    def __init__(self, docs: List[str]):
        self._doc_tokens = [doc_tokens(d) for d in docs]
        self._bm25 = BM25Okapi(self._doc_tokens)

    def get_top_k(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        q = doc_tokens(query)
        scores = self._bm25.get_scores(q)
        idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idx]

    @staticmethod
    def minmax_normalize(scores: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if not scores:
            return []
        vals = np.array([s for _, s in scores], dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return [(i, 0.0) for i, _ in scores]
        return [(i, float((s - lo) / (hi - lo))) for i, s in scores]
