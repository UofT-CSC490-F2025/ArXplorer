r"""
Lightweight text utilities for weak-label pipeline.
- tokenization: lowercase alnum tokens
- stopwords: base + ML extras
- IDF computation per corpus
- overlap features
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")

BASE_STOP = {
    "the","a","an","and","or","of","to","in","for","on","is","are","with","by","as","that","this","we","be","from","at","it","its","our","their","using","use","used","can","will","may","such","into","over","under","than","also","these","those","both","have","has","had","however","there","which","within","between","per","via","more","less","based","new","paper","study","work","results","method","approach","show","present","propose","provide","analysis","task","tasks","model","models","data","dataset","learning","machine","deep","neural"
}


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return _TOKEN_RE.findall(text)


def filter_stop(tokens: Iterable[str], stop: Set[str] | None = None) -> List[str]:
    stop = stop or BASE_STOP
    return [t for t in tokens if t not in stop]


def doc_tokens(text: str, stop: Set[str] | None = None) -> List[str]:
    return filter_stop(tokenize(text), stop)


def compute_idf(docs: Iterable[List[str]]) -> Dict[str, float]:
    docs = list(docs)
    N = len(docs)
    df: Counter[str] = Counter()
    for d in docs:
        for tok in set(d):
            df[tok] += 1
    idf: Dict[str, float] = {}
    for tok, c in df.items():
        idf[tok] = math.log((N + 1) / (c + 1)) + 1.0
    return idf


def overlap_idf(q_tokens: List[str], a_tokens: List[str], idf: Dict[str, float]) -> float:
    qs = set(q_tokens)
    inter = qs.intersection(a_tokens)
    return float(sum(idf.get(t, 0.0) for t in inter))


def term_overlap_count(q_tokens: List[str], a_tokens: List[str]) -> int:
    return int(len(set(q_tokens).intersection(a_tokens)))
