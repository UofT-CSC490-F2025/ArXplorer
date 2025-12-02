from __future__ import annotations

import logging
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

ML_KEYWORDS = [
    "machine learning",
    "deep learning",
    "neural",
    "representation learning",
    "reinforcement learning",
    "nlp",
    "computer vision",
    "transformer",
    "diffusion",
]


def is_ml_paper(record: dict) -> bool:
    title = (record.get("title") or "").lower()
    categories = " ".join(record.get("categories", [])).lower()
    text = f"{title} {categories}"
    return any(k in text for k in ML_KEYWORDS)


def normalize_record(record: dict) -> dict:
    return {
        "id": record.get("id") or record.get("paper_id") or record.get("arxiv_id"),
        "title": record.get("title", "").strip(),
        "abstract": record.get("abstract", "").strip(),
        "authors": record.get("authors", []),
        "year": record.get("year") or record.get("published_year"),
        "categories": record.get("categories", []),
    }


def filter_and_normalize(records: Iterable[dict], max_samples: int | None = None) -> Iterator[dict]:
    count = 0
    for rec in records:
        if max_samples is not None and count >= max_samples:
            break
        if not is_ml_paper(rec):
            continue
        yield normalize_record(rec)
        count += 1


def build_text_template(rec: dict) -> str:
    authors = ", ".join(rec.get("authors", []))
    title = rec.get("title", "")
    year = rec.get("year", "")
    abstract = rec.get("abstract", "")
    return f"Title: {title}\nAuthors: {authors}\nYear: {year}\nAbstract: {abstract}"
