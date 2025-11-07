"""
Judge prompting utilities.

Contract:
- Input: query (str), abstract (str)
- Output: strictly "yes" or "no" (lowercase, single line)

We keep prompts compact and clamp abstract length to control VRAM.
"""

from __future__ import annotations

import re
from typing import Tuple


def truncate_text(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    t = s[: max_chars]
    # avoid cutting mid-word
    return t.rsplit(" ", 1)[0]


def build_prompt(query: str, abstract: str, max_abstract_chars: int = 2000) -> str:
    a = truncate_text(abstract.strip().replace("\n", " "), max_abstract_chars)
    q = query.strip().replace("\n", " ")
    # Single-turn instruction for causal LM
    return (
        "You are a relevance judge. Answer only with 'yes' or 'no' in lowercase.\n"
        "Question: Is the following abstract relevant to the query?\n\n"
        f"Query: {q}\n"
        f"Abstract: {a}\n\n"
        "Answer (yes/no only):"
    )


YES_RE = re.compile(r"^\s*yes\s*$")
NO_RE = re.compile(r"^\s*no\s*$")


def parse_answer(text: str) -> Tuple[str | None, bool]:
    """
    Returns (normalized_answer, format_ok) where normalized_answer is
    'yes'|'no' or None if not parsable; format_ok is True only if the
    answer is strictly 'yes' or 'no' (single line, lowercase allowed around whitespace).
    """
    if text is None:
        return None, False
    line = text.strip()
    if YES_RE.match(line):
        return "yes", True
    if NO_RE.match(line):
        return "no", True
    # Soft normalization attempt (lowercase, first token)
    low = line.lower().split()[0:1]
    if low == ["yes"]:
        return "yes", False
    if low == ["no"]:
        return "no", False
    return None, False
