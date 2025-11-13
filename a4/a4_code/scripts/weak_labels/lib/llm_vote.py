r"""
LLM voting via Ollama (HTTP API). Strict yes/no format, self-consistency.
"""
from __future__ import annotations

from typing import List, Tuple
import requests

from scripts.rl.prompts import parse_answer


def ollama_generate(base_url: str, model: str, prompt: str, temperature: float = 0.7, num_predict: int = 4, stop: List[str] | None = None, timeout: int = 60) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            # Stop on newline to keep single-token-ish outputs when possible
            "stop": stop if stop is not None else ["\n"]
        },
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return str(data.get("response", "")).strip()


def strict_vote(prompt: str, *, base_url: str = "http://localhost:11434", model: str = "llama3:8b", votes: int = 3, temperature: float = 0.7) -> Tuple[int, float]:
    """
    Returns (num_yes, score) where score in [0,1] is num_yes/num_votes if all answers strictly parse,
    otherwise averaged over strictly parsed votes only. Unparsable votes count as 0 and reduce confidence.
    """
    yes = 0
    ok = 0
    for _ in range(max(1, votes)):
        text = ollama_generate(base_url, model, prompt, temperature=temperature)
        ans, fmt_ok = parse_answer(text)
        if fmt_ok and ans in ("yes", "no"):
            ok += 1
            if ans == "yes":
                yes += 1
    score = (yes / ok) if ok > 0 else 0.0
    return yes, score
