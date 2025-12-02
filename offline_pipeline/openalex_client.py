from __future__ import annotations

import logging
import time
from typing import Iterable

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

OPENALEX_URL = "https://api.openalex.org/works/{}"


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(5))
def fetch_openalex(work_id: str, email: str | None = None) -> dict | None:
    params = {}
    if email:
        params["mailto"] = email
    url = OPENALEX_URL.format(work_id)
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def enrich_with_openalex(records: Iterable[dict], email: str | None = None, rate_limit: float = 2.0):
    delay = 1.0 / max(rate_limit, 0.1)
    for rec in records:
        work_id = rec.get("doi") or rec.get("openalex_id")
        if not work_id:
            yield rec
            continue
        try:
            data = fetch_openalex(work_id, email=email)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAlex fetch failed for %s: %s", work_id, exc)
            yield rec
            continue

        if data:
            rec["citation_count"] = data.get("cited_by_count")
            rec["concepts"] = [c.get("display_name") for c in data.get("concepts", []) if c.get("display_name")]
        yield rec
        time.sleep(delay)
