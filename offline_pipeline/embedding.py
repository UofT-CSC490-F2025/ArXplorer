from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_dense_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError("sentence-transformers not installed inside the image") from exc
    logger.info("Loading dense model: %s", model_name)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    return model


def load_sparse_model(model_name: str):
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError("transformers not installed inside the image") from exc
    logger.info("Loading sparse model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return tokenizer, model


def dense_embed(model, texts: List[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def sparse_embed(tokenizer, model, texts: List[str], batch_size: int) -> Tuple[List[List[int]], List[List[float]]]:
    # This produces bag-of-words weights similar to SPLADE-style using MLM logits.
    import torch  # noqa: WPS433

    input_batches = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        input_batches.append(tokenizer(chunk, padding=True, truncation=True, return_tensors="pt"))

    indices: List[List[int]] = []
    values: List[List[float]] = []

    with torch.no_grad():
        for batch in input_batches:
            outputs = model(**batch)
            # Sum log probabilities across sequence length to approximate term importance.
            scores = torch.log_softmax(outputs.logits, dim=-1)
            token_importance = scores.max(dim=1).values  # [batch, vocab]
            for row in token_importance:
                nonzero = torch.nonzero(row > -10, as_tuple=False).squeeze(-1)
                row_vals = row[nonzero].cpu().numpy().astype("float32").tolist()
                row_idx = nonzero.cpu().numpy().astype("int64").tolist()
                indices.append(row_idx)
                values.append(row_vals)
    return indices, values
