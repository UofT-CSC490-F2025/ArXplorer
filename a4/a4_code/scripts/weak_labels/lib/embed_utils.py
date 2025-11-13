r"""
Sentence-transformers embedding helpers.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str], batch_size: int = 128, normalize: bool = True) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, device=self.device, normalize_embeddings=normalize)
        return emb.astype(np.float32)

    @staticmethod
    def cosine_top_k(query_vec: np.ndarray, doc_matrix: np.ndarray, k: int = 50) -> List[Tuple[int, float]]:
        # doc_matrix and query_vec must be L2-normalized when normalize=True above
        sims = doc_matrix @ query_vec
        idx = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in idx]
