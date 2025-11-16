"""Dense searcher using FAISS index."""

import numpy as np
from typing import List

from .base import BaseSearcher, SearchResult
from ..encoders import DenseEncoder
from ..indexers import DenseIndexer


class DenseSearcher(BaseSearcher):
    """Dense retrieval using FAISS similarity search."""
    
    def __init__(self, indexer: DenseIndexer, encoder: DenseEncoder):
        """
        Args:
            indexer: Loaded dense indexer with FAISS index
            encoder: Dense encoder for query encoding
        """
        self.indexer = indexer
        self.encoder = encoder
        
        if self.indexer.index is None:
            raise ValueError("Indexer must be loaded before creating searcher")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using dense embeddings."""
        # Encode query
        query_emb = self.encoder.encode(query, batch_size=1)
        
        # Search FAISS index
        scores, indices = self.indexer.index.search(query_emb, top_k)
        
        scores = scores[0]
        indices = indices[0]
        
        # Convert to SearchResult objects
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            doc_entry = self.indexer.doc_map.get(idx, {"id": f"Unknown_{idx}"})
            doc_id = doc_entry["id"] if isinstance(doc_entry, dict) else doc_entry
            results.append(SearchResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank
            ))
        
        return results
