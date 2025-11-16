"""Sparse searcher using scipy sparse matrix."""

import numpy as np
import scipy.sparse
from typing import List

from .base import BaseSearcher, SearchResult
from ..encoders import SparseEncoder
from ..indexers import SparseIndexer


class SparseSearcher(BaseSearcher):
    """Sparse retrieval using SPLADE vectors."""
    
    def __init__(self, indexer: SparseIndexer, encoder: SparseEncoder):
        """
        Args:
            indexer: Loaded sparse indexer
            encoder: Sparse encoder for query encoding
        """
        self.indexer = indexer
        self.encoder = encoder
        
        if self.indexer.sparse_matrix is None:
            raise ValueError("Indexer must be loaded before creating searcher")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using sparse vectors."""
        # Encode query
        query_sparse = self.encoder.encode(query, batch_size=1)[0]
        indices, values = query_sparse
        
        # Create sparse query vector
        vocab_size = self.encoder.get_dimension()
        query_vec = scipy.sparse.csr_matrix(
            (values, (np.zeros_like(indices), indices)),
            shape=(1, vocab_size),
            dtype=np.float32
        )
        
        # Compute dot product with all documents
        scores = self.indexer.sparse_matrix.dot(query_vec.transpose())
        scores_dense = scores.toarray().squeeze()
        
        # Get top-k
        if len(scores_dense) < top_k:
            top_k = len(scores_dense)
        
        top_indices = np.argsort(scores_dense)[-top_k:][::-1]
        
        # Convert to SearchResult objects
        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = scores_dense[idx]
            if score <= 0:
                continue
            
            doc_entry = self.indexer.doc_map.get(idx, {"id": f"Unknown_{idx}"})
            doc_id = doc_entry["id"] if isinstance(doc_entry, dict) else doc_entry
            results.append(SearchResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank
            ))
        
        return results
