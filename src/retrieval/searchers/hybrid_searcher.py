"""Hybrid searcher combining dense and sparse retrieval with RRF."""

from typing import List, Dict
from collections import defaultdict

from .base import BaseSearcher, SearchResult
from .dense_searcher import DenseSearcher
from .sparse_searcher import SparseSearcher


class HybridSearcher(BaseSearcher):
    """Hybrid retrieval using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(
        self,
        dense_searcher: DenseSearcher,
        sparse_searcher: SparseSearcher,
        rrf_k: int = 60
    ):
        """
        Args:
            dense_searcher: Dense searcher instance
            sparse_searcher: Sparse searcher instance
            rrf_k: RRF constant (default 60 as per literature)
        """
        self.dense_searcher = dense_searcher
        self.sparse_searcher = sparse_searcher
        self.rrf_k = rrf_k
    
    def search(self, query: str, top_k: int = 10, retrieval_k: int = 100) -> List[SearchResult]:
        """
        Search using hybrid RRF fusion.
        
        Args:
            query: Query text
            top_k: Final number of results to return
            retrieval_k: Number of results to retrieve from each system before fusion
            
        Returns:
            List of SearchResult objects fused via RRF
        """
        # Retrieve from both systems
        dense_results = self.dense_searcher.search(query, top_k=retrieval_k)
        sparse_results = self.sparse_searcher.search(query, top_k=retrieval_k)
        
        # Apply Reciprocal Rank Fusion
        rrf_scores = defaultdict(float)
        doc_info = {}  # Store original scores and ranks
        
        # Add dense results
        for result in dense_results:
            rrf_scores[result.doc_id] += 1.0 / (self.rrf_k + result.rank)
            if result.doc_id not in doc_info:
                doc_info[result.doc_id] = {
                    'dense_rank': result.rank,
                    'dense_score': result.score,
                    'sparse_rank': None,
                    'sparse_score': None
                }
            else:
                doc_info[result.doc_id]['dense_rank'] = result.rank
                doc_info[result.doc_id]['dense_score'] = result.score
        
        # Add sparse results
        for result in sparse_results:
            rrf_scores[result.doc_id] += 1.0 / (self.rrf_k + result.rank)
            if result.doc_id not in doc_info:
                doc_info[result.doc_id] = {
                    'dense_rank': None,
                    'dense_score': None,
                    'sparse_rank': result.rank,
                    'sparse_score': result.score
                }
            else:
                doc_info[result.doc_id]['sparse_rank'] = result.rank
                doc_info[result.doc_id]['sparse_score'] = result.score
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create final results
        results = []
        for rank, (doc_id, rrf_score) in enumerate(sorted_docs[:top_k], 1):
            results.append(SearchResult(
                doc_id=doc_id,
                score=rrf_score,
                rank=rank
            ))
        
        return results
