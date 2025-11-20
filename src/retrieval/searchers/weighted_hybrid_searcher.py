"""Weighted hybrid searcher with component score tracking."""

from typing import List, Dict
from collections import defaultdict

from .base import BaseSearcher, SearchResult
from .dense_searcher import DenseSearcher
from .sparse_searcher import SparseSearcher


class WeightedHybridSearcher(BaseSearcher):
    """
    Hybrid searcher using weighted fusion of dense, sparse, and cross-encoder scores.
    
    Fusion strategy:
    1. Dense retrieval: For each query rewrite, retrieve docs → Store max similarity per doc
    2. Sparse retrieval: For each query rewrite, retrieve docs → Store RRF score per doc
    3. RRF fusion: Apply RRF over ALL (num_queries × 2 methods) results based on ranks
    4. Cross-encoder: Score top-K from RRF fusion
    5. Final score: Normalize all scores → w_dense * dense + w_sparse * sparse + w_cross * cross
    
    Key: Dense/sparse scores are metadata; RRF fusion determines cross-encoder candidates.
    """
    
    def __init__(
        self,
        dense_searcher: DenseSearcher,
        sparse_searcher: SparseSearcher,
        rrf_k: int = 60,
        dense_weight: float = 0.2,
        sparse_weight: float = 0.2,
        cross_encoder_weight: float = 0.6,
        normalize_scores: bool = True
    ):
        """
        Args:
            dense_searcher: Dense searcher instance
            sparse_searcher: Sparse searcher instance
            rrf_k: RRF constant for ranking fusion
            dense_weight: Weight for dense scores in final fusion
            sparse_weight: Weight for sparse scores in final fusion
            cross_encoder_weight: Weight for cross-encoder scores in final fusion
            normalize_scores: Whether to min-max normalize scores
        """
        self.dense_searcher = dense_searcher
        self.sparse_searcher = sparse_searcher
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.cross_encoder_weight = cross_encoder_weight
        self.should_normalize_scores = normalize_scores  # Renamed to avoid conflict
        
        # Validate weights
        weight_sum = dense_weight + sparse_weight + cross_encoder_weight
        if abs(weight_sum - 1.0) > 1e-6:
            print(f"Warning: Weights sum to {weight_sum:.4f}, not 1.0. Normalizing...")
            self.dense_weight /= weight_sum
            self.sparse_weight /= weight_sum
            self.cross_encoder_weight /= weight_sum
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Min-max normalize scores to [0, 1] range.
        
        Args:
            scores: Dict mapping doc_id -> score
            
        Returns:
            Dict with normalized scores
        """
        if not scores:
            return scores
        
        min_score = min(scores.values())
        max_score = max(scores.values())
        
        if max_score == min_score:
            # All scores are the same
            return {doc_id: 1.0 for doc_id in scores}
        
        normalized = {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in scores.items()
        }
        
        return normalized
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        retrieval_k: int = 100
    ) -> List[SearchResult]:
        """
        Search using single query (no query rewriting handled here).
        
        This is the base search method. For query rewriting, use search_multi_query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            retrieval_k: Number of candidates to retrieve from each method
            
        Returns:
            List of SearchResult with component scores
        """
        return self.search_multi_query([query], top_k, retrieval_k)
    
    def search_multi_query(
        self,
        queries: List[str],
        top_k: int = 10,
        retrieval_k: int = 100
    ) -> List[SearchResult]:
        """
        Search using multiple query variations with weighted fusion.
        
        Flow:
        1. Retrieve from dense/sparse for each query (ONCE per query per method)
        2. Aggregate dense scores (max similarity) and sparse scores (RRF) as metadata
        3. RRF fusion over ALL (num_queries × 2 methods) result lists based on ranks
        4. Return top-k from RRF fusion with component scores attached
        
        Args:
            queries: List of query variations (original + rewrites)
            top_k: Number of results to return
            retrieval_k: Number of candidates to retrieve per method per query
            
        Returns:
            List of SearchResult with dense, sparse scores populated (for final fusion)
        """
        # Step 1: Retrieve ALL results ONCE (store for reuse)
        all_dense_results = []
        all_sparse_results = []
        
        for query in queries:
            dense_results = self.dense_searcher.search(query, top_k=retrieval_k)
            sparse_results = self.sparse_searcher.search(query, top_k=retrieval_k)
            all_dense_results.append(dense_results)
            all_sparse_results.append(sparse_results)
        
        # Step 2: Aggregate dense scores (max similarity across queries)
        dense_scores = {}
        for results in all_dense_results:
            for result in results:
                if result.doc_id not in dense_scores:
                    dense_scores[result.doc_id] = result.score
                else:
                    dense_scores[result.doc_id] = max(dense_scores[result.doc_id], result.score)
        
        # Step 3: Aggregate sparse scores (RRF across queries)
        sparse_scores = defaultdict(float)
        for results in all_sparse_results:
            for result in results:
                sparse_scores[result.doc_id] += 1.0 / (self.rrf_k + result.rank)
        sparse_scores = dict(sparse_scores)
        
        # Step 4: Normalize scores to [0, 1] for final weighted fusion
        if self.should_normalize_scores:
            dense_scores = self.normalize_scores(dense_scores)
            sparse_scores = self.normalize_scores(sparse_scores)
        
        # Step 5: RRF fusion over ALL retrieval results (num_queries × 2 methods)
        # Reuse the same results we already retrieved
        rrf_scores = defaultdict(float)
        for results in all_dense_results:
            for result in results:
                rrf_scores[result.doc_id] += 1.0 / (self.rrf_k + result.rank)
        for results in all_sparse_results:
            for result in results:
                rrf_scores[result.doc_id] += 1.0 / (self.rrf_k + result.rank)
        
        # Step 6: Create SearchResult objects with component scores
        # Sort by RRF score to get top-k candidates for cross-encoder
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, rrf_score) in enumerate(sorted_docs[:top_k], 1):
            results.append(SearchResult(
                doc_id=doc_id,
                score=rrf_score,  # Initial score is RRF
                rank=rank,
                dense_score=dense_scores.get(doc_id, 0.0),  # Metadata for final fusion
                sparse_score=sparse_scores.get(doc_id, 0.0),  # Metadata for final fusion
                cross_encoder_score=None  # Will be filled by reranker
            ))
        
        return results
    
    def apply_weighted_fusion(
        self,
        results: List[SearchResult],
        normalize: bool = True
    ) -> List[SearchResult]:
        """
        Apply weighted fusion of dense, sparse, and cross-encoder scores.
        
        This should be called AFTER cross-encoder reranking has populated
        the cross_encoder_score field.
        
        Args:
            results: List of SearchResult with all component scores
            normalize: Whether to normalize cross-encoder scores
            
        Returns:
            List of SearchResult sorted by final weighted score
        """
        # Extract cross-encoder scores
        cross_scores = {
            r.doc_id: r.cross_encoder_score
            for r in results
            if r.cross_encoder_score is not None
        }
        
        # Normalize cross-encoder scores if requested
        if normalize and cross_scores:
            cross_scores = self.normalize_scores(cross_scores)
        
        # Compute final weighted scores
        for result in results:
            dense = result.dense_score if result.dense_score is not None else 0.0
            sparse = result.sparse_score if result.sparse_score is not None else 0.0
            cross = cross_scores.get(result.doc_id, 0.0)
            
            # Update cross-encoder score with normalized value
            result.cross_encoder_score = cross
            
            # Compute final weighted score
            result.score = (
                self.dense_weight * dense +
                self.sparse_weight * sparse +
                self.cross_encoder_weight * cross
            )
        
        # Sort by final score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank
        
        return results
