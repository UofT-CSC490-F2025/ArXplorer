"""Cross-encoder reranker for refining search results."""

import torch
from typing import List, Dict
from sentence_transformers import CrossEncoder

from .base import BaseReranker
from ..searchers.base import SearchResult


class CrossEncoderReranker(BaseReranker):
    """Reranker using cross-encoder models for precise relevance scoring."""
    
    def __init__(
        self,
        doc_texts: Dict[str, str],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None,
        max_length: int = 512,
        batch_size: int = 32
    ):
        """
        Args:
            doc_texts: Mapping of doc_id → document text
            model_name: HuggingFace cross-encoder model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum token length for (query, doc) pairs
            batch_size: Batch size for cross-encoder inference
        """
        self.doc_texts = doc_texts
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading cross-encoder: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Rerank results using cross-encoder.
        
        Preserves dense_score and sparse_score from original results,
        and populates cross_encoder_score.
        
        Args:
            query: Query text
            results: Initial search results to rerank
            top_k: Number of top results to return after reranking
            
        Returns:
            List of reranked SearchResult objects with cross_encoder_score populated
        """
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = []
        valid_results = []
        
        for result in results:
            doc_text = self.doc_texts.get(result.doc_id)
            if doc_text:
                pairs.append([query, doc_text])
                valid_results.append(result)
            else:
                # Keep results without text at the end with low scores
                print(f"Warning: No text found for doc_id {result.doc_id}, skipping reranking")
        
        if not pairs:
            print("Warning: No valid document texts found for reranking")
            return results[:top_k]
        
        # Score all pairs with cross-encoder
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Update results with cross-encoder scores (preserve ALL component scores)
        reranked_results = []
        for idx, (result, score) in enumerate(zip(valid_results, scores)):
            new_result = SearchResult(
                doc_id=result.doc_id,
                score=float(score),  # Temporarily set to cross-encoder score
                rank=0,  # Will be set after sorting
                dense_score=result.dense_score,  # Preserve
                sparse_score=result.sparse_score,  # Preserve
                cross_encoder_score=float(score),  # Populate
                citation_score=result.citation_score,  # Preserve citation metadata
                citation_count=result.citation_count,  # Preserve citation count
                publication_year=result.publication_year,  # Preserve publication year
                year_score=result.year_score  # Preserve year score
            )
            # Copy metadata attributes
            if hasattr(result, 'title'):
                new_result.title = result.title
            if hasattr(result, 'abstract'):
                new_result.abstract = result.abstract
            if hasattr(result, 'authors'):
                new_result.authors = result.authors
            if hasattr(result, 'categories'):
                new_result.categories = result.categories
            
            reranked_results.append(new_result)
        
        # Sort by cross-encoder score (descending)
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(reranked_results[:top_k], 1):
            result.rank = rank
        
        return reranked_results[:top_k]
