"""Jina reranker for refining search results."""

import torch
from typing import List, Dict
from transformers import AutoModel

from .base import BaseReranker
from ..searchers.base import SearchResult


class JinaReranker(BaseReranker):
    """Reranker using Jina AI's jina-reranker-v3 model (0.6B listwise reranker)."""
    
    def __init__(
        self,
        doc_texts: Dict[str, str],
        model_name: str = "jinaai/jina-reranker-v3",
        device: str = None,
        batch_size: int = 32
    ):
        """
        Args:
            doc_texts: Mapping of doc_id → document text
            model_name: HuggingFace Jina reranker model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for reranker inference (max 64 docs per batch)
        """
        self.doc_texts = doc_texts
        self.model_name = model_name
        self.batch_size = min(batch_size, 64)  # Jina processes up to 64 docs at once
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading Jina reranker: {model_name} on {device}")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Move to device if CUDA
        if device == "cuda":
            self.model = self.model.to(device)
        
        self.model.eval()
        
        # Report model size
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"  Model loaded: {param_count / 1e9:.2f}B parameters")
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Rerank results using Jina reranker.
        
        Preserves dense_score and sparse_score from original results,
        and populates cross_encoder_score with the reranker score.
        
        Args:
            query: Query text
            results: Initial search results to rerank
            top_k: Number of top results to return after reranking
            
        Returns:
            List of reranked SearchResult objects with cross_encoder_score populated
        """
        if not results:
            return []
        
        # Prepare documents list
        documents = []
        valid_results = []
        
        for result in results:
            doc_text = self.doc_texts.get(result.doc_id)
            if doc_text:
                documents.append(doc_text)
                valid_results.append(result)
            else:
                print(f"Warning: No text found for doc_id {result.doc_id}, skipping reranking")
        
        if not documents:
            print("Warning: No valid document texts found for reranking")
            return results[:top_k]
        
        # Rerank using Jina's API (processes all docs at once, up to 64)
        # For larger batches, we need to split
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(documents), self.batch_size):
                batch_docs = documents[i:i + self.batch_size]
                
                # Jina reranker returns list of dicts with 'relevance_score', 'document', 'index'
                batch_results = self.model.rerank(query, batch_docs)
                
                # Extract scores (results are already sorted by relevance)
                batch_scores = [r['relevance_score'] for r in batch_results]
                all_scores.extend(batch_scores)
        
        # Update results with reranker scores (preserve ALL component scores)
        reranked_results = []
        for idx, (result, score) in enumerate(zip(valid_results, all_scores)):
            new_result = SearchResult(
                doc_id=result.doc_id,
                score=float(score),  # Temporarily set to reranker score
                rank=0,  # Will be set after sorting
                dense_score=result.dense_score,  # Preserve
                sparse_score=result.sparse_score,  # Preserve
                cross_encoder_score=float(score),  # Populate with reranker score
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
        
        # Sort by reranker score (descending)
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(reranked_results[:top_k], 1):
            result.rank = rank
        
        return reranked_results[:top_k]
