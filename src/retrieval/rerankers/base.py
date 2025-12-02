"""Base abstract class for rerankers."""

from abc import ABC, abstractmethod
from typing import List, Dict

from ..searchers.base import SearchResult


class BaseReranker(ABC):
    """Abstract base class for reranking search results."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Rerank search results using a more sophisticated model.
        
        Args:
            query: Query text
            results: Initial search results to rerank
            top_k: Number of top results to return after reranking
            
        Returns:
            List of reranked SearchResult objects
        """
        pass
