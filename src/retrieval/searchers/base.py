"""Base abstract class for searchers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class SearchResult:
    """Represents a single search result."""
    doc_id: str
    score: float
    rank: int = 0


class BaseSearcher(ABC):
    """Abstract base class for search implementations."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for documents matching the query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects sorted by score (descending)
        """
        pass
