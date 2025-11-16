"""Base class for query rewriters."""

from abc import ABC, abstractmethod
from typing import List


class BaseQueryRewriter(ABC):
    """Abstract base class for query rewriting."""
    
    @abstractmethod
    def rewrite(self, query: str, num_rewrites: int = 1) -> List[str]:
        """
        Rewrite a search query to improve retrieval.
        
        Args:
            query: Original user query
            num_rewrites: Number of rewritten queries to generate
            
        Returns:
            List of rewritten/expanded queries
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass
