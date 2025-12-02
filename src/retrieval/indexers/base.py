"""Base abstract class for indexers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from ...data import Document


class BaseIndexer(ABC):
    """Abstract base class for building and managing document indexes."""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Directory to save index files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 16,
        show_progress: bool = True
    ) -> None:
        """
        Add documents to the index.
        
        Args:
            documents: List of documents to index
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        """
        pass
    
    @abstractmethod
    def save(self) -> None:
        """Save the index to disk."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load the index from disk."""
        pass
    
    @abstractmethod
    def get_num_documents(self) -> int:
        """Return the number of indexed documents."""
        pass
