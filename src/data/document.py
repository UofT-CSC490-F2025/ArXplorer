"""Document dataclass for representing corpus items."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    """Represents a document in the corpus."""
    
    id: str
    text: str
    title: Optional[str] = None
    metadata: Optional[dict] = None
    published_year: Optional[int] = None  # Publication year extracted from dataset
    citation_count: int = 0  # Citation count from OpenAlex
    year: Optional[int] = None  # Year for Milvus schema (same as published_year or from citations)
    
    def __post_init__(self):
        if not self.id:
            raise ValueError("Document ID cannot be empty")
        if not self.text:
            raise ValueError("Document text cannot be empty")
        
        # Sync year with published_year if not set
        if self.year is None and self.published_year is not None:
            self.year = self.published_year
