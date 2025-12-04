"""Streaming data loader for JSONL corpus files."""

import json
from pathlib import Path
from typing import Iterator, Optional
from datetime import datetime

from .document import Document


class StreamingJSONLLoader:
    """Loads documents from JSONL file with streaming to handle large corpora."""
    
    def __init__(
        self,
        filepath: str,
        text_key: str = "abstract",
        id_key: str = "id",
        title_key: Optional[str] = "title",
        skip_empty: bool = True,
        use_metadata: bool = False,
        categories_key: Optional[str] = "categories",
        authors_key: Optional[str] = "authors",
        metadata_template: Optional[str] = None
    ):
        """
        Args:
            filepath: Path to JSONL file
            text_key: JSON key for document text content
            id_key: JSON key for document ID
            title_key: JSON key for document title (optional)
            skip_empty: Skip documents with empty text
            use_metadata: If True, construct text using metadata_template
            categories_key: JSON key for categories list
            authors_key: JSON key for authors list
            metadata_template: Template string for combining metadata (e.g., "Title: {title}\\n\\nAbstract: {abstract}")
        """
        self.filepath = Path(filepath)
        self.text_key = text_key
        self.id_key = id_key
        self.title_key = title_key
        self.skip_empty = skip_empty
        self.use_metadata = use_metadata
        self.categories_key = categories_key
        self.authors_key = authors_key
        self.metadata_template = metadata_template
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
    
    def _parse_year_from_date(self, date_string: str) -> Optional[int]:
        """
        Extract year from date string.
        
        Handles formats like: "Sun, 1 Apr 2007 13:06:50 GMT"
        
        Args:
            date_string: Date string to parse
            
        Returns:
            Year as integer, or None if parsing fails
        """
        if not date_string:
            return None
        
        try:
            # Try RFC 2822 format (e.g., "Sun, 1 Apr 2007 13:06:50 GMT")
            dt = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S %Z")
            return dt.year
        except (ValueError, AttributeError):
            # Try to extract year with simple parsing (YYYY in string)
            import re
            match = re.search(r'\b(19|20)\d{2}\b', date_string)
            if match:
                return int(match.group(0))
        
        return None
    
    def load(self) -> Iterator[Document]:
        """Stream documents one at a time from JSONL file."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue
                
                text = data.get(self.text_key, "").strip()
                doc_id = data.get(self.id_key)
                
                if not doc_id:
                    doc_id = f"auto_{line_num}"
                else:
                    doc_id = str(doc_id)
                
                # Get title for metadata or storage
                title = data.get(self.title_key) if self.title_key else None
                
                # If use_metadata is enabled, construct enhanced text
                if self.use_metadata and self.metadata_template:
                    # Gather metadata fields
                    template_data = {
                        'abstract': text,
                        'title': title or '',
                    }
                    
                    # Add categories (join if list)
                    if self.categories_key:
                        categories = data.get(self.categories_key, [])
                        if isinstance(categories, list):
                            template_data['categories'] = ', '.join(categories)
                        else:
                            template_data['categories'] = str(categories)
                    else:
                        template_data['categories'] = ''
                    
                    # Add authors (join if list)
                    if self.authors_key:
                        authors = data.get(self.authors_key, [])
                        if isinstance(authors, list):
                            template_data['authors'] = ', '.join(authors)
                        else:
                            template_data['authors'] = str(authors)
                    else:
                        template_data['authors'] = ''
                    
                    # Format using template
                    try:
                        text = self.metadata_template.format(**template_data)
                    except KeyError as e:
                        print(f"Warning: Missing key {e} in template at line {line_num}, using original text")
                
                if self.skip_empty and not text:
                    continue
                
                # Extract publication year from published_date field
                published_year = None
                if 'published_date' in data:
                    published_year = self._parse_year_from_date(data['published_date'])
                
                # Store remaining fields as metadata
                metadata = {k: v for k, v in data.items() 
                           if k not in [self.text_key, self.id_key, self.title_key]}
                
                yield Document(
                    id=doc_id,
                    text=text,
                    title=title,
                    metadata=metadata if metadata else None,
                    published_year=published_year
                )
    
    def count_documents(self) -> int:
        """Count total documents in file (for progress bars)."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
