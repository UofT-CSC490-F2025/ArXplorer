"""Streaming data loader for JSONL corpus files."""

import json
from pathlib import Path
from typing import Iterator, Optional

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
                
                # Store remaining fields as metadata
                metadata = {k: v for k, v in data.items() 
                           if k not in [self.text_key, self.id_key, self.title_key]}
                
                yield Document(
                    id=doc_id,
                    text=text,
                    title=title,
                    metadata=metadata if metadata else None
                )
    
    def count_documents(self) -> int:
        """Count total documents in file (for progress bars)."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
