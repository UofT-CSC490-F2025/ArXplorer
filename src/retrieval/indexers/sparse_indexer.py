"""Sparse indexer using scipy sparse matrices."""

import json
import numpy as np
import scipy.sparse
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from .base import BaseIndexer
from ..encoders import SparseEncoder
from ...data import Document


class SparseIndexer(BaseIndexer):
    """Sparse indexer using scipy for sparse vector storage."""
    
    def __init__(
        self,
        encoder: SparseEncoder,
        output_dir: str,
        chunk_size: int = 10000,  # Process docs in chunks to save memory
        sparse_encoder_batch_size: int = 4  # Batch size for encoder to manage VRAM
    ):
        """
        Args:
            encoder: Sparse encoder instance
            output_dir: Directory to save index files
            chunk_size: Number of documents to process before converting to CSR
            sparse_encoder_batch_size: Batch size for SPLADE encoder (smaller = less VRAM)
        """
        super().__init__(output_dir)
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.sparse_encoder_batch_size = sparse_encoder_batch_size
        
        self.index_file = self.output_dir / "sparse_index.npz"
        self.docmap_file = self.output_dir / "doc_map.json"
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        
        self.vocab_size = encoder.get_dimension()
        self.sparse_matrix = None
        self.doc_map = {}
        self._next_idx = 0
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 16,
        show_progress: bool = True
    ) -> None:
        """Add documents to the index using chunked processing."""
        num_docs = len(documents)
        texts = [doc.text for doc in documents]
        
        # Process documents in chunks to avoid memory issues
        chunks = []
        
        for chunk_start in range(0, num_docs, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, num_docs)
            chunk_texts = texts[chunk_start:chunk_end]
            chunk_docs = documents[chunk_start:chunk_end]
            chunk_size_actual = len(chunk_texts)
            
            # Build LIL matrix for this chunk only
            chunk_matrix = scipy.sparse.lil_matrix(
                (chunk_size_actual, self.vocab_size),
                dtype=np.float32
            )
            
            # Encode chunk
            iterator = range(0, len(chunk_texts), batch_size)
            if show_progress:
                desc = f"Encoding chunk {chunk_start//self.chunk_size + 1}"
                iterator = tqdm(iterator, desc=desc, total=(len(chunk_texts) + batch_size - 1) // batch_size)
            
            row_idx = 0
            for i in iterator:
                batch_texts = chunk_texts[i:i + batch_size]
                # Use configured encoder batch size to manage VRAM
                # SPLADE creates large intermediate tensors
                encoder_batch_size = min(self.sparse_encoder_batch_size, len(batch_texts))
                batch_sparse = self.encoder.encode(batch_texts, batch_size=encoder_batch_size)
                
                # Add to chunk matrix
                for sparse_vec in batch_sparse:
                    indices, values = sparse_vec
                    for idx, val in zip(indices, values):
                        if val > 0:
                            chunk_matrix[row_idx, idx] = val
                    row_idx += 1
            
            # Convert chunk to CSR immediately (memory-efficient)
            chunk_csr = chunk_matrix.tocsr()
            chunks.append(chunk_csr)
            
            # Update doc map for this chunk
            for idx, doc in enumerate(chunk_docs):
                self.doc_map[self._next_idx + chunk_start + idx] = {
                    "id": doc.id,
                    "text": doc.text[:512] if len(doc.text) > 512 else doc.text,
                    "title": doc.title
                }
            
            # Clear chunk_matrix to free memory
            del chunk_matrix
        
        # Combine all CSR chunks
        if show_progress:
            print("Combining chunks into final matrix...")
        
        if self.sparse_matrix is None:
            # First time: just stack the chunks
            self.sparse_matrix = scipy.sparse.vstack(chunks, format='csr')
        else:
            # Append to existing matrix
            all_chunks = [self.sparse_matrix] + chunks
            self.sparse_matrix = scipy.sparse.vstack(all_chunks, format='csr')
        
        self._next_idx += num_docs
    
    def save(self) -> None:
        """Save index to disk."""
        if self.sparse_matrix is None:
            print("Warning: No sparse matrix to save")
            return
        
        # Already in CSR format from chunked building
        print(f"Saving sparse index to {self.index_file}")
        scipy.sparse.save_npz(self.index_file, self.sparse_matrix)
        
        print(f"Saving document map to {self.docmap_file}")
        with open(self.docmap_file, 'w') as f:
            json.dump(self.doc_map, f)
        
        print("Sparse index saved successfully")
    
    def load(self) -> None:
        """Load index from disk."""
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_file}")
        
        print(f"Loading sparse index from {self.index_file}")
        self.sparse_matrix = scipy.sparse.load_npz(self.index_file)
        
        print(f"Loading document map from {self.docmap_file}")
        with open(self.docmap_file, 'r') as f:
            doc_map = json.load(f)
            # Handle both old format (string id) and new format (dict with metadata)
            self.doc_map = {}
            for k, v in doc_map.items():
                if isinstance(v, str):
                    # Old format: just doc_id
                    self.doc_map[int(k)] = {"id": v, "text": "", "title": None}
                else:
                    # New format: dict with id, text, title
                    self.doc_map[int(k)] = v
        
        self._next_idx = len(self.doc_map)
        print(f"Loaded index with {self._next_idx} documents")
    
    def get_num_documents(self) -> int:
        """Return number of indexed documents."""
        return self._next_idx
    
    def save_checkpoint(self) -> None:
        """Save checkpoint for incremental indexing."""
        checkpoint = {
            "num_documents": self._next_idx,
            "vocab_size": self.vocab_size
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
    
    def load_checkpoint(self) -> Optional[dict]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
