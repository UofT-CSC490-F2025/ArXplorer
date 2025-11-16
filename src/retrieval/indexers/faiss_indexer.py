"""Dense indexer using FAISS."""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from .base import BaseIndexer
from ..encoders import DenseEncoder
from ...data import Document


class DenseIndexer(BaseIndexer):
    """Dense indexer using FAISS for fast similarity search."""
    
    def __init__(
        self,
        encoder: DenseEncoder,
        output_dir: str,
        use_gpu: bool = False
    ):
        """
        Args:
            encoder: Dense encoder instance
            output_dir: Directory to save index files
            use_gpu: Whether to use GPU for FAISS (if available)
        """
        super().__init__(output_dir)
        self.encoder = encoder
        self.use_gpu = use_gpu
        
        self.embeddings_file = self.output_dir / "embeddings.npy"
        self.index_file = self.output_dir / "index.faiss"
        self.docmap_file = self.output_dir / "doc_map.json"
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        
        self.embeddings = []
        self.doc_map = {}
        self.index = None
        self._next_idx = 0
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 16,
        show_progress: bool = True
    ) -> None:
        """Add documents to the index."""
        texts = [doc.text for doc in documents]
        
        # Encode in batches
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding documents", total=(len(texts) + batch_size - 1) // batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_embs = self.encoder.encode(batch_texts, batch_size=len(batch_texts))
            all_embeddings.append(batch_embs)
        
        if not all_embeddings:
            return
        
        embeddings_array = np.vstack(all_embeddings)
        
        # Add to doc map (store text for cross-encoder reranking)
        for idx, doc in enumerate(documents):
            self.doc_map[self._next_idx + idx] = {
                "id": doc.id,
                "text": doc.text[:512] if len(doc.text) > 512 else doc.text,  # Truncate for memory
                "title": doc.title
            }
        
        self.embeddings.append(embeddings_array)
        self._next_idx += len(documents)
    
    def save(self) -> None:
        """Save index to disk."""
        if not self.embeddings:
            print("Warning: No embeddings to save")
            return
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(self.embeddings)
        
        print(f"Saving {len(all_embeddings)} embeddings to {self.embeddings_file}")
        np.save(self.embeddings_file, all_embeddings)
        
        # Build FAISS index
        print("Building FAISS index...")
        dim = all_embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (embeddings are normalized)
        index = faiss.IndexFlatIP(dim)
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("Moving FAISS index to GPU")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        index.add(all_embeddings)
        
        # Save to disk (move to CPU if on GPU)
        if self.use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_gpu_to_cpu(index)
        
        print(f"Saving FAISS index to {self.index_file}")
        faiss.write_index(index, str(self.index_file))
        
        # Save doc map
        print(f"Saving document map to {self.docmap_file}")
        with open(self.docmap_file, 'w') as f:
            json.dump(self.doc_map, f)
        
        print("Dense index saved successfully")
    
    def load(self) -> None:
        """Load index from disk."""
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_file}")
        
        print(f"Loading FAISS index from {self.index_file}")
        self.index = faiss.read_index(str(self.index_file))
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("Moving FAISS index to GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
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
            "embedding_dim": self.encoder.get_dimension()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
    
    def load_checkpoint(self) -> Optional[dict]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
