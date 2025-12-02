"""Milvus indexer for hybrid dense + sparse retrieval with metadata."""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

from .base import BaseIndexer
from ..encoders import DenseEncoder, SparseEncoder
from ...data import Document


class MilvusIndexer(BaseIndexer):
    """
    Unified indexer storing dense vectors, sparse vectors, and metadata in Milvus.
    
    Replaces:
    - FAISS dense index
    - scipy CSR sparse index
    - doc_map.json metadata storage
    - MongoDB paper collections
    
    All data stored in a single Milvus collection with hybrid search support.
    """
    
    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        output_dir: str,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "arxplorer_papers",
        dense_index_type: str = "IVF_FLAT",
        dense_nlist: int = 1024,
        dense_nprobe: int = 64,
        batch_size: int = 1000,
        use_metadata: bool = False,
        metadata_template: str = None
    ):
        """
        Args:
            dense_encoder: Dense encoder (SPECTER)
            sparse_encoder: Sparse encoder (SPLADE)
            output_dir: Directory for checkpoints (not used for storage)
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of Milvus collection
            dense_index_type: FAISS index type (IVF_FLAT, HNSW, FLAT)
            dense_nlist: Number of clusters for IVF index
            dense_nprobe: Number of clusters to search
            batch_size: Batch size for inserting into Milvus
            use_metadata: Whether to use metadata template for encoding
            metadata_template: Template string (e.g., 'Title: {title}\n\nAbstract: {abstract}')
        """
        super().__init__(output_dir)
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dense_index_type = dense_index_type
        self.dense_nlist = dense_nlist
        self.dense_nprobe = dense_nprobe
        self.batch_size = batch_size
        self.use_metadata = use_metadata
        self.metadata_template = metadata_template
        
        self.collection = None
        self._dense_dim = self.dense_encoder.get_dimension()
        
        # Connect to Milvus
        self._connect()
        
    def _connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            print(f"✓ Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {e}")
    
    def _create_collection(self):
        """Create Milvus collection with schema for hybrid search."""
        # Drop existing collection if it exists
        if utility.has_collection(self.collection_name):
            print(f"⚠️  Collection '{self.collection_name}' already exists. Dropping...")
            utility.drop_collection(self.collection_name)
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="authors", dtype=DataType.JSON),  # List of author names
            FieldSchema(name="categories", dtype=DataType.JSON),  # List of arXiv categories
            FieldSchema(name="year", dtype=DataType.INT32),
            FieldSchema(name="citation_count", dtype=DataType.INT32),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self._dense_dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="ArXplorer papers with hybrid dense + sparse search",
            enable_dynamic_field=True
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default'
        )
        
        print(f"✓ Created collection '{self.collection_name}'")
        print(f"  Fields: id, title, abstract, authors, categories, year, citation_count")
        print(f"  Vectors: dense ({self._dense_dim}D), sparse")
    
    def _create_indexes(self):
        """Create indexes for dense and sparse vectors."""
        print("Building indexes...")
        
        # Dense vector index
        if self.dense_index_type == "IVF_FLAT":
            dense_index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "IP",  # Inner product (cosine similarity for normalized vecs)
                "params": {"nlist": self.dense_nlist}
            }
        elif self.dense_index_type == "HNSW":
            dense_index_params = {
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 16, "efConstruction": 256}
            }
        else:  # FLAT
            dense_index_params = {
                "index_type": "FLAT",
                "metric_type": "IP"
            }
        
        self.collection.create_index(
            field_name="dense_vector",
            index_params=dense_index_params
        )
        print(f"✓ Created {self.dense_index_type} index for dense vectors")
        
        # Sparse vector index
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP"
        }
        
        self.collection.create_index(
            field_name="sparse_vector",
            index_params=sparse_index_params
        )
        print("✓ Created SPARSE_INVERTED_INDEX for sparse vectors")
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 16,
        show_progress: bool = True
    ) -> None:
        """
        Encode documents and add to Milvus collection.
        
        Args:
            documents: List of documents to index
            batch_size: Batch size for encoding (not insertion)
            show_progress: Show progress bar
        """
        if not documents:
            print("No documents to add")
            return
        
        # Create collection if it doesn't exist
        if self.collection is None:
            self._create_collection()
        
        # Format texts using metadata template if enabled
        if self.use_metadata and self.metadata_template:
            texts = []
            for doc in documents:
                formatted_text = self.metadata_template.format(
                    title=doc.title or "",
                    abstract=doc.text,
                    authors=", ".join(doc.authors) if hasattr(doc, 'authors') and doc.authors else "",
                    categories=", ".join(doc.categories) if hasattr(doc, 'categories') and doc.categories else "",
                    year=doc.published_year if doc.published_year else "Unknown"
                )
                texts.append(formatted_text)
        else:
            texts = [doc.text for doc in documents]
        
        # Encode dense vectors
        print(f"Encoding {len(documents)} documents (dense)...")
        if self.use_metadata and self.metadata_template:
            print(f"  Using metadata template: {self.metadata_template[:80]}...")
        dense_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Dense encoding", total=(len(texts) + batch_size - 1) // batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_embs = self.dense_encoder.encode(batch_texts, batch_size=len(batch_texts))
            dense_embeddings.append(batch_embs)
        
        dense_embeddings = np.vstack(dense_embeddings)
        
        # Encode sparse vectors
        print(f"Encoding {len(documents)} documents (sparse)...")
        sparse_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Sparse encoding", total=(len(texts) + batch_size - 1) // batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_sparse = self.sparse_encoder.encode(batch_texts, batch_size=4)  # Smaller batch for SPLADE
            sparse_embeddings.extend(batch_sparse)
        
        # Insert into Milvus in batches
        print(f"Inserting into Milvus collection '{self.collection_name}'...")
        
        for i in tqdm(range(0, len(documents), self.batch_size), desc="Inserting batches"):
            batch_end = min(i + self.batch_size, len(documents))
            batch_docs = documents[i:batch_end]
            batch_dense = dense_embeddings[i:batch_end]
            batch_sparse = sparse_embeddings[i:batch_end]
            
            # Prepare data for insertion - list of dicts format
            entities = []
            for j, doc in enumerate(batch_docs):
                sparse_dict = {int(idx): float(val) for idx, val in zip(batch_sparse[j][0], batch_sparse[j][1])}
                
                # Get year from doc (use year field if set, otherwise published_year, default 0)
                year_val = 0
                if hasattr(doc, 'year') and doc.year:
                    year_val = doc.year
                elif doc.published_year:
                    year_val = doc.published_year
                
                # Get citation_count from doc (default 0)
                citation_val = getattr(doc, 'citation_count', 0) if hasattr(doc, 'citation_count') else 0
                
                entity = {
                    "id": doc.id,
                    "title": doc.title or "",
                    "abstract": doc.text[:8192],  # Truncate to max length
                    "authors": doc.authors if hasattr(doc, 'authors') else [],
                    "categories": doc.categories if hasattr(doc, 'categories') else [],
                    "year": year_val,
                    "citation_count": citation_val,
                    "dense_vector": batch_dense[j].tolist(),
                    "sparse_vector": sparse_dict,
                }
                entities.append(entity)
            
            self.collection.insert(entities)
        
        print(f"✓ Inserted {len(documents)} documents into Milvus")
    
    def _convert_sparse_to_milvus(self, sparse_vectors):
        """Convert SPLADE sparse format to Milvus sparse format."""
        milvus_sparse = []
        for indices, values in sparse_vectors:
            # Milvus expects dict format: {index: value}
            sparse_dict = {int(idx): float(val) for idx, val in zip(indices, values)}
            milvus_sparse.append(sparse_dict)
        return milvus_sparse
    
    def save(self) -> None:
        """Build indexes and flush data to disk."""
        if self.collection is None:
            print("No collection to save")
            return
        
        print("Flushing data to Milvus...")
        self.collection.flush()
        
        # Create indexes
        self._create_indexes()
        
        # Load collection into memory for searching
        self.collection.load()
        
        print(f"✓ Collection '{self.collection_name}' saved and loaded")
        print(f"  Total documents: {self.collection.num_entities}")
    
    def load(self) -> None:
        """Load existing collection."""
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' does not exist")
        
        self.collection = Collection(self.collection_name)
        self.collection.load()
        
        print(f"✓ Loaded collection '{self.collection_name}'")
        print(f"  Total documents: {self.collection.num_entities}")
    
    def get_num_documents(self) -> int:
        """Return number of indexed documents."""
        if self.collection is None:
            return 0
        return self.collection.num_entities
    
    def update_citation_counts(self, citation_data: Dict[str, int]):
        """
        Update citation counts for papers (incremental update).
        
        Args:
            citation_data: Dict mapping arxiv_id -> citation_count
        """
        print(f"Updating citation counts for {len(citation_data)} papers...")
        
        # Milvus doesn't support in-place updates, need to upsert
        for arxiv_id, citation_count in tqdm(citation_data.items(), desc="Updating citations"):
            # Delete old entry
            self.collection.delete(expr=f'id == "{arxiv_id}"')
            
            # Would need to re-fetch document data and re-insert
            # For now, use a simpler approach: track citations externally
            # and join during search
        
        print("⚠️  Note: Full upsert not implemented. Use external citation file for now.")
