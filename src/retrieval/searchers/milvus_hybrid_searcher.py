"""Milvus hybrid searcher with RRF fusion."""

import json
from typing import List, Dict, Optional
import numpy as np

from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker

from .base import BaseSearcher, SearchResult
from ..encoders import DenseEncoder, SparseEncoder


class MilvusHybridSearcher(BaseSearcher):
    """
    Hybrid searcher using Milvus built-in RRF fusion.
    
    Replaces legacy architecture:
    - FAISS dense search + scipy sparse search
    - Custom RRF fusion
    - doc_map.json metadata lookups
    
    Milvus handles:
    - Dense vector search (SPECTER)
    - Sparse vector search (SPLADE)
    - RRF fusion
    - Metadata retrieval (title, abstract, year, etc.)
    
    You still handle (post-search):
    - Cross-encoder reranking
    - Citation scoring
    - Year scoring (canonical queries)
    - Weighted fusion of all components
    """
    
    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "arxplorer_papers",
        rrf_k: int = 60
    ):
        """
        Args:
            dense_encoder: Dense encoder for query encoding
            sparse_encoder: Sparse encoder for query encoding
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of collection to search
            rrf_k: RRF constant (k parameter)
        """
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.rrf_k = rrf_k
        
        self.collection = None
        self._connect()
    
    def _connect(self):
        """Connect to Milvus and load collection."""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            
            self.collection = Collection(self.collection_name)
            self.collection.load()
            
            print(f"✓ Connected to Milvus collection '{self.collection_name}'")
            print(f"  Documents: {self.collection.num_entities}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {e}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        retrieval_k: int = 1000,
        filter_expr: Optional[str] = None,
        query_variants: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search using hybrid dense + sparse with RRF fusion.
        
        Args:
            query: Query text (primary query)
            top_k: Number of final results to return
            retrieval_k: Number of candidates to retrieve for reranking
            filter_expr: Optional Milvus filter expression (e.g., "year >= 2020 and citation_count > 100")
            query_variants: Optional list of rewritten queries to include in hybrid search
            
        Returns:
            List of SearchResult with dense_score, sparse_score, and metadata
        """
        # Build list of all queries to encode (original + variants)
        all_queries = [query]
        if query_variants:
            all_queries.extend(query_variants)
        
        # Encode all queries
        dense_vecs = self.dense_encoder.encode(all_queries)
        sparse_vecs = self.sparse_encoder.encode(all_queries, batch_size=1)
        
        # Convert sparse to Milvus format
        sparse_dicts = []
        for sparse_vec in sparse_vecs:
            sparse_indices, sparse_values = sparse_vec
            sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_indices, sparse_values)}
            sparse_dicts.append(sparse_dict)
        
        # Create search requests for hybrid search with optional filtering
        # For each query, create both dense and sparse requests
        search_requests = []
        
        for dense_vec in dense_vecs:
            dense_request = AnnSearchRequest(
                data=[dense_vec.tolist()],
                anns_field="dense_vector",
                param={"metric_type": "IP", "params": {"nprobe": 64}},
                limit=retrieval_k,
                expr=filter_expr
            )
            search_requests.append(dense_request)
        
        for sparse_dict in sparse_dicts:
            sparse_request = AnnSearchRequest(
                data=[sparse_dict],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=retrieval_k,
                expr=filter_expr
            )
            search_requests.append(sparse_request)
        
        # Perform hybrid search with RRF fusion across ALL queries and modalities
        results = self.collection.hybrid_search(
            reqs=search_requests,
            rerank=RRFRanker(k=self.rrf_k),
            limit=top_k,
            output_fields=["id", "title", "abstract", "authors", "categories", "year", "citation_count"]
        )
        
        # Convert Milvus results to SearchResult format
        search_results = []
        for rank, hit in enumerate(results[0], 1):  # results[0] because we only sent one query
            doc_id = hit.entity.get("id")
            
            result = SearchResult(
                doc_id=doc_id,
                score=hit.score,  # RRF score from Milvus
                rank=rank,
                dense_score=None,  # No longer tracking component scores
                sparse_score=None,
                citation_count=hit.entity.get("citation_count", 0),
                publication_year=hit.entity.get("year")
            )
            # Store additional metadata as attributes (not in __init__)
            result.title = hit.entity.get("title", "")
            result.abstract = hit.entity.get("abstract", "")
            result.authors = hit.entity.get("authors", [])
            result.categories = hit.entity.get("categories", [])
            
            search_results.append(result)
        
        return search_results
    
    def search_with_scores(
        self,
        query: str,
        top_k: int = 10,
        retrieval_k: int = 1000,
        filter_expr: Optional[str] = None
    ) -> tuple[List[SearchResult], Dict[str, Dict[str, float]]]:
        """
        Search and return results with separate component scores.
        
        Args:
            query: Query text
            top_k: Number of results
            retrieval_k: Retrieval pool size
            filter_expr: Optional Milvus filter expression
        
        Returns:
            tuple: (search_results, component_scores)
            component_scores: Dict mapping doc_id -> {dense_score, sparse_score}
        """
        # For now, Milvus doesn't expose individual dense/sparse scores easily
        # after RRF fusion. We return the RRF score as the main score.
        results = self.search(query, top_k, retrieval_k, filter_expr)
        
        # Placeholder component scores (would need separate searches to get these)
        component_scores = {
            r.doc_id: {
                "dense_score": r.dense_score,
                "sparse_score": r.sparse_score,
                "rrf_score": r.score
            }
            for r in results
        }
        
        return results, component_scores
    
    def get_doc_metadata(self, doc_ids: List[str]) -> List[Dict]:
        """
        Fetch metadata for specific documents.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            List of metadata dicts
        """
        # Query by IDs
        expr = f'id in {json.dumps(doc_ids)}'
        results = self.collection.query(
            expr=expr,
            output_fields=["id", "title", "abstract", "authors", "categories", "year", "citation_count"]
        )
        
        return results
