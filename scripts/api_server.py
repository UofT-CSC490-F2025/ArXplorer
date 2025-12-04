"""FastAPI server for ArXplorer query processing.

Handles:
- Query embeddings (SPECTER2 + SPLADE)
- Milvus hybrid search
- LLM query analysis (Bedrock)
- Intent boosting
- Title/author matching
- Optional reranking (Jina)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.config import Config
from src.retrieval.encoders import DenseEncoder, SparseEncoder
from src.retrieval.searchers import MilvusHybridSearcher
from src.retrieval.query_rewriting import LLMQueryRewriter
from src.retrieval.rerankers.intent_booster import IntentBooster
from src.retrieval.rerankers.title_author_matcher import TitleAuthorMatcher
from src.retrieval.rerankers.jina_reranker import JinaReranker
from src.retrieval.searchers.base import SearchResult

# Load configuration
config = Config.from_yaml("config.yaml")

# Override with environment variables if present
if os.getenv("MILVUS_HOST"):
    config.milvus.host = os.getenv("MILVUS_HOST")
if os.getenv("MILVUS_PORT"):
    config.milvus.port = int(os.getenv("MILVUS_PORT"))
if os.getenv("BEDROCK_REGION"):
    config.query_rewriting.bedrock_region = os.getenv("BEDROCK_REGION")
if os.getenv("BEDROCK_MODEL_ID"):
    config.query_rewriting.bedrock_model_id = os.getenv("BEDROCK_MODEL_ID")

# Initialize components
print("Initializing ArXplorer Query API...")
print(f"Milvus: {config.milvus.host}:{config.milvus.port}")
print(f"Bedrock: {config.query_rewriting.bedrock_region}/{config.query_rewriting.bedrock_model_id}")

# Dense encoder
print(f"Loading dense encoder: {config.encoder.dense_model}")
dense_encoder = DenseEncoder(
    model_name=config.encoder.dense_model,
    device=config.encoder.device,
    normalize=config.encoder.normalize_dense,
    use_specter2=config.encoder.use_specter2,
    specter2_base_adapter=config.encoder.specter2_base_adapter,
    specter2_query_adapter=config.encoder.specter2_query_adapter
)

# Switch to query adapter for searching
if dense_encoder.use_specter2:
    dense_encoder.use_query_adapter()

# Sparse encoder
print(f"Loading sparse encoder: {config.encoder.sparse_model}")
sparse_encoder = SparseEncoder(
    model_name=config.encoder.sparse_model,
    device=config.encoder.device,
    max_length=config.encoder.max_length
)

# Milvus searcher
print(f"Connecting to Milvus: {config.milvus.host}:{config.milvus.port}")
searcher = MilvusHybridSearcher(
    dense_encoder=dense_encoder,
    sparse_encoder=sparse_encoder,
    host=config.milvus.host,
    port=config.milvus.port,
    collection_name=config.milvus.collection_name,
    rrf_k=config.search.rrf_k
)

# Query rewriter (Bedrock)
query_rewriter = None
if config.query_rewriting.enabled:
    print("Initializing LLM query rewriter (Bedrock)")
    query_rewriter = LLMQueryRewriter(
        model_name=config.query_rewriting.model,
        device=config.query_rewriting.device,
        max_length=config.query_rewriting.max_length,
        temperature=config.query_rewriting.temperature,
        num_rewrites=config.query_rewriting.num_rewrites,
        use_vllm=config.query_rewriting.use_vllm,
        vllm_endpoint=config.query_rewriting.vllm_endpoint,
        vllm_timeout=config.query_rewriting.vllm_timeout,
        use_bedrock=getattr(config.query_rewriting, 'use_bedrock', False),
        bedrock_model_id=getattr(config.query_rewriting, 'bedrock_model_id', 'mistral.mistral-7b-instruct-v0:2'),
        bedrock_region=getattr(config.query_rewriting, 'bedrock_region', 'ca-central-1'),
        bedrock_max_tokens=getattr(config.query_rewriting, 'bedrock_max_tokens', 512)
    )

# Intent booster
intent_booster = None
if hasattr(config, 'intent_boosting') and config.intent_boosting.enabled:
    print("Initializing intent-based boosting")
    intent_booster = IntentBooster(
        citation_weights=config.intent_boosting.citation_weights,
        date_weights=config.intent_boosting.date_weights
    )

# Title/author matcher
title_author_matcher = None
if hasattr(config, 'title_author_matching') and config.title_author_matching.enabled:
    print("Initializing title/author fuzzy matching")
    title_author_matcher = TitleAuthorMatcher(
        title_threshold=config.title_author_matching.title_threshold,
        author_threshold=config.title_author_matching.author_threshold,
        title_boost_weight=config.title_author_matching.title_boost_weight,
        author_boost_weight=config.title_author_matching.author_boost_weight
    )

# Reranker (optional, disabled by default)
reranker = None
if config.reranker.enabled:
    print(f"Loading reranker: {config.reranker.model}")
    if config.reranker.type.lower() == "jina":
        reranker = JinaReranker(
            doc_texts={},
            model_name=config.reranker.model,
            device=config.encoder.device,
            batch_size=config.reranker.batch_size
        )
        reranker._milvus_searcher = searcher

print("✓ API components initialized\n")


# FastAPI app
app = FastAPI(
    title="ArXplorer Query API",
    description="Academic paper search with hybrid retrieval and LLM-based query analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    retrieval_k: int = Field(200, ge=10, le=1000, description="Number of candidates for reranking")
    rewrite_query: bool = Field(True, description="Enable LLM query analysis")
    enable_reranking: bool = Field(False, description="Enable Jina reranking (slower)")


class SearchResultResponse(BaseModel):
    doc_id: str
    score: float
    rank: int
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    year: Optional[int] = None
    citation_count: Optional[int] = None


class QueryResponse(BaseModel):
    results: List[SearchResultResponse]
    intent: str
    query_time_ms: int
    components: Dict[str, Any]
    metadata: Dict[str, Any]


def search_result_to_dict(result: SearchResult) -> Dict:
    """Convert SearchResult to dictionary."""
    return {
        "doc_id": result.doc_id,
        "score": float(result.score),
        "rank": result.rank,
        "title": getattr(result, 'title', None),
        "abstract": getattr(result, 'abstract', None),
        "authors": getattr(result, 'authors', None),
        "categories": getattr(result, 'categories', None),
        "year": getattr(result, 'publication_year', None),
        "citation_count": result.citation_count
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ArXplorer Query API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "/api/v1/query",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Milvus connection
        num_entities = searcher.collection.num_entities
        
        return {
            "status": "healthy",
            "milvus": {
                "connected": True,
                "collection": config.milvus.collection_name,
                "documents": num_entities
            },
            "components": {
                "dense_encoder": "loaded",
                "sparse_encoder": "loaded",
                "query_rewriter": "enabled" if query_rewriter else "disabled",
                "intent_booster": "enabled" if intent_booster else "disabled",
                "reranker": "enabled" if reranker else "disabled"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_papers(request: QueryRequest):
    """
    Main query endpoint.
    
    Executes full retrieval pipeline:
    1. LLM query analysis (optional)
    2. Multi-query hybrid search
    3. Intent boosting
    4. Title/author matching
    5. Reranking (optional)
    """
    start_time = time.time()
    timings = {}
    
    try:
        # Step 1: LLM query analysis
        intent = 'default'
        year_constraint = None
        citation_threshold = None
        target_title = None
        target_authors = None
        rewrites = []
        
        if request.rewrite_query and query_rewriter:
            llm_start = time.time()
            from datetime import datetime
            
            result = query_rewriter.extract_intent_filters_and_rewrite(
                request.query,
                num_rewrites=config.query_rewriting.num_rewrites,
                current_year=datetime.now().year
            )
            
            intent = result.get('intent', 'default')
            year_constraint = result.get('year_constraint')
            citation_threshold = result.get('citation_threshold')
            target_title = result.get('target_title')
            target_authors = result.get('target_authors')
            rewrites = result.get('rewrites', [])
            
            timings['llm_analysis_ms'] = int((time.time() - llm_start) * 1000)
        
        # Add target title/authors as additional search queries
        additional_queries = []
        if target_title:
            additional_queries.append(target_title)
        if target_authors:
            authors_query = " ".join(target_authors)
            additional_queries.append(authors_query)
        
        all_rewrites = rewrites + additional_queries
        
        # Step 2: Multi-query hybrid search
        search_start = time.time()
        results = searcher.search_multi_query_with_filters(
            original_query=request.query,
            rewrites=all_rewrites,
            year_constraint=year_constraint,
            citation_threshold=citation_threshold,
            top_k=request.retrieval_k,
            retrieval_k=request.retrieval_k
        )
        timings['search_ms'] = int((time.time() - search_start) * 1000)
        
        if not results:
            return QueryResponse(
                results=[],
                intent=intent,
                query_time_ms=int((time.time() - start_time) * 1000),
                components=timings,
                metadata={
                    "filters_applied": bool(year_constraint or citation_threshold),
                    "rewrites_generated": len(rewrites)
                }
            )
        
        # Step 3: Intent boosting
        if intent_booster:
            boost_start = time.time()
            results = intent_booster.boost(results, intent)
            timings['boosting_ms'] = int((time.time() - boost_start) * 1000)
        
        # Step 4: Title/author matching
        if title_author_matcher and (target_title or target_authors):
            if intent in ['specific_paper', 'foundational']:
                match_start = time.time()
                results = title_author_matcher.match_and_boost(
                    results=results,
                    target_title=target_title,
                    target_authors=target_authors
                )
                timings['matching_ms'] = int((time.time() - match_start) * 1000)
        
        # Step 5: Reranking (optional)
        if request.enable_reranking and reranker and len(results) > 0:
            rerank_start = time.time()
            candidates = results[:config.reranker.rerank_top_k]
            
            # Build doc_texts
            doc_texts = {r.doc_id: r.abstract[:512] if hasattr(r, 'abstract') else '' for r in candidates}
            reranker.doc_texts = doc_texts
            
            # Rerank
            reranked_results = reranker.rerank(request.query, candidates, top_k=len(candidates))
            
            # Score fusion
            min_rerank = min(r.cross_encoder_score for r in reranked_results if r.cross_encoder_score is not None)
            max_rerank = max(r.cross_encoder_score for r in reranked_results if r.cross_encoder_score is not None)
            rerank_range = max_rerank - min_rerank if max_rerank > min_rerank else 1.0
            
            min_boost = min(r.score for r in reranked_results)
            max_boost = max(r.score for r in reranked_results)
            boost_range = max_boost - min_boost if max_boost > min_boost else 1.0
            
            for result in reranked_results:
                norm_boost = (result.score - min_boost) / boost_range
                norm_rerank = (result.cross_encoder_score - min_rerank) / rerank_range if result.cross_encoder_score is not None else 0.0
                
                result.score = (
                    config.reranker.pre_rerank_weight * norm_boost +
                    config.reranker.rerank_weight * norm_rerank
                )
            
            reranked_results.sort(key=lambda r: r.score, reverse=True)
            for rank, result in enumerate(reranked_results, 1):
                result.rank = rank
            
            results = reranked_results
            timings['reranking_ms'] = int((time.time() - rerank_start) * 1000)
        
        # Return top_k
        final_results = results[:request.top_k]
        
        total_time = int((time.time() - start_time) * 1000)
        
        return QueryResponse(
            results=[search_result_to_dict(r) for r in final_results],
            intent=intent,
            query_time_ms=total_time,
            components=timings,
            metadata={
                "filters_applied": bool(year_constraint or citation_threshold),
                "rewrites_generated": len(rewrites),
                "reranking_enabled": request.enable_reranking,
                "year_constraint": year_constraint,
                "citation_threshold": citation_threshold,
                "target_title": target_title,
                "target_authors": target_authors
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ArXplorer Query API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
