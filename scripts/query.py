"""Unified query script for dense, sparse, and hybrid search.

Usage:
    # Dense search only
    python scripts/query.py --method dense
    
    # Sparse search only
    python scripts/query.py --method sparse
    
    # Hybrid search with RRF fusion
    python scripts/query.py --method hybrid
    
    # Use custom config
    python scripts/query.py --config my_config.yaml --method hybrid
    
    # Override top-k
    python scripts/query.py --method hybrid --top-k 20
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.retrieval.encoders import DenseEncoder, SparseEncoder
from src.retrieval.indexers import DenseIndexer, SparseIndexer
from src.retrieval.searchers import DenseSearcher, SparseSearcher, HybridSearcher, SearchResult
from src.retrieval.query_rewriting import LLMQueryRewriter
from src.retrieval.rerankers import CrossEncoderReranker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query dense, sparse, or hybrid search indexes"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["dense", "sparse", "hybrid"],
        help="Search method: dense, sparse, or hybrid (RRF fusion)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override number of results to return"
    )
    
    parser.add_argument(
        "--retrieval-k",
        type=int,
        help="Override retrieval-k for hybrid (retrieve from each before fusion)"
    )
    
    parser.add_argument(
        "--dense-index",
        type=str,
        help="Override dense index directory"
    )
    
    parser.add_argument(
        "--sparse-index",
        type=str,
        help="Override sparse index directory"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Override device (cuda or cpu)"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to execute (non-interactive mode)"
    )
    
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking"
    )
    
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking (overrides config)"
    )
    
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        help="Override number of candidates to rerank"
    )
    
    parser.add_argument(
        "--rewrite-query",
        action="store_true",
        help="Enable LLM-based query rewriting"
    )
    
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Disable query rewriting (overrides config)"
    )
    
    return parser.parse_args()


def setup_dense_searcher(config: Config) -> DenseSearcher:
    """Initialize dense searcher."""
    print("Loading dense encoder and index...")
    
    encoder = DenseEncoder(
        model_name=config.encoder.dense_model,
        device=config.encoder.device,
        normalize=config.encoder.normalize_dense,
        use_specter2=config.encoder.use_specter2,
        specter2_adapter=config.encoder.specter2_adapter
    )
    
    indexer = DenseIndexer(
        encoder=encoder,
        output_dir=config.index.dense_output_dir
    )
    indexer.load()
    
    return DenseSearcher(indexer=indexer, encoder=encoder)


def setup_sparse_searcher(config: Config) -> SparseSearcher:
    """Initialize sparse searcher."""
    print("Loading sparse encoder and index...")
    
    encoder = SparseEncoder(
        model_name=config.encoder.sparse_model,
        device=config.encoder.device,
        max_length=config.encoder.max_length
    )
    
    indexer = SparseIndexer(
        encoder=encoder,
        output_dir=config.index.sparse_output_dir
    )
    indexer.load()
    
    return SparseSearcher(indexer=indexer, encoder=encoder)


def setup_reranker(config: Config, doc_map: dict) -> CrossEncoderReranker:
    """Initialize cross-encoder reranker."""
    print("Loading cross-encoder reranker...")
    
    # Extract doc_id -> text mapping from doc_map
    doc_texts = {}
    for idx, entry in doc_map.items():
        if isinstance(entry, dict):
            doc_texts[entry["id"]] = entry.get("text", "")
        else:
            # Old format compatibility
            doc_texts[entry] = ""
    
    reranker = CrossEncoderReranker(
        doc_texts=doc_texts,
        model_name=config.reranker.model,
        device=config.encoder.device,
        max_length=config.reranker.max_length,
        batch_size=config.reranker.batch_size
    )
    
    return reranker


def multi_query_rrf_fusion(result_lists: List[List[SearchResult]], top_k: int, rrf_k: int = 60) -> List[SearchResult]:
    """Combine results from multiple queries using RRF.
    
    Args:
        result_lists: List of result lists from different queries
        top_k: Number of final results to return
        rrf_k: RRF constant (default 60)
        
    Returns:
        Fused results sorted by RRF score
    """
    rrf_scores = defaultdict(float)
    
    # Apply RRF across all result lists
    for results in result_lists:
        for result in results:
            rrf_scores[result.doc_id] += 1.0 / (rrf_k + result.rank)
    
    # Sort by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create final results
    fused_results = []
    for rank, (doc_id, rrf_score) in enumerate(sorted_docs[:top_k], 1):
        fused_results.append(SearchResult(
            doc_id=doc_id,
            score=rrf_score,
            rank=rank
        ))
    
    return fused_results


def print_results(results, method: str, reranked: bool = False):
    """Print search results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{method.upper()} SEARCH RESULTS")
    print(f"{'='*60}")
    
    if not results:
        print("No results found.")
        return
    
    for result in results:
        print(f"Rank {result.rank:2d} | Score: {result.score:8.4f} | Doc ID: {result.doc_id}")
    
    print(f"{'='*60}\n")


def interactive_mode(searcher, method: str, config: Config, reranker=None, query_rewriter=None):
    """Run interactive query loop."""
    rerank_status = " with reranking" if reranker else ""
    if query_rewriter:
        num_rewrites = config.query_rewriting.num_rewrites
        rewrite_status = f" with {num_rewrites} query rewrite(s) + multi-query RRF"
    else:
        rewrite_status = ""
    print(f"\n{method.upper()} search ready{rerank_status}{rewrite_status}. Type your query (or 'exit' to quit).\n")
    
    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not query:
            continue
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        # Determine queries to execute
        original_query = query
        queries_to_search = [original_query]
        
        if query_rewriter:
            rewritten_queries = query_rewriter.rewrite(original_query)
            print(f"Original query: {original_query}")
            for i, rewritten in enumerate(rewritten_queries, 1):
                print(f"Rewritten query {i}: {rewritten}")
            
            # Add unique rewrites
            for rewritten in rewritten_queries:
                if rewritten.lower() != original_query.lower() and rewritten.lower() not in [q.lower() for q in queries_to_search]:
                    queries_to_search.append(rewritten)
            
            if len(queries_to_search) > 1:
                print(f"Retrieving results from {len(queries_to_search)} queries...")
            else:
                print(f"All rewrites identical to original, using single query...")
        
        # Search with all queries
        all_results = []
        retrieval_k = config.reranker.rerank_top_k if reranker else config.search.top_k
        
        for query_text in queries_to_search:
            if method == "hybrid":
                results = searcher.search(
                    query_text,
                    top_k=retrieval_k,
                    retrieval_k=config.search.retrieval_k
                )
            else:
                results = searcher.search(
                    query_text,
                    top_k=retrieval_k
                )
            all_results.append(results)
        
        # Fuse results if multiple queries
        if len(all_results) > 1:
            results = multi_query_rrf_fusion(
                all_results,
                top_k=retrieval_k,
                rrf_k=config.search.rrf_k
            )
        else:
            results = all_results[0]
        
        # Rerank if enabled
        if reranker and results:
            results = reranker.rerank(original_query, results, top_k=config.search.top_k)
        
        print_results(results, method, reranked=(reranker is not None))


def single_query_mode(searcher, query: str, method: str, config: Config, reranker=None, query_rewriter=None):
    """Execute a single query and exit."""
    rerank_status = " with reranking" if reranker else ""
    if query_rewriter:
        num_rewrites = config.query_rewriting.num_rewrites
        rewrite_status = f" with {num_rewrites} query rewrite(s) + multi-query RRF"
    else:
        rewrite_status = ""
    print(f"\nExecuting {method} search{rerank_status}{rewrite_status} for: {query}\n")
    
    # Determine queries to execute
    original_query = query
    queries_to_search = [original_query]
    
    if query_rewriter:
        rewritten_queries = query_rewriter.rewrite(original_query)
        print(f"Original query: {original_query}")
        for i, rewritten in enumerate(rewritten_queries, 1):
            print(f"Rewritten query {i}: {rewritten}")
        
        # Add unique rewrites
        for rewritten in rewritten_queries:
            if rewritten.lower() != original_query.lower() and rewritten.lower() not in [q.lower() for q in queries_to_search]:
                queries_to_search.append(rewritten)
        
        if len(queries_to_search) > 1:
            print(f"Retrieving results from {len(queries_to_search)} queries...\n")
        else:
            print(f"All rewrites identical to original, using single query...\n")
    
    # Search with all queries
    all_results = []
    retrieval_k = config.reranker.rerank_top_k if reranker else config.search.top_k
    
    for query_text in queries_to_search:
        if method == "hybrid":
            results = searcher.search(
                query_text,
                top_k=retrieval_k,
                retrieval_k=config.search.retrieval_k
            )
        else:
            results = searcher.search(
                query_text,
                top_k=retrieval_k
            )
        all_results.append(results)
    
    # Fuse results if multiple queries
    if len(all_results) > 1:
        results = multi_query_rrf_fusion(
            all_results,
            top_k=retrieval_k,
            rrf_k=config.search.rrf_k
        )
    else:
        results = all_results[0]
    
    # Rerank if enabled
    if reranker and results:
        results = reranker.rerank(original_query, results, top_k=config.search.top_k)
    
    print_results(results, method, reranked=(reranker is not None))


def main():
    args = parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config from: {args.config}")
        config = Config.from_yaml(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration")
        config = Config.default()
    
    # Apply CLI overrides
    if args.top_k:
        config.search.top_k = args.top_k
    if args.retrieval_k:
        config.search.retrieval_k = args.retrieval_k
    if args.rerank_top_k:
        config.reranker.rerank_top_k = args.rerank_top_k
    if args.dense_index:
        config.index.dense_output_dir = args.dense_index
    if args.sparse_index:
        config.index.sparse_output_dir = args.sparse_index
    if args.device:
        config.encoder.device = args.device
    
    # Handle reranking enable/disable
    if args.rerank:
        config.reranker.enabled = True
    elif args.no_rerank:
        config.reranker.enabled = False
    
    # Handle query rewriting enable/disable
    if args.rewrite_query:
        config.query_rewriting.enabled = True
    elif args.no_rewrite:
        config.query_rewriting.enabled = False
    
    # Setup searcher based on method
    print("\n" + "="*60)
    print(f"INITIALIZING {args.method.upper()} SEARCH")
    print("="*60 + "\n")
    
    indexer_for_reranker = None  # Will hold the indexer with doc_map
    
    if args.method == "dense":
        encoder = DenseEncoder(
            model_name=config.encoder.dense_model,
            device=config.encoder.device,
            normalize=config.encoder.normalize_dense,
            use_specter2=config.encoder.use_specter2,
            specter2_adapter=config.encoder.specter2_adapter
        )
        indexer_for_reranker = DenseIndexer(
            encoder=encoder,
            output_dir=config.index.dense_output_dir
        )
        indexer_for_reranker.load()
        searcher = DenseSearcher(indexer=indexer_for_reranker, encoder=encoder)
        
    elif args.method == "sparse":
        encoder = SparseEncoder(
            model_name=config.encoder.sparse_model,
            device=config.encoder.device,
            max_length=config.encoder.max_length
        )
        indexer_for_reranker = SparseIndexer(
            encoder=encoder,
            output_dir=config.index.sparse_output_dir
        )
        indexer_for_reranker.load()
        searcher = SparseSearcher(indexer=indexer_for_reranker, encoder=encoder)
        
    elif args.method == "hybrid":
        dense_encoder = DenseEncoder(
            model_name=config.encoder.dense_model,
            device=config.encoder.device,
            normalize=config.encoder.normalize_dense,
            use_specter2=config.encoder.use_specter2,
            specter2_adapter=config.encoder.specter2_adapter
        )
        dense_indexer = DenseIndexer(
            encoder=dense_encoder,
            output_dir=config.index.dense_output_dir
        )
        dense_indexer.load()
        dense_searcher = DenseSearcher(indexer=dense_indexer, encoder=dense_encoder)
        
        sparse_encoder = SparseEncoder(
            model_name=config.encoder.sparse_model,
            device=config.encoder.device,
            max_length=config.encoder.max_length
        )
        sparse_indexer = SparseIndexer(
            encoder=sparse_encoder,
            output_dir=config.index.sparse_output_dir
        )
        sparse_indexer.load()
        sparse_searcher = SparseSearcher(indexer=sparse_indexer, encoder=sparse_encoder)
        
        searcher = HybridSearcher(
            dense_searcher=dense_searcher,
            sparse_searcher=sparse_searcher,
            rrf_k=config.search.rrf_k
        )
        # Use dense indexer's doc_map for reranker (both should have same docs)
        indexer_for_reranker = dense_indexer
    
    print("✓ Initialization complete\n")
    
    # Setup reranker if enabled
    reranker = None
    if config.reranker.enabled and indexer_for_reranker:
        reranker = setup_reranker(config, indexer_for_reranker.doc_map)
        print("✓ Reranker ready\n")
    
    # Setup query rewriter if enabled
    query_rewriter = None
    if config.query_rewriting.enabled:
        print("Loading query rewriting model...")
        query_rewriter = LLMQueryRewriter(
            model_name=config.query_rewriting.model,
            device=config.query_rewriting.device,
            max_length=config.query_rewriting.max_length,
            temperature=config.query_rewriting.temperature,
            num_rewrites=config.query_rewriting.num_rewrites
        )
        print("✓ Query rewriter ready\n")
    
    # Run in appropriate mode
    if args.query:
        single_query_mode(searcher, args.query, args.method, config, reranker, query_rewriter)
    else:
        interactive_mode(searcher, args.method, config, reranker, query_rewriter)


if __name__ == "__main__":
    main()
