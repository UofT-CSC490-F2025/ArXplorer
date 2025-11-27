"""Query script for Milvus hybrid search with reranking and weighted fusion.

Usage:
    # Interactive mode
    python scripts/query_milvus.py
    
    # Single query
    python scripts/query_milvus.py --query "neural networks"
    
    # Override top-k
    python scripts/query_milvus.py --top-k 20
    
    # Disable reranking for faster queries
    python scripts/query_milvus.py --no-rerank
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.retrieval.encoders import DenseEncoder, SparseEncoder
from src.retrieval.searchers import MilvusHybridSearcher, SearchResult
from src.retrieval.query_rewriting import LLMQueryRewriter, build_milvus_filter_expr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query Milvus hybrid search with reranking"
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
        help="Override retrieval-k for hybrid (retrieve from Milvus before reranking)"
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


def setup_milvus_searcher(config: Config) -> MilvusHybridSearcher:
    """Initialize Milvus hybrid searcher."""
    print("Loading encoders and connecting to Milvus...")
    
    # Dense encoder
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
    sparse_encoder = SparseEncoder(
        model_name=config.encoder.sparse_model,
        device=config.encoder.device,
        max_length=config.encoder.max_length
    )
    
    # Create Milvus searcher
    searcher = MilvusHybridSearcher(
        dense_encoder=dense_encoder,
        sparse_encoder=sparse_encoder,
        host=config.milvus.host,
        port=config.milvus.port,
        collection_name=config.milvus.collection_name,
        rrf_k=config.search.rrf_k
    )
    
    print(f"✓ Connected to Milvus at {config.milvus.host}:{config.milvus.port}")
    print(f"✓ Using collection: {config.milvus.collection_name}\n")
    
    return searcher


def setup_reranker(config: Config, searcher: MilvusHybridSearcher):
    """Initialize reranker based on config type."""
    reranker_type = config.reranker.type.lower()
    
    if reranker_type == "qwen":
        from src.retrieval.rerankers import QwenReranker
        print(f"Loading Qwen reranker: {config.reranker.model}...")
        
        # Get all doc_ids from Milvus to build doc_texts mapping
        # We'll fetch texts on-demand during reranking
        doc_texts = {}
        
        reranker = QwenReranker(
            doc_texts=doc_texts,
            model_name=config.reranker.model,
            device=config.encoder.device,
            max_length=config.reranker.max_length,
            batch_size=config.reranker.batch_size,
            instruction=config.reranker.instruction
        )
    elif reranker_type == "jina":
        from src.retrieval.rerankers import JinaReranker
        print(f"Loading Jina reranker: {config.reranker.model}...")
        
        doc_texts = {}
        
        reranker = JinaReranker(
            doc_texts=doc_texts,
            model_name=config.reranker.model,
            device=config.encoder.device,
            batch_size=config.reranker.batch_size
        )
    else:  # cross-encoder (default)
        from src.retrieval.rerankers import CrossEncoderReranker
        print(f"Loading cross-encoder reranker: {config.reranker.model}...")
        
        doc_texts = {}
        
        reranker = CrossEncoderReranker(
            doc_texts=doc_texts,
            model_name=config.reranker.model,
            device=config.encoder.device,
            max_length=config.reranker.max_length,
            batch_size=config.reranker.batch_size
        )
    
    # Set searcher for fetching texts on-demand
    reranker._milvus_searcher = searcher
    
    return reranker


def print_results(results: List[SearchResult], reranked: bool = False, citation_boosted: bool = False):
    """Print search results in a formatted way."""
    print(f"\n{'='*70}")
    print(f"MILVUS HYBRID SEARCH RESULTS")
    if reranked:
        print("(with reranking)")
    if citation_boosted:
        print("(with citation boost)")
    print(f"{'='*70}")
    
    if not results:
        print("No results found.")
        return
    
    for result in results:
        # Base info
        print(f"\nRank {result.rank:2d} | Score: {result.score:8.4f}")
        print(f"  Doc ID: {result.doc_id}")
        
        # Show metadata if available
        if hasattr(result, 'title') and result.title:
            print(f"  Title: {result.title}")
        if hasattr(result, 'publication_year') and result.publication_year:
            print(f"  Year: {result.publication_year}")
        if hasattr(result, 'citation_count') and result.citation_count is not None:
            print(f"  Citations: {result.citation_count}")
        
        # Show reranker score if available
        if reranked and result.cross_encoder_score is not None:
            print(f"  Reranker Score: {result.cross_encoder_score:.4f}")
    
    print(f"\n{'='*70}\n")


def apply_citation_boost(results: List[SearchResult], citation_boost_weight: float = 0.1) -> List[SearchResult]:
    """Apply simple citation boost to reranked results.
    
    Formula: final_score = rerank_score + citation_boost_weight * log(citations + 1)
    
    Args:
        results: Reranked search results
        citation_boost_weight: Weight for citation boost (default 0.1)
        
    Returns:
        Results with citation-boosted scores, re-sorted and re-ranked
    """
    import math
    
    boosted_results = []
    for result in results:
        citation_count = result.citation_count if hasattr(result, 'citation_count') and result.citation_count is not None else 0
        citation_boost = citation_boost_weight * math.log10(citation_count + 1)
        
        # Create new result with boosted score
        boosted_result = SearchResult(
            doc_id=result.doc_id,
            score=result.score + citation_boost,  # Add boost to reranker score
            rank=0,  # Will be updated after sorting
            dense_score=result.dense_score,
            sparse_score=result.sparse_score,
            cross_encoder_score=result.cross_encoder_score,  # Preserve original reranker score
            citation_count=citation_count,
            publication_year=result.publication_year if hasattr(result, 'publication_year') else None
        )
        
        # Copy metadata attributes
        if hasattr(result, 'title'):
            boosted_result.title = result.title
        if hasattr(result, 'abstract'):
            boosted_result.abstract = result.abstract
        if hasattr(result, 'authors'):
            boosted_result.authors = result.authors
        if hasattr(result, 'categories'):
            boosted_result.categories = result.categories
        
        boosted_results.append(boosted_result)
    
    # Sort by boosted score and update ranks
    boosted_results.sort(key=lambda x: x.score, reverse=True)
    for rank, result in enumerate(boosted_results, 1):
        result.rank = rank
    
    return boosted_results


def interactive_mode(searcher: MilvusHybridSearcher, config: Config, reranker=None, query_rewriter=None):
    """Run interactive query loop."""
    rerank_status = " with reranking" if reranker else ""
    rewrite_status = ""
    if query_rewriter:
        num_rewrites = config.query_rewriting.num_rewrites
        rewrite_status = f" with {num_rewrites} query rewrite(s)"
    
    print(f"\nMilvus hybrid search ready{rerank_status}{rewrite_status}. Type your query (or 'exit' to quit).\n")
    
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
        
        original_query = query
        
        # Extract filters and rewrite query if enabled
        filter_expr = None
        filter_info = None
        
        if query_rewriter:
            from datetime import datetime
            result = query_rewriter.extract_filters_and_rewrite(
                original_query,
                current_year=datetime.now().year
            )
            
            filters = result.get('filters', {})
            confidence = result.get('confidence', 0.0)
            rewrites = result.get('rewrites', [original_query])
            
            # Display filter info
            if filters:
                year_filter = filters.get('year', {})
                citation_filter = filters.get('citation_count', {})
                
                filter_parts = []
                if year_filter.get('min') or year_filter.get('max'):
                    if year_filter.get('min') and year_filter.get('max'):
                        if year_filter['min'] == year_filter['max']:
                            filter_parts.append(f"Year: {year_filter['min']}")
                        else:
                            filter_parts.append(f"Year: {year_filter['min']}-{year_filter['max']}")
                    elif year_filter.get('min'):
                        filter_parts.append(f"Year: ≥{year_filter['min']}")
                    else:
                        filter_parts.append(f"Year: ≤{year_filter['max']}")
                
                if citation_filter.get('min'):
                    filter_parts.append(f"Citations: ≥{citation_filter['min']}")
                
                if filter_parts:
                    filter_info = f"🔍 Filters: {', '.join(filter_parts)} (confidence: {confidence:.2f})"
                else:
                    filter_info = "🔍 No filters applied"
                
                # Build filter expression if confidence is high enough
                if confidence >= config.query_rewriting.filter_confidence_threshold:
                    filter_expr = build_milvus_filter_expr(
                        filters,
                        enable_citation_filters=config.query_rewriting.enable_citation_filters,
                        enable_year_filters=config.query_rewriting.enable_year_filters
                    )
            else:
                filter_info = "🔍 No filters applied"
            
            # Display filter status
            print(filter_info)
            
            # Display rewrites
            if rewrites and rewrites[0] != original_query:
                print(f"Rewritten query: {rewrites[0]}")
        
        # Search with Milvus with filter (use rewrites as query_variants if available)
        retrieval_k = config.reranker.rerank_top_k if reranker else config.search.top_k
        query_variants = rewrites[1:] if rewrites and len(rewrites) > 1 else None  # Skip first rewrite (already displayed)
        results = searcher.search(original_query, top_k=retrieval_k, filter_expr=filter_expr, query_variants=query_variants)
        
        # Fallback: if filters are too restrictive, retry without them
        if filter_expr and len(results) < 5:
            print(f"⚠️  Only {len(results)} results with filters. Retrying without filters...")
            results = searcher.search(original_query, top_k=retrieval_k, filter_expr=None, query_variants=query_variants)
        
        # Rerank if enabled
        if reranker and results:
            # Build doc_texts for reranker from abstract attribute
            doc_texts = {r.doc_id: r.abstract[:512] if hasattr(r, 'abstract') else '' for r in results}
            reranker.doc_texts = doc_texts
            
            # Rerank
            results = reranker.rerank(original_query, results, top_k=config.search.top_k)
            
            # Apply citation boost
            if config.reranker.citation_boost_weight > 0:
                results = apply_citation_boost(results, config.reranker.citation_boost_weight)
        
        print_results(results, reranked=(reranker is not None), citation_boosted=(config.reranker.citation_boost_weight > 0 if reranker else False))


def single_query_mode(searcher: MilvusHybridSearcher, query: str, config: Config, reranker=None, query_rewriter=None):
    """Execute a single query and exit."""
    rerank_status = " with reranking" if reranker else ""
    rewrite_status = ""
    if query_rewriter:
        rewrite_status = " with filter extraction"
    
    print(f"\nExecuting Milvus hybrid search{rerank_status}{rewrite_status} for: {query}\n")
    
    original_query = query
    
    # Extract filters and rewrite query if enabled
    filter_expr = None
    filter_info = None

    rewrites = None
    
    if query_rewriter:
        from datetime import datetime
        result = query_rewriter.extract_filters_and_rewrite(
            original_query,
            current_year=datetime.now().year
        )
        
        filters = result.get('filters', {})
        confidence = result.get('confidence', 0.0)
        rewrites = result.get('rewrites', [original_query])
        
        # Display filter info
        if filters:
            year_filter = filters.get('year', {})
            citation_filter = filters.get('citation_count', {})
            
            filter_parts = []
            if year_filter.get('min') or year_filter.get('max'):
                if year_filter.get('min') and year_filter.get('max'):
                    if year_filter['min'] == year_filter['max']:
                        filter_parts.append(f"Year: {year_filter['min']}")
                    else:
                        filter_parts.append(f"Year: {year_filter['min']}-{year_filter['max']}")
                elif year_filter.get('min'):
                    filter_parts.append(f"Year: ≥{year_filter['min']}")
                else:
                    filter_parts.append(f"Year: ≤{year_filter['max']}")
            
            if citation_filter.get('min'):
                filter_parts.append(f"Citations: ≥{citation_filter['min']}")
            
            if filter_parts:
                filter_info = f"🔍 Filters: {', '.join(filter_parts)} (confidence: {confidence:.2f})"
            else:
                filter_info = "🔍 No filters applied"
            
            # Build filter expression if confidence is high enough
            if confidence >= config.query_rewriting.filter_confidence_threshold:
                filter_expr = build_milvus_filter_expr(
                    filters,
                    enable_citation_filters=config.query_rewriting.enable_citation_filters,
                    enable_year_filters=config.query_rewriting.enable_year_filters
                )
        else:
            filter_info = "🔍 No filters applied"
        
        # Display filter status
        print(filter_info)
        
        # Display rewrites
        if rewrites and rewrites[0] != original_query:
            print(f"Rewritten query: {rewrites[0]}\n")
    
    # Search with Milvus with filter (use rewrites as query_variants if available)
    retrieval_k = config.reranker.rerank_top_k if reranker else config.search.top_k
    query_variants = rewrites[1:] if rewrites and len(rewrites) > 1 else None  # Skip first rewrite (already displayed)
    results = searcher.search(original_query, top_k=retrieval_k, filter_expr=filter_expr, query_variants=query_variants)
    
    # Fallback: if filters are too restrictive, retry without them
    if filter_expr and len(results) < 5:
        print(f"⚠️  Only {len(results)} results with filters. Retrying without filters...\n")
        results = searcher.search(original_query, top_k=retrieval_k, filter_expr=None, query_variants=query_variants)
    
    # Rerank if enabled
    if reranker and results:
        # Build doc_texts for reranker from abstract attribute
        doc_texts = {r.doc_id: r.abstract[:512] if hasattr(r, 'abstract') else '' for r in results}
        reranker.doc_texts = doc_texts
        
        # Rerank
        results = reranker.rerank(original_query, results, top_k=config.search.top_k)
        
        # Apply citation boost
        if config.reranker.citation_boost_weight > 0:
            results = apply_citation_boost(results, config.reranker.citation_boost_weight)
    
    print_results(results, reranked=(reranker is not None), citation_boosted=(config.reranker.citation_boost_weight > 0 if reranker else False))


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
    
    # Setup Milvus searcher
    print("\n" + "="*70)
    print("INITIALIZING MILVUS HYBRID SEARCH")
    print("="*70 + "\n")
    
    searcher = setup_milvus_searcher(config)
    
    print("✓ Milvus searcher ready\n")
    
    # Setup reranker if enabled
    reranker = None
    if config.reranker.enabled:
        reranker = setup_reranker(config, searcher)
        print("✓ Reranker ready\n")
        if config.reranker.citation_boost_weight > 0:
            print(f"Citation boost enabled: weight={config.reranker.citation_boost_weight:.2f}\n")
    
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
        single_query_mode(searcher, args.query, config, reranker, query_rewriter)
    else:
        interactive_mode(searcher, config, reranker, query_rewriter)


if __name__ == "__main__":
    main()
