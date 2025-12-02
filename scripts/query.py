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
from src.retrieval.query_rewriting import LLMQueryRewriter
from src.retrieval.rerankers.intent_booster import IntentBooster
from src.retrieval.rerankers.title_author_matcher import TitleAuthorMatcher


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
    
    if reranker_type == "jina":
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


def execute_query_pipeline(
    query: str,
    searcher: MilvusHybridSearcher,
    config: Config,
    reranker=None,
    query_rewriter=None,
    intent_booster=None,
    title_author_matcher=None
) -> List[SearchResult]:
    """
    Execute full query pipeline with structured query analysis.
    
    Pipeline:
    1. LLM: Extract intent, filters, target_title, target_authors, and generate rewrites
    2. Search: Multi-query hybrid search with varied filters + unfiltered fallback
    3. Intent boosting: Apply citation/date boosts based on intent
    4. Title/Author matching: Boost papers with matching titles/authors (specific_paper/foundational only)
    5. Reranking: Cross-encoder reranking (Jina listwise)
    6. Final fusion: Combine boosted scores + reranker scores (0.7 + 0.3)
    
    Args:
        query: User query string
        searcher: MilvusHybridSearcher instance
        config: Configuration object
        reranker: Optional reranker (Jina/Qwen/CrossEncoder)
        query_rewriter: Optional LLM rewriter
        intent_booster: Optional IntentBooster for post-RRF boosting
        title_author_matcher: Optional TitleAuthorMatcher for fuzzy matching
        
    Returns:
        Final ranked results
    """
    from datetime import datetime
    
    original_query = query
    intent = 'default'
    year_constraint = None
    citation_threshold = None
    target_title = None
    target_authors = None
    rewrites = []
    
    # Step 1: LLM structured query analysis
    if query_rewriter:
        print("Analyzing query with LLM...")
        result = query_rewriter.extract_intent_filters_and_rewrite(
            original_query,
            num_rewrites=config.query_rewriting.num_rewrites,
            current_year=datetime.now().year
        )
        
        intent = result.get('intent', 'default')
        year_constraint = result.get('year_constraint')
        citation_threshold = result.get('citation_threshold')
        target_title = result.get('target_title')
        target_authors = result.get('target_authors')
        rewrites = result.get('rewrites', [])
        
        # Display extracted information
        print(f"  Intent: {intent}")
        
        # Display target title/authors if extracted
        if target_title:
            print(f"  Target Title: {target_title}")
        if target_authors:
            print(f"  Target Authors: {', '.join(target_authors)}")
        
        filter_parts = []
        if year_constraint:
            year_min = year_constraint.get('min')
            year_max = year_constraint.get('max')
            if year_min and year_max:
                if year_min == year_max:
                    filter_parts.append(f"Year: {year_min}")
                else:
                    filter_parts.append(f"Year: {year_min}-{year_max}")
            elif year_min:
                filter_parts.append(f"Year: ≥{year_min}")
            elif year_max:
                filter_parts.append(f"Year: ≤{year_max}")
        
        if citation_threshold:
            filter_parts.append(f"Citations: ≥{citation_threshold}")
        
        if filter_parts:
            print(f"  Filters: {', '.join(filter_parts)}")
        else:
            print(f"  Filters: None")
        
        if rewrites:
            print(f"  Rewrites: {len(rewrites)} variants generated")
            for i, rewrite in enumerate(rewrites[:2], 1):  # Show first 2
                if rewrite != original_query:
                    print(f"    {i}. {rewrite[:80]}...")
    
    # Step 2: Multi-query hybrid search with varied filters
    print(f"\nSearching Milvus (retrieval_k={config.search.retrieval_k})...")
    
    # Add target_title and target_authors as additional search queries if extracted
    additional_queries = []
    if target_title:
        additional_queries.append(target_title)
        print(f"  Added target_title as search query: {target_title[:60]}...")
    if target_authors:
        # Join authors into a single query string
        authors_query = " ".join(target_authors)
        additional_queries.append(authors_query)
        print(f"  Added target_authors as search query: {authors_query[:60]}...")
    
    # Combine rewrites with additional queries
    all_rewrites = rewrites + additional_queries
    
    results = searcher.search_multi_query_with_filters(
        original_query=original_query,
        rewrites=all_rewrites,
        year_constraint=year_constraint,
        citation_threshold=citation_threshold,
        top_k=config.search.retrieval_k,  # Retrieve more for reranking
        retrieval_k=config.search.retrieval_k
    )
    
    print(f"  Retrieved: {len(results)} results")
    
    if not results:
        print("  No results found.")
        return []
    
    # Step 3: Intent-based boosting (post-RRF)
    if intent_booster:
        print(f"\nApplying intent-based boosting (intent={intent})...")
        results = intent_booster.boost(results, intent)
        print(f"  Boosted scores computed")
        
        # Show boost summary for top results
        if len(results) >= 3:
            print(f"\n  Top 3 after boosting:")
            for i, r in enumerate(results[:3], 1):
                bc = r.boost_components if hasattr(r, 'boost_components') else {}
                print(f"    {i}. {r.doc_id[:30]} | Score: {r.score:.4f} | Citations: {r.citation_count}")
    
    # Step 4: Title/Author fuzzy matching (specific_paper and foundational intents only)
    if title_author_matcher and (target_title or target_authors):
        # Only apply for specific_paper and foundational intents
        if intent in ['specific_paper', 'foundational']:
            print(f"\nApplying title/author fuzzy matching...")
            
            # Match on all results from hybrid search
            results = title_author_matcher.match_and_boost(
                results=results,
                target_title=target_title,
                target_authors=target_authors
            )
            
            print(f"  Matching boost applied to {len(results)} candidates")
            
            # Show papers with matches
            matched_papers = [r for r in results if hasattr(r, 'boost_components') and 
                            (r.boost_components.get('title_match') or r.boost_components.get('author_match'))]
            
            if matched_papers:
                print(f"\n  Papers with title/author matches:")
                for i, r in enumerate(matched_papers[:3], 1):
                    bc = r.boost_components
                    match_info = []
                    if bc.get('title_match'):
                        match_info.append(f"Title: {bc.get('title_score', 0):.2f}")
                    if bc.get('author_match'):
                        match_info.append(f"Author: {bc.get('author_score', 0):.2f}")
                    print(f"    {i}. {r.doc_id[:30]} | Boost: +{bc.get('match_boost', 0):.2f} | {', '.join(match_info)}")
    
    # Step 4b: Apply title/author matching to original query (fallback if LLM didn't extract)
    # This catches cases where user query directly contains title/author info
    if title_author_matcher:
        # Only match against original query if we're in specific_paper/foundational intent
        # AND if LLM didn't already extract title/authors (to avoid double-boosting)
        if not target_title and not target_authors:
            print(f"\nApplying title/author matching to original query...")
            
            # Match using the original query as target
            results = title_author_matcher.match_and_boost(
                results=results,
                target_title=original_query,  # Use query as title target
                target_authors=None  # Don't try to extract authors from query string
            )
            
            print(f"  Query-based matching applied to {len(results)} candidates")
            
            # Show papers with matches
            matched_papers = [r for r in results if hasattr(r, 'boost_components') and 
                            (r.boost_components.get('title_match') or r.boost_components.get('author_match'))]
            
            if matched_papers:
                print(f"\n  Papers matching original query:")
                for i, r in enumerate(matched_papers[:3], 1):
                    bc = r.boost_components
                    match_info = []
                    if bc.get('title_match'):
                        match_info.append(f"Title: {bc.get('title_score', 0):.2f}")
                    if bc.get('author_match'):
                        match_info.append(f"Author: {bc.get('author_score', 0):.2f}")
                    print(f"    {i}. {r.doc_id[:30]} | Boost: +{bc.get('match_boost', 0):.2f} | {', '.join(match_info)}")
    
    # Step 5: Cross-encoder reranking
    reranked = False
    if reranker and len(results) > 0:
        # Take top rerank_top_k for reranking
        candidates = results[:config.reranker.rerank_top_k]
        
        print(f"\nReranking with {config.reranker.type} (top {len(candidates)})...")
        
        # Build doc_texts from abstracts
        doc_texts = {r.doc_id: r.abstract[:512] if hasattr(r, 'abstract') else '' for r in candidates}
        reranker.doc_texts = doc_texts
        
        # Rerank
        reranked_results = reranker.rerank(original_query, candidates, top_k=len(candidates))
        reranked = True
        
        # Step 6: Final score fusion (0.7 boosted + 0.3 reranker)
        print(f"\nFusing scores ({config.reranker.pre_rerank_weight:.1f} boosted + {config.reranker.rerank_weight:.1f} reranker)...")
        
        # Normalize reranker scores to [0, 1]
        reranker_scores = [r.cross_encoder_score for r in reranked_results if r.cross_encoder_score is not None]
        if reranker_scores:
            min_rerank = min(reranker_scores)
            max_rerank = max(reranker_scores)
            rerank_range = max_rerank - min_rerank if max_rerank > min_rerank else 1.0
        else:
            min_rerank = 0.0
            rerank_range = 1.0
        
        # Normalize boosted scores to [0, 1]
        boosted_scores = [r.score for r in reranked_results]
        min_boost = min(boosted_scores)
        max_boost = max(boosted_scores)
        boost_range = max_boost - min_boost if max_boost > min_boost else 1.0
        
        # Fuse scores
        for result in reranked_results:
            # Normalize scores
            norm_boost = (result.score - min_boost) / boost_range
            norm_rerank = (result.cross_encoder_score - min_rerank) / rerank_range if result.cross_encoder_score is not None else 0.0
            
            # Weighted fusion
            result.score = (
                config.reranker.pre_rerank_weight * norm_boost +
                config.reranker.rerank_weight * norm_rerank
            )
        
        # Re-sort by fused scores
        reranked_results.sort(key=lambda r: r.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(reranked_results, 1):
            result.rank = rank
        
        results = reranked_results
    
    # Return top_k final results
    return results[:config.search.top_k], intent


def print_results(results: List[SearchResult], reranked: bool = False, intent: str = None):
    """Print search results in a formatted way."""
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    if reranked:
        print("(with intent boosting + reranking + fusion)")
    if intent:
        print(f"Intent: {intent}")
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
        
        # Show component scores if available
        if hasattr(result, 'boost_components'):
            bc = result.boost_components
            print(f"  Boosting: Base RRF={bc.get('base_rrf', 0):.4f}, Cite={bc.get('citation_boost', 0):.4f}, Date={bc.get('date_boost', 0):.4f}")
        
        # Show reranker score if available
        if reranked and result.cross_encoder_score is not None:
            print(f"  Reranker: {result.cross_encoder_score:.4f}")
    
    print(f"\n{'='*70}\n")


def interactive_mode(searcher: MilvusHybridSearcher, config: Config, reranker=None, query_rewriter=None, intent_booster=None, title_author_matcher=None):
    """Run interactive query loop."""
    rerank_status = " + reranking" if reranker else ""
    rewrite_status = " + LLM analysis" if query_rewriter else ""
    boost_status = " + intent boosting" if intent_booster else ""
    match_status = " + title/author matching" if title_author_matcher else ""
    
    print(f"\nQuery pipeline ready{rewrite_status}{boost_status}{match_status}{rerank_status}. Type your query (or 'exit' to quit).\n")
    
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
        
        try:
            results, intent = execute_query_pipeline(
                query=query,
                searcher=searcher,
                config=config,
                reranker=reranker,
                query_rewriter=query_rewriter,
                intent_booster=intent_booster,
                title_author_matcher=title_author_matcher
            )
            
            print_results(results, reranked=(reranker is not None), intent=intent)
            
        except Exception as e:
            print(f"\nError executing query: {e}")
            import traceback
            traceback.print_exc()


def single_query_mode(searcher: MilvusHybridSearcher, query: str, config: Config, reranker=None, query_rewriter=None, intent_booster=None, title_author_matcher=None):
    """Execute a single query and exit."""
    rerank_status = " + reranking" if reranker else ""
    rewrite_status = " + LLM analysis" if query_rewriter else ""
    boost_status = " + intent boosting" if intent_booster else ""
    match_status = " + title/author matching" if title_author_matcher else ""
    
    print(f"\nExecuting query pipeline{rewrite_status}{boost_status}{match_status}{rerank_status}:\n  '{query}'\n")
    
    try:
        results, intent = execute_query_pipeline(
            query=query,
            searcher=searcher,
            config=config,
            reranker=reranker,
            query_rewriter=query_rewriter,
            intent_booster=intent_booster,
            title_author_matcher=title_author_matcher
        )
        
        print_results(results, reranked=(reranker is not None), intent=intent)
        
    except Exception as e:
        print(f"\nError executing query: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
    
    print("Milvus searcher ready\n")
    
    # Setup reranker if enabled
    reranker = None
    if config.reranker.enabled:
        reranker = setup_reranker(config, searcher)
        print("Reranker ready\n")
        if config.reranker.citation_boost_weight > 0:
            print(f"Citation boost enabled: weight={config.reranker.citation_boost_weight:.2f}\n")
    
    # Setup query rewriter if enabled
    query_rewriter = None
    if config.query_rewriting.enabled:
        if hasattr(config.query_rewriting, 'use_bedrock') and config.query_rewriting.use_bedrock:
            print(f"Initializing AWS Bedrock query rewriter: {config.query_rewriting.bedrock_model_id}")
        elif config.query_rewriting.use_vllm:
            print(f"Initializing vLLM query rewriter: {config.query_rewriting.vllm_endpoint}")
        else:
            print("Loading query rewriting model...")
        
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
        print("Query rewriter ready\n")
    
    # Setup intent booster if config exists and enabled
    intent_booster = None
    if hasattr(config, 'intent_boosting') and config.intent_boosting.enabled:
        print("Initializing intent-based boosting...")
        intent_booster = IntentBooster(
            citation_weights=config.intent_boosting.citation_weights,
            date_weights=config.intent_boosting.date_weights
        )
        print("Intent booster ready\n")
    
    # Setup title/author matcher if config exists and enabled
    title_author_matcher = None
    if hasattr(config, 'title_author_matching') and config.title_author_matching.enabled:
        print("Initializing title/author fuzzy matching...")
        title_author_matcher = TitleAuthorMatcher(
            title_threshold=config.title_author_matching.title_threshold,
            author_threshold=config.title_author_matching.author_threshold,
            title_boost_weight=config.title_author_matching.title_boost_weight,
            author_boost_weight=config.title_author_matching.author_boost_weight
        )
        print("Title/author matcher ready\n")
    
    # Run in appropriate mode
    if args.query:
        single_query_mode(searcher, args.query, config, reranker, query_rewriter, intent_booster, title_author_matcher)
    else:
        interactive_mode(searcher, config, reranker, query_rewriter, intent_booster, title_author_matcher)


if __name__ == "__main__":
    main()
