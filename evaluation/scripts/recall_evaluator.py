"""
Recall@10 and MRR evaluator for canonical papers.
Tests if target papers are retrieved in top-10 results for variant queries.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    query_api, 
    check_paper_in_results, 
    calculate_reciprocal_rank,
    get_api_endpoint_from_terraform, 
    test_api_connection
)


def evaluate_recall_and_mrr(
    canonical_queries_json: str,
    api_endpoint: str,
    output_file: str,
    top_k: int = 10
) -> Dict:
    """
    Evaluate Recall@10 and MRR on canonical paper queries.
    
    Args:
        canonical_queries_json: Path to canonical queries JSON
        api_endpoint: API endpoint URL
        output_file: Path to save detailed results
        top_k: Number of results to retrieve
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 60)
    print("Recall@10 and MRR Evaluation")
    print("=" * 60)
    print(f"API: {api_endpoint}")
    print(f"Dataset: {canonical_queries_json}")
    print(f"Top-K: {top_k}")
    print()
    
    # Load canonical queries
    with open(canonical_queries_json, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    print(f"Loaded {len(queries)} canonical queries")
    print()
    
    # Group queries by target paper
    papers = {}
    for q in queries:
        target_id = q['target_id']
        if target_id not in papers:
            papers[target_id] = {
                'entry': q['entry'],
                'target_id': target_id,
                'target_title': q['target_title'],
                'queries': []
            }
        papers[target_id]['queries'].append(q)
    
    print(f"Evaluating {len(papers)} unique canonical papers")
    print()
    
    # Evaluate each paper
    results = []
    total_reciprocal_rank = 0.0
    papers_found_any_variant = 0
    total_queries_with_hit = 0
    
    start_time = time.time()
    
    for i, (target_id, paper_data) in enumerate(papers.items()):
        entry = paper_data['entry']
        target_title = paper_data['target_title']
        paper_queries = paper_data['queries']
        
        print(f"[{i+1}/{len(papers)}] Paper {entry}: {target_title[:50]}...")
        print(f"  Testing {len(paper_queries)} query variants")
        
        query_results = []
        paper_found_in_any = False
        best_rank = None
        reciprocal_ranks = []
        
        for q in paper_queries:
            query = q['query']
            variant_type = q['variant_type']
            
            # Query API
            api_response = query_api(query, api_endpoint, top_k=top_k)
            
            if 'error' in api_response:
                print(f"    ✗ Query failed: {query[:40]}...")
                query_results.append({
                    'query': query,
                    'variant_type': variant_type,
                    'found': False,
                    'rank': None,
                    'error': api_response['error']
                })
                continue
            
            papers_retrieved = api_response.get('results', [])
            
            # Check if target paper is in results
            found, rank = check_paper_in_results(target_id, papers_retrieved)
            rr = calculate_reciprocal_rank(target_id, papers_retrieved)
            
            if found:
                paper_found_in_any = True
                total_queries_with_hit += 1
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                print(f"    ✓ [{variant_type}] Found at rank {rank}")
            else:
                print(f"    ✗ [{variant_type}] Not found")
            
            reciprocal_ranks.append(rr)
            
            query_results.append({
                'query': query,
                'variant_type': variant_type,
                'found': found,
                'rank': rank,
                'reciprocal_rank': rr,
                'num_results': len(papers_retrieved)
            })
            
            # Small delay
            time.sleep(0.3)
        
        if paper_found_in_any:
            papers_found_any_variant += 1
        
        # Calculate MRR for this paper (average of all variants)
        avg_rr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        total_reciprocal_rank += avg_rr
        
        results.append({
            'entry': entry,
            'target_id': target_id,
            'target_title': target_title,
            'found_in_any_variant': paper_found_in_any,
            'best_rank': best_rank,
            'avg_reciprocal_rank': avg_rr,
            'query_results': query_results
        })
        
        print(f"  Paper found: {paper_found_in_any}, Best rank: {best_rank}, Avg RR: {avg_rr:.3f}")
        print()
    
    elapsed_time = time.time() - start_time
    
    # Calculate overall metrics
    recall_at_10 = papers_found_any_variant / len(papers) if papers else 0.0
    mrr = total_reciprocal_rank / len(papers) if papers else 0.0
    query_level_recall = total_queries_with_hit / len(queries) if queries else 0.0
    
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'canon_papers_60',
        'num_canonical_papers': len(papers),
        'total_queries': len(queries),
        'top_k': top_k,
        'recall_at_10': recall_at_10,
        'mrr': mrr,
        'papers_found_any_variant': papers_found_any_variant,
        'query_level_recall': query_level_recall,
        'elapsed_time_seconds': elapsed_time,
        'api_endpoint': api_endpoint,
        'detailed_results': results
    }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Total canonical papers: {len(papers)}")
    print(f"Total queries: {len(queries)}")
    print(f"Papers found (any variant): {papers_found_any_variant}/{len(papers)} ({recall_at_10:.1%})")
    print(f"Recall@{top_k}: {recall_at_10:.3f}")
    print(f"MRR: {mrr:.3f}")
    print(f"Query-level recall: {query_level_recall:.3f}")
    print(f"Elapsed time: {elapsed_time:.1f}s")
    print(f"Saved to: {output_file}")
    print()
    
    return evaluation_results


def main():
    """Run recall and MRR evaluation."""
    # Paths
    eval_dir = Path(__file__).parent.parent
    data_dir = eval_dir / "data"
    results_dir = eval_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    canonical_queries = data_dir / "canonical_queries.json"
    output_file = results_dir / "recall_results.json"
    
    # Check if canonical queries exist
    if not canonical_queries.exists():
        print(f"Error: Canonical queries not found: {canonical_queries}")
        print("Run query_generator.py first to generate queries")
        return
    
    # Get API endpoint
    api_endpoint = get_api_endpoint_from_terraform()
    if not api_endpoint:
        print("Error: Could not get API endpoint from terraform")
        print("Set ARXPLORER_API_ENDPOINT environment variable or provide manually")
        return
    
    # Test connection
    if not test_api_connection(api_endpoint):
        print("Error: Could not connect to API")
        return
    
    print()
    
    # Run evaluation
    evaluate_recall_and_mrr(
        str(canonical_queries),
        api_endpoint,
        str(output_file),
        top_k=10
    )


if __name__ == "__main__":
    main()
