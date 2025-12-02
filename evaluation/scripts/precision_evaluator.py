"""
Precision@10 evaluator for queries_ml_100.csv.
Uses LLM judge to assess relevance of retrieved papers.
"""

import csv
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import query_api, get_api_endpoint_from_terraform, test_api_connection
from llm_judge import LLMJudge


def evaluate_precision(
    queries_csv: str,
    api_endpoint: str,
    output_file: str,
    top_k: int = 10
) -> Dict:
    """
    Evaluate Precision@10 on queries_ml_100.csv.
    
    Args:
        queries_csv: Path to queries CSV file
        api_endpoint: API endpoint URL
        output_file: Path to save detailed results
        top_k: Number of results to retrieve and evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 60)
    print("Precision@10 Evaluation")
    print("=" * 60)
    print(f"API: {api_endpoint}")
    print(f"Dataset: {queries_csv}")
    print(f"Top-K: {top_k}")
    print()
    
    # Initialize LLM judge
    judge = LLMJudge()
    print()
    
    # Load queries
    queries = []
    with open(queries_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                'query_id': row['query_id'],
                'query': row['query']
            })
    
    print(f"Loaded {len(queries)} queries")
    print()
    
    # Evaluate each query
    results = []
    total_precision = 0.0
    queries_with_zero_relevant = 0
    queries_with_all_relevant = 0
    
    start_time = time.time()
    
    for i, query_data in enumerate(queries):
        query_id = query_data['query_id']
        query = query_data['query']
        
        print(f"[{i+1}/{len(queries)}] Query {query_id}: {query[:60]}...")
        
        # Query API
        api_response = query_api(query, api_endpoint, top_k=top_k)
        
        if 'error' in api_response:
            print(f"  ✗ Query failed: {api_response['error']}")
            results.append({
                'query_id': query_id,
                'query': query,
                'error': api_response['error'],
                'precision': 0.0,
                'relevant_count': 0,
                'total_results': 0
            })
            continue
        
        papers = api_response.get('results', [])
        
        if not papers:
            print(f"  ⚠ No results returned")
            results.append({
                'query_id': query_id,
                'query': query,
                'precision': 0.0,
                'relevant_count': 0,
                'total_results': 0,
                'papers': []
            })
            continue
        
        print(f"  Retrieved {len(papers)} results, judging relevance...")
        
        # Judge relevance for each paper
        paper_judgments = []
        relevant_count = 0
        
        for rank, paper in enumerate(papers, 1):
            judgment = judge.judge_relevance(
                query,
                paper.get('title', ''),
                paper.get('abstract', '')
            )
            
            is_relevant = judgment['relevant']
            if is_relevant:
                relevant_count += 1
            
            paper_judgments.append({
                'rank': rank,
                'doc_id': paper.get('doc_id', ''),
                'title': paper.get('title', ''),
                'score': paper.get('score', 0.0),
                'relevant': is_relevant,
                'reasoning': judgment['reasoning']
            })
        
        # Calculate precision
        precision = relevant_count / len(papers) if papers else 0.0
        total_precision += precision
        
        if relevant_count == 0:
            queries_with_zero_relevant += 1
        elif relevant_count == len(papers):
            queries_with_all_relevant += 1
        
        print(f"  Precision@{top_k}: {precision:.3f} ({relevant_count}/{len(papers)} relevant)")
        
        results.append({
            'query_id': query_id,
            'query': query,
            'precision': precision,
            'relevant_count': relevant_count,
            'total_results': len(papers),
            'papers': paper_judgments
        })
        
        # Small delay to avoid overwhelming API
        time.sleep(0.5)
    
    elapsed_time = time.time() - start_time
    
    # Calculate overall metrics
    avg_precision = total_precision / len(queries) if queries else 0.0
    
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'queries_ml_100',
        'num_queries': len(queries),
        'top_k': top_k,
        'avg_precision_at_10': avg_precision,
        'queries_with_zero_relevant': queries_with_zero_relevant,
        'queries_with_all_relevant': queries_with_all_relevant,
        'elapsed_time_seconds': elapsed_time,
        'api_endpoint': api_endpoint,
        'detailed_results': results
    }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Total queries: {len(queries)}")
    print(f"Average Precision@{top_k}: {avg_precision:.3f}")
    print(f"Queries with 0 relevant: {queries_with_zero_relevant}")
    print(f"Queries with all relevant: {queries_with_all_relevant}")
    print(f"Elapsed time: {elapsed_time:.1f}s")
    print(f"Saved to: {output_file}")
    print()
    
    return evaluation_results


def main():
    """Run precision evaluation."""
    # Paths
    eval_dir = Path(__file__).parent.parent
    data_dir = eval_dir / "data"
    results_dir = eval_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    queries_csv = data_dir / "queries_ml_100.csv"
    output_file = results_dir / "precision_results.json"
    
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
    evaluate_precision(
        str(queries_csv),
        api_endpoint,
        str(output_file),
        top_k=10
    )


if __name__ == "__main__":
    main()
