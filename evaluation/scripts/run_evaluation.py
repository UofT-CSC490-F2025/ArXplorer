"""
Main orchestrator for running complete evaluation pipeline.
Runs both precision and recall evaluations, combines results.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import get_api_endpoint_from_terraform, test_api_connection
from precision_evaluator import evaluate_precision
from recall_evaluator import evaluate_recall_and_mrr


def print_summary(precision_results: dict, recall_results: dict):
    """Print combined evaluation summary."""
    print()
    print("=" * 70)
    print(" " * 20 + "EVALUATION SUMMARY")
    print("=" * 70)
    print()
    
    print("PRECISION@10 (queries_ml_100)")
    print("-" * 70)
    print(f"  Queries evaluated:      {precision_results['num_queries']}")
    print(f"  Average Precision@10:   {precision_results['avg_precision_at_10']:.3f}")
    print(f"  Zero relevant:          {precision_results['queries_with_zero_relevant']}")
    print(f"  All relevant:           {precision_results['queries_with_all_relevant']}")
    print(f"  Time:                   {precision_results['elapsed_time_seconds']:.1f}s")
    print()
    
    print("RECALL@10 & MRR (canon_papers_60)")
    print("-" * 70)
    print(f"  Canonical papers:       {recall_results['num_canonical_papers']}")
    print(f"  Total queries:          {recall_results['total_queries']}")
    print(f"  Recall@10:              {recall_results['recall_at_10']:.3f} ({recall_results['papers_found_any_variant']}/{recall_results['num_canonical_papers']} papers found)")
    print(f"  MRR:                    {recall_results['mrr']:.3f}")
    print(f"  Query-level recall:     {recall_results['query_level_recall']:.3f}")
    print(f"  Time:                   {recall_results['elapsed_time_seconds']:.1f}s")
    print()
    
    print("OVERALL")
    print("-" * 70)
    total_time = precision_results['elapsed_time_seconds'] + recall_results['elapsed_time_seconds']
    total_queries = precision_results['num_queries'] + recall_results['total_queries']
    print(f"  Total queries executed: {total_queries}")
    print(f"  Total time:             {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()
    print("=" * 70)
    print()


def main():
    """Run complete evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Run ArXplorer evaluation")
    parser.add_argument(
        "--api-endpoint",
        help="API endpoint URL (auto-detected from terraform if not provided)"
    )
    parser.add_argument(
        "--skip-precision",
        action="store_true",
        help="Skip Precision@10 evaluation"
    )
    parser.add_argument(
        "--skip-recall",
        action="store_true",
        help="Skip Recall@10 and MRR evaluation"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print(" " * 20 + "ArXplorer Evaluation")
    print("=" * 70)
    print()
    
    # Setup paths
    eval_dir = Path(__file__).parent.parent
    data_dir = eval_dir / "data"
    results_dir = eval_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Get API endpoint
    api_endpoint = args.api_endpoint
    if not api_endpoint:
        api_endpoint = get_api_endpoint_from_terraform()
    
    if not api_endpoint:
        print("Error: Could not determine API endpoint")
        print("Options:")
        print("  1. Set ARXPLORER_API_ENDPOINT environment variable")
        print("  2. Use --api-endpoint argument")
        print("  3. Run from repository with terraform/outputs.tf")
        return 1
    
    print(f"API Endpoint: {api_endpoint}")
    print()
    
    # Test connection
    print("Testing API connection...")
    if not test_api_connection(api_endpoint):
        print("Error: Could not connect to API")
        print(f"  Endpoint: {api_endpoint}")
        print("  Make sure the API is running and accessible")
        return 1
    
    print("✓ API connection successful")
    print()
    
    # Run evaluations
    precision_results = None
    recall_results = None
    
    # Precision evaluation
    if not args.skip_precision:
        print()
        print("=" * 70)
        print("Starting Precision@10 Evaluation")
        print("=" * 70)
        print()
        
        queries_csv = data_dir / "queries_ml_100.csv"
        precision_output = results_dir / "precision_results.json"
        
        if not queries_csv.exists():
            print(f"Error: Queries file not found: {queries_csv}")
            return 1
        
        precision_results = evaluate_precision(
            str(queries_csv),
            api_endpoint,
            str(precision_output),
            top_k=args.top_k
        )
    
    # Recall evaluation
    if not args.skip_recall:
        print()
        print("=" * 70)
        print("Starting Recall@10 and MRR Evaluation")
        print("=" * 70)
        print()
        
        canonical_queries = data_dir / "canonical_queries.json"
        recall_output = results_dir / "recall_results.json"
        
        if not canonical_queries.exists():
            print(f"Error: Canonical queries not found: {canonical_queries}")
            print("Run query_generator.py first to generate queries")
            return 1
        
        recall_results = evaluate_recall_and_mrr(
            str(canonical_queries),
            api_endpoint,
            str(recall_output),
            top_k=args.top_k
        )
    
    # Combined summary
    if precision_results and recall_results:
        # Save combined summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = results_dir / f"summary_{timestamp}.json"
        
        combined = {
            'timestamp': datetime.now().isoformat(),
            'api_endpoint': api_endpoint,
            'top_k': args.top_k,
            'precision': {
                'avg_precision_at_10': precision_results['avg_precision_at_10'],
                'num_queries': precision_results['num_queries'],
                'queries_with_zero_relevant': precision_results['queries_with_zero_relevant'],
                'queries_with_all_relevant': precision_results['queries_with_all_relevant']
            },
            'recall': {
                'recall_at_10': recall_results['recall_at_10'],
                'mrr': recall_results['mrr'],
                'num_canonical_papers': recall_results['num_canonical_papers'],
                'papers_found': recall_results['papers_found_any_variant'],
                'query_level_recall': recall_results['query_level_recall']
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        
        print_summary(precision_results, recall_results)
        print(f"Combined summary saved to: {summary_file}")
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
