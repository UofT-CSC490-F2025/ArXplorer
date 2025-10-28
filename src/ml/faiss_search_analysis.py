#!/usr/bin/env python3
"""
FAISS Search Analysis: K-Parameter Tuning and Cluster Search Mechanics
Demonstrates how FAISS actually searches within clusters and hyperparameter optimization

This module addresses critical gaps in the ArXplorer pipeline:
1. Hyperparameter tuning for k, nprobe, and nlist
2. Proper FAISS clustering search implementation
3. Performance benchmarking and evaluation metrics
4. Within-cluster search mechanics demonstration

Usage:
    python faiss_search_analysis.py --dataset-size 10000 --run-full-analysis
    python faiss_search_analysis.py --quick-demo
"""

import numpy as np
import faiss
import time
from typing import List, Tuple, Dict, Any, Optional
import json
import argparse
import logging
from pathlib import Path

# Import our existing schemas
from schemas import PaperEmbedding
from pipeline import VectorIndexer, PipelineConfig

class FAISSSearchAnalysis:
    """Analyze FAISS search mechanics and hyperparameter tuning"""
    
    def __init__(self, num_papers: int = 10000, embedding_dim: int = 768):
        """Generate synthetic dataset for analysis"""
        
        np.random.seed(42)
        self.embeddings = np.random.normal(0, 1, (num_papers, embedding_dim)).astype(np.float32)
        faiss.normalize_L2(self.embeddings)
        
        # Create synthetic ground truth relevance scores
        self.relevance_scores = np.random.exponential(0.3, num_papers)
        
        # Generate synthetic paper IDs
        self.paper_ids = [f"arxiv:{1000+i}" for i in range(num_papers)]
        
        self.num_papers = num_papers
        self.embedding_dim = embedding_dim
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        print(f"🔬 Created {num_papers} synthetic papers for FAISS analysis")
    
    def demonstrate_faiss_search_mechanics(self):
        """Show exactly how FAISS searches within clusters"""
        
        print("\n" + "="*60)
        print("🔍 FAISS SEARCH MECHANICS BREAKDOWN")
        print("="*60)
        
        # Setup IVF index
        nlist = 100  # Smaller for demonstration
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        
        # Train and add vectors
        print("🏋️  Training FAISS index with K-means clustering...")
        train_start = time.time()
        index.train(self.embeddings)
        train_time = time.time() - train_start
        
        index.add(self.embeddings)
        
        # Create a query
        query = np.random.normal(0, 1, (1, self.embedding_dim)).astype(np.float32)
        faiss.normalize_L2(query)
        
        print(f"📊 Index Setup:")
        print(f"   Papers: {self.num_papers:,}")
        print(f"   Clusters (nlist): {nlist}")
        print(f"   Avg papers per cluster: {self.num_papers/nlist:.1f}")
        print(f"   Training time: {train_time:.2f}s")
        
        # Test different nprobe values
        print(f"\n🎯 Search Process Analysis:")
        
        for nprobe in [1, 5, 10, 20]:
            index.nprobe = nprobe
            
            print(f"\n--- nprobe = {nprobe} ---")
            
            # Perform search
            start_time = time.time()
            scores, indices = index.search(query, k=10)
            search_time = time.time() - start_time
            
            # Calculate how many papers were actually examined
            papers_examined = min(nprobe * (self.num_papers // nlist), self.num_papers)
            
            print(f"   Clusters searched: {nprobe}")
            print(f"   Papers examined: ~{papers_examined:,} ({papers_examined/self.num_papers:.1%} of total)")
            print(f"   Search time: {search_time*1000:.2f}ms")
            print(f"   Top result score: {scores[0][0]:.4f}")
            
            # Show the actual search process
            if nprobe <= 10:  # Only for small nprobe values
                print(f"   🔎 What FAISS actually did:")
                print(f"      1. Found {nprobe} nearest cluster centroids")
                print(f"      2. Searched ~{papers_examined//nprobe} papers in each cluster")
                print(f"      3. Combined and ranked all candidates")
                print(f"      4. Returned top 10 results")
    
    def hyperparameter_tuning_analysis(self) -> Dict[str, Any]:
        """Comprehensive hyperparameter tuning for FAISS search"""
        
        print("\n" + "="*60)
        print("⚙️  HYPERPARAMETER TUNING ANALYSIS")
        print("="*60)
        
        # Parameters to test
        nlist_values = [50, 100, 256, 512, 1024] if self.num_papers >= 1024 else [16, 32, 64]
        nprobe_values = [1, 5, 10, 20, 50]
        k_values = [5, 10, 20, 50]
        
        results = {}
        
        # Generate test queries
        num_queries = 100
        queries = np.random.normal(0, 1, (num_queries, self.embedding_dim)).astype(np.float32)
        faiss.normalize_L2(queries)
        
        print(f"🧪 Testing combinations of:")
        print(f"   nlist (clusters): {nlist_values}")
        print(f"   nprobe (clusters searched): {nprobe_values}")
        print(f"   k (results returned): {k_values}")
        print(f"   Test queries: {num_queries}")
        
        # Create ground truth (flat index)
        print(f"📊 Creating ground truth with flat index...")
        flat_index = faiss.IndexFlatIP(self.embedding_dim)
        flat_index.add(self.embeddings)
        
        print(f"\n📊 Results:")
        print(f"{'nlist':<6} {'nprobe':<7} {'k':<3} {'Time(ms)':<9} {'Accuracy':<9} {'Recall@k':<9}")
        print("-" * 55)
        
        for nlist in nlist_values:
            if nlist > self.num_papers // 10:  # Skip if too many clusters
                continue
                
            # Create and train index
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            try:
                index.train(self.embeddings)
                index.add(self.embeddings)
            except Exception as e:
                self.logger.warning(f"Failed to create index with nlist={nlist}: {e}")
                continue
            
            for nprobe in nprobe_values:
                if nprobe > nlist:  # Can't search more clusters than exist
                    continue
                    
                index.nprobe = nprobe
                
                for k in k_values:
                    if k > self.num_papers:
                        continue
                        
                    # Perform search
                    start_time = time.time()
                    ivf_scores, ivf_indices = index.search(queries, k)
                    search_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Calculate accuracy vs ground truth
                    flat_scores, flat_indices = flat_index.search(queries, k)
                    
                    # Calculate recall@k
                    total_recall = 0
                    for i in range(num_queries):
                        flat_set = set(flat_indices[i])
                        ivf_set = set(ivf_indices[i])
                        recall = len(flat_set & ivf_set) / len(flat_set) if len(flat_set) > 0 else 0
                        total_recall += recall
                    
                    avg_recall = total_recall / num_queries
                    
                    # Calculate score similarity
                    score_similarity = np.mean([
                        np.corrcoef(flat_scores[i], ivf_scores[i])[0, 1] 
                        for i in range(num_queries) 
                        if not np.isnan(np.corrcoef(flat_scores[i], ivf_scores[i])[0, 1])
                    ])
                    
                    if np.isnan(score_similarity):
                        score_similarity = 0.0
                    
                    results[f"{nlist}_{nprobe}_{k}"] = {
                        'nlist': nlist,
                        'nprobe': nprobe,
                        'k': k,
                        'time_ms': search_time,
                        'score_similarity': score_similarity,
                        'recall': avg_recall,
                        'papers_examined': min(nprobe * (self.num_papers // nlist), self.num_papers)
                    }
                    
                    print(f"{nlist:<6} {nprobe:<7} {k:<3} {search_time:<9.2f} {score_similarity:<9.3f} {avg_recall:<9.3f}")
        
        return results
    
    def find_optimal_parameters(self, results: Dict) -> Dict[str, Any]:
        """Find optimal parameters based on different criteria"""
        
        print("\n" + "="*60)
        print("🎯 OPTIMAL PARAMETER ANALYSIS")
        print("="*60)
        
        if not results:
            print("❌ No results to analyze")
            return {}
        
        # Convert to list for analysis
        result_list = list(results.values())
        
        # Find best for different criteria
        best_speed = min(result_list, key=lambda x: x['time_ms'])
        best_accuracy = max(result_list, key=lambda x: x['score_similarity'])
        best_recall = max(result_list, key=lambda x: x['recall'])
        
        # Find best balanced (speed vs accuracy)
        for r in result_list:
            # Balance score: recall per ms (higher is better)
            r['balance_score'] = r['recall'] / (r['time_ms'] / 100) if r['time_ms'] > 0 else 0
        
        best_balanced = max(result_list, key=lambda x: x['balance_score'])
        
        print(f"🏆 OPTIMAL PARAMETERS:")
        print(f"\n⚡ Best Speed:")
        print(f"   nlist={best_speed['nlist']}, nprobe={best_speed['nprobe']}, k={best_speed['k']}")
        print(f"   Time: {best_speed['time_ms']:.2f}ms, Recall: {best_speed['recall']:.3f}")
        
        print(f"\n🎯 Best Score Accuracy:")
        print(f"   nlist={best_accuracy['nlist']}, nprobe={best_accuracy['nprobe']}, k={best_accuracy['k']}")
        print(f"   Time: {best_accuracy['time_ms']:.2f}ms, Score Similarity: {best_accuracy['score_similarity']:.3f}")
        
        print(f"\n📊 Best Recall:")
        print(f"   nlist={best_recall['nlist']}, nprobe={best_recall['nprobe']}, k={best_recall['k']}")
        print(f"   Time: {best_recall['time_ms']:.2f}ms, Recall: {best_recall['recall']:.3f}")
        
        print(f"\n⚖️  Best Balanced (Speed vs Accuracy):")
        print(f"   nlist={best_balanced['nlist']}, nprobe={best_balanced['nprobe']}, k={best_balanced['k']}")
        print(f"   Time: {best_balanced['time_ms']:.2f}ms, Recall: {best_balanced['recall']:.3f}")
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        print(f"\n💡 RECOMMENDATIONS:")
        for scenario, params in recommendations.items():
            print(f"   {scenario}: {params}")
        
        return {
            'best_speed': best_speed,
            'best_accuracy': best_accuracy,
            'best_recall': best_recall,
            'best_balanced': best_balanced,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self) -> Dict[str, str]:
        """Generate recommendations based on dataset size"""
        
        recommendations = {}
        
        if self.num_papers < 1000:
            recommendations["Small Dataset (<1K papers)"] = "Use Flat index (no clustering needed)"
        elif self.num_papers < 10000:
            recommendations["Medium Dataset (1K-10K papers)"] = "nlist=100, nprobe=10, k=10"
        elif self.num_papers < 100000:
            recommendations["Large Dataset (10K-100K papers)"] = "nlist=1024, nprobe=20, k=20"
        else:
            recommendations["Very Large Dataset (>100K papers)"] = "nlist=4096, nprobe=50, k=50"
        
        return recommendations
    
    def demonstrate_within_cluster_search(self):
        """Show exactly what happens within each cluster during search"""
        
        print("\n" + "="*60)
        print("🔍 WITHIN-CLUSTER SEARCH MECHANICS")
        print("="*60)
        
        # Create small example for illustration
        nlist = 4  # Just 4 clusters for clear demonstration
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        
        # Use subset of data for clarity
        subset_size = min(100, self.num_papers)
        subset_embeddings = self.embeddings[:subset_size]
        
        index.train(subset_embeddings)
        index.add(subset_embeddings)
        
        # Create query
        query = np.random.normal(0, 1, (1, self.embedding_dim)).astype(np.float32)
        faiss.normalize_L2(query)
        
        print(f"📊 Setup: {subset_size} papers in {nlist} clusters (~{subset_size//nlist} per cluster)")
        
        # Test with nprobe=2 (search 2 clusters)
        index.nprobe = 2
        k = 5
        
        print(f"\n🎯 Search Query with nprobe=2, k={k}:")
        
        # Get cluster assignments for demonstration
        cluster_scores, cluster_indices = index.quantizer.search(query, nlist)
        
        print(f"\n1️⃣  CLUSTER SELECTION PHASE:")
        print(f"   Query compared to {nlist} cluster centroids")
        for i in range(min(nlist, 4)):
            print(f"   Cluster {i}: similarity = {cluster_scores[0][i]:.4f}")
        
        print(f"\n   ✅ Selected top {index.nprobe} clusters: {cluster_indices[0][:index.nprobe]}")
        
        print(f"\n2️⃣  WITHIN-CLUSTER SEARCH PHASE:")
        print(f"   For each selected cluster:")
        print(f"   • Get ALL papers in that cluster")
        print(f"   • Calculate similarity with query vector") 
        print(f"   • Keep all candidates for global ranking")
        
        # Perform actual search
        scores, indices = index.search(query, k)
        
        print(f"\n3️⃣  FINAL RANKING PHASE:")
        print(f"   • Combine candidates from all {index.nprobe} clusters")
        print(f"   • Sort by similarity score globally")
        print(f"   • Return top {k} results")
        
        print(f"\n📋 Final Results:")
        for i in range(k):
            if i < len(indices[0]) and indices[0][i] >= 0:
                print(f"   Rank {i+1}: Paper {indices[0][i]} (score: {scores[0][i]:.4f})")
        
        # Show what we would miss with nprobe=1
        index.nprobe = 1
        scores_1, indices_1 = index.search(query, k)
        
        missed_papers = set(indices[0]) - set(indices_1[0])
        print(f"\n❌ With nprobe=1, we would miss {len(missed_papers)} papers from the top-{k}")
        if missed_papers:
            print(f"   Missed papers: {list(missed_papers)}")

def run_quick_demo():
    """Run a quick demonstration of FAISS search mechanics"""
    
    print("🚀 FAISS Search Mechanics - Quick Demo")
    print("="*50)
    
    analyzer = FAISSSearchAnalysis(num_papers=1000, embedding_dim=128)
    
    # Show basic search mechanics
    analyzer.demonstrate_faiss_search_mechanics()
    
    # Show within-cluster search
    analyzer.demonstrate_within_cluster_search()
    
    print(f"\n✅ Quick demo completed!")
    print(f"   Run with --run-full-analysis for comprehensive hyperparameter tuning")

def run_full_analysis(dataset_size: int, output_dir: str):
    """Run comprehensive FAISS analysis with hyperparameter tuning"""
    
    print(f"🔬 FAISS Comprehensive Analysis - {dataset_size:,} papers")
    print("="*60)
    
    analyzer = FAISSSearchAnalysis(num_papers=dataset_size)
    
    # 1. Show search mechanics
    analyzer.demonstrate_faiss_search_mechanics()
    
    # 2. Hyperparameter tuning
    results = analyzer.hyperparameter_tuning_analysis()
    
    # 3. Find optimal parameters
    optimal = analyzer.find_optimal_parameters(results)
    
    # 4. Show within-cluster search (only for smaller datasets)
    if dataset_size <= 1000:
        analyzer.demonstrate_within_cluster_search()
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"1. k (number of results) doesn't affect search speed significantly")
    print(f"2. nprobe vs recall is the main trade-off")
    print(f"3. nlist should be ~sqrt(num_papers) for optimal clustering")
    print(f"4. Within each cluster, ALL papers are examined")
    print(f"5. Final results come from combining candidates from all searched clusters")

def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description="FAISS Search Analysis and Hyperparameter Tuning")
    parser.add_argument("--dataset-size", type=int, default=10000,
                       help="Number of synthetic papers to generate for analysis")
    parser.add_argument("--quick-demo", action="store_true",
                       help="Run quick demonstration of search mechanics")
    parser.add_argument("--run-full-analysis", action="store_true",
                       help="Run comprehensive hyperparameter tuning analysis")
    parser.add_argument("--output-dir", type=str, default="./analysis_results",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.quick_demo:
        run_quick_demo()
    elif args.run_full_analysis:
        run_full_analysis(args.dataset_size, args.output_dir)
    else:
        print("Please specify either --quick-demo or --run-full-analysis")
        print("Use --help for more options")

if __name__ == "__main__":
    main()