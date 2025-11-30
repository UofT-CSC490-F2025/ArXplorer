"""
A5 Part 1 - Performance Profiling for ArXplorer
Profile key functions to identify performance bottlenecks
"""

import cProfile
import pstats
import io
import time
import numpy as np
import asyncio
from typing import Dict, List, Any
from pathlib import Path


class ArXplorerProfiler:
    """Performance profiler for ArXplorer pipeline components"""
    
    def __init__(self):
        self.results = {}
        
    def profile_function(self, func, *args, **kwargs):
        """Profile a single function and return statistics"""
        pr = cProfile.Profile()
        
        print(f"🔍 Profiling: {func.__name__}")
        print("-" * 40)
        
        # Run profiling
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        # Get statistics
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()
        
        # Store results
        self.results[func.__name__] = {
            'stats': s.getvalue(),
            'result': result
        }
        
        print(s.getvalue()[:500] + "...")
        print(f"✅ Completed profiling {func.__name__}\n")
        
        return result


# Mock versions of ArXplorer functions for profiling
class MockArXplorerFunctions:
    """Mock implementations of key ArXplorer functions for profiling"""
    
    @staticmethod
    def mock_embedding_generation(texts: List[str]) -> List[List[float]]:
        """
        Mock embedding generation (simulates SciBERT processing)
        This represents your EmbeddingGenerator.generate_embeddings()
        """
        print("🧠 Generating embeddings...")
        embeddings = []
        
        for text in texts:
            # Simulate tokenization overhead
            tokens = text.split()[:512]  # Max sequence length
            
            # Simulate embedding computation (matrix operations)
            embedding = np.random.normal(0, 1, 768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            # Simulate processing time
            time.sleep(0.001 * len(tokens))  # 1ms per token
            
            embeddings.append(embedding.tolist())
        
        return embeddings
    
    @staticmethod
    def mock_faiss_index_build(embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Mock FAISS index building
        This represents your VectorIndexer.build_index()
        """
        print("🏗️ Building FAISS index...")
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        n_papers, dimension = embeddings_array.shape
        
        if n_papers > 1000:
            # Simulate IVF index creation (expensive)
            n_clusters = min(int(np.sqrt(n_papers)), 1024)
            
            # Simulate k-means clustering overhead
            for _ in range(n_clusters):
                cluster_center = np.mean(embeddings_array[:10], axis=0)
                time.sleep(0.0001)  # Clustering overhead
        
        # Simulate index building
        time.sleep(0.01 * n_papers / 1000)  # Scale with dataset size
        
        return {
            'index_type': 'IVF' if n_papers > 1000 else 'Flat',
            'n_papers': n_papers,
            'dimension': dimension,
            'clusters': n_clusters if n_papers > 1000 else None
        }
    
    @staticmethod
    def mock_faiss_search(query_embedding: List[float], 
                         index_info: Dict[str, Any], 
                         k: int = 20) -> Dict[str, Any]:
        """
        Mock FAISS search operation
        This represents your VectorIndexer.search()
        """
        print(f"🔍 Searching FAISS index for top {k} results...")
        
        n_papers = index_info['n_papers']
        query = np.array(query_embedding, dtype=np.float32)
        
        # Simulate search overhead based on index type
        if index_info.get('index_type') == 'IVF':
            # IVF search: probe multiple clusters
            n_clusters = index_info.get('clusters', 100)
            nprobe = min(20, n_clusters)  # Search 20 clusters
            
            papers_per_cluster = n_papers // n_clusters
            papers_examined = nprobe * papers_per_cluster
            
            # Search overhead proportional to papers examined
            time.sleep(0.00001 * papers_examined)
        else:
            # Flat search: examine all papers
            time.sleep(0.00001 * n_papers)
        
        # Generate mock results
        scores = np.random.uniform(0.7, 1.0, k)
        indices = np.random.randint(0, n_papers, k)
        
        return {
            'scores': scores.tolist(),
            'indices': indices.tolist(),
            'papers_examined': papers_examined if index_info.get('index_type') == 'IVF' else n_papers
        }
    
    @staticmethod
    def mock_mongodb_batch_insert(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mock MongoDB batch insertion
        This represents your MongoDBManager batch operations
        """
        print(f"💾 Inserting {len(papers)} papers to MongoDB...")
        
        # Simulate network latency and insertion overhead
        base_latency = 0.01  # 10ms base latency
        per_doc_overhead = 0.001  # 1ms per document
        
        total_time = base_latency + (per_doc_overhead * len(papers))
        
        # Simulate batch size optimization
        optimal_batch_size = 100
        if len(papers) > optimal_batch_size:
            # Simulate multiple batch operations
            n_batches = (len(papers) + optimal_batch_size - 1) // optimal_batch_size
            total_time = n_batches * (base_latency + per_doc_overhead * optimal_batch_size)
        
        time.sleep(total_time)
        
        return {
            'inserted_count': len(papers),
            'time_taken': total_time,
            'batches_used': n_batches if len(papers) > optimal_batch_size else 1
        }
    
    @staticmethod
    def mock_text_processing_pipeline(raw_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Mock text processing pipeline
        This represents your TextProcessor operations
        """
        print(f"📝 Processing {len(raw_texts)} texts...")
        
        processed_papers = []
        
        for text in raw_texts:
            # Simulate text cleaning operations
            cleaned_text = text.lower()
            
            # Simulate keyword extraction (expensive regex operations)
            keywords = []
            for word in text.split():
                if len(word) > 6:  # Simulate keyword filtering
                    keywords.append(word.lower())
                time.sleep(0.0001)  # Per-word processing overhead
            
            # Simulate readability scoring
            word_count = len(text.split())
            readability_score = min(100, word_count / 10)
            time.sleep(0.001)  # Readability calculation overhead
            
            processed_papers.append({
                'cleaned_text': cleaned_text,
                'keywords': keywords[:10],  # Top 10 keywords
                'word_count': word_count,
                'readability_score': readability_score
            })
        
        return processed_papers


def run_profiling_suite():
    """Run comprehensive profiling of ArXplorer functions"""
    
    print("🚀 ArXplorer Performance Profiling Suite")
    print("=" * 60)
    
    profiler = ArXplorerProfiler()
    mock_functions = MockArXplorerFunctions()
    
    # Test data generation
    print("📊 Generating test data...")
    test_texts = [
        f"Sample research paper {i} about machine learning and natural language processing with various technical details." * 20
        for i in range(50)
    ]
    
    test_papers = [{'text': text, 'id': f'paper_{i}'} for i, text in enumerate(test_texts)]
    
    # 1. Profile text processing
    print("\n1️⃣ Profiling Text Processing Pipeline")
    processed = profiler.profile_function(
        mock_functions.mock_text_processing_pipeline,
        test_texts
    )
    
    # 2. Profile embedding generation
    print("\n2️⃣ Profiling Embedding Generation")
    embeddings = profiler.profile_function(
        mock_functions.mock_embedding_generation,
        test_texts[:20]  # Smaller batch for embeddings
    )
    
    # 3. Profile FAISS index building
    print("\n3️⃣ Profiling FAISS Index Building")
    index_info = profiler.profile_function(
        mock_functions.mock_faiss_index_build,
        embeddings * 100  # Scale up for realistic dataset
    )
    
    # 4. Profile FAISS search
    print("\n4️⃣ Profiling FAISS Search")
    search_results = profiler.profile_function(
        mock_functions.mock_faiss_search,
        embeddings[0],  # Query embedding
        index_info,
        20  # Top K results
    )
    
    # 5. Profile MongoDB operations
    print("\n5️⃣ Profiling MongoDB Batch Insert")
    mongodb_results = profiler.profile_function(
        mock_functions.mock_mongodb_batch_insert,
        test_papers
    )
    
    # Generate performance report
    print("\n📊 PERFORMANCE ANALYSIS COMPLETE")
    print("=" * 60)
    
    return profiler.results


def analyze_bottlenecks(profiling_results: Dict[str, Any]):
    """Analyze profiling results and identify bottlenecks"""
    
    print("\n🎯 PERFORMANCE BOTTLENECK ANALYSIS")
    print("=" * 50)
    
    recommendations = []
    
    print("\n🔍 Key Findings:")
    
    # Analyze each function
    for func_name, data in profiling_results.items():
        print(f"\n📈 {func_name}:")
        
        # Extract key metrics from stats
        stats_lines = data['stats'].split('\n')
        function_calls = 0
        total_time = 0.0
        
        for line in stats_lines:
            if 'function calls' in line:
                parts = line.split()
                if parts:
                    try:
                        function_calls = int(parts[0])
                    except:
                        pass
            elif 'seconds' in line:
                parts = line.split()
                for part in parts:
                    try:
                        total_time = float(part)
                        break
                    except:
                        pass
        
        print(f"   Function calls: {function_calls}")
        print(f"   Total time: {total_time:.3f}s")
        
        # Generate recommendations
        if 'embedding' in func_name.lower():
            recommendations.append(
                "🧠 Embedding Generation: Consider batch processing, GPU acceleration, or model caching"
            )
        elif 'faiss' in func_name.lower():
            recommendations.append(
                "🔍 FAISS Operations: Optimize index parameters (nlist, nprobe) and consider index caching"
            )
        elif 'mongodb' in func_name.lower():
            recommendations.append(
                "💾 MongoDB Operations: Implement connection pooling and optimize batch sizes"
            )
        elif 'text' in func_name.lower():
            recommendations.append(
                "📝 Text Processing: Use compiled regex, parallel processing, or pre-computed features"
            )
    
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Implement optimizations for highest-impact functions")
    print(f"2. Set up continuous profiling for production monitoring")
    print(f"3. Create performance benchmarks for regression testing")
    print(f"4. Consider async/await patterns for I/O bound operations")
    print(f"5. Implement caching strategies for expensive computations")


if __name__ == "__main__":
    # Run profiling
    results = run_profiling_suite()
    
    # Analyze results
    analyze_bottlenecks(results)
    
    print(f"\n✅ A5 Part 1 Profiling Complete!")
    print(f"📁 Results saved for PR: a5-part1")