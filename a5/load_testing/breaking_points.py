"""
Assignment A5 Part 2 - Breaking Point Attack Scenarios
Systematic stress testing to find application vulnerabilities

This module implements specific attack scenarios designed to break
different components of the ArXplorer system.
"""

import threading
import time
import psutil
import sys
import os
import asyncio
import concurrent.futures
from datetime import datetime
import numpy as np
import json

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

# For testing without full ArXplorer setup, we'll create mock classes
try:
    from src.core.pipeline import EmbeddingGenerator, VectorIndexer
    from src.core.schemas import PipelineConfig, ProcessedPaper
    ARXPLORER_AVAILABLE = True
except ImportError:
    print("⚠️ ArXplorer modules not available, using mock implementations")
    ARXPLORER_AVAILABLE = False
    
    # Mock classes for testing the framework
    class PipelineConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
                
    class ProcessedPaper:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
                
    class EmbeddingGenerator:
        def __init__(self, config):
            self.config = config
            
        def generate_embeddings(self, paper):
            # Mock embedding generation with some CPU work
            time.sleep(0.01)  # Simulate processing time
            return np.random.random(768)  # Mock 768D embedding
            
    class VectorIndexer:
        def __init__(self, config):
            self.config = config
            
        def build_index(self, embeddings):
            # Mock index building with some processing
            time.sleep(len(embeddings) * 0.001)  # Scale with number of embeddings
            return f"Mock index with {len(embeddings)} vectors"


class SystemMonitor:
    """Monitor system resources during attacks"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = []
        
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return stats"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        return self.stats
        
    def _monitor_loop(self):
        """Monitor loop - collect resource usage"""
        while self.monitoring:
            try:
                stats = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                    'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                    'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
                }
                self.stats.append(stats)
                time.sleep(1)  # Collect stats every second
            except Exception as e:
                print(f"Monitoring error: {e}")
                break


class AttackScenario:
    """Base class for attack scenarios"""
    
    def __init__(self, name):
        self.name = name
        self.monitor = SystemMonitor()
        self.results = {}
        
    def run_attack(self):
        """Run the attack scenario"""
        print(f"\n🚨 STARTING ATTACK: {self.name}")
        print("=" * 50)
        
        # Start monitoring
        self.monitor.start_monitoring()
        start_time = time.time()
        
        try:
            # Run the actual attack
            self.execute_attack()
            
        except Exception as e:
            print(f"💥 ATTACK CAUSED EXCEPTION: {e}")
            self.results['exception'] = str(e)
            
        finally:
            # Stop monitoring and collect results
            duration = time.time() - start_time
            stats = self.monitor.stop_monitoring()
            
            self.results.update({
                'duration_seconds': duration,
                'system_stats': stats,
                'peak_cpu': max((s['cpu_percent'] for s in stats), default=0),
                'peak_memory': max((s['memory_percent'] for s in stats), default=0),
                'peak_memory_gb': max((s['memory_used_gb'] for s in stats), default=0)
            })
            
            self._print_results()
            return self.results
            
    def execute_attack(self):
        """Override in subclasses"""
        raise NotImplementedError
        
    def _print_results(self):
        """Print attack results"""
        print(f"\n📊 ATTACK RESULTS: {self.name}")
        print("-" * 30)
        print(f"Duration: {self.results['duration_seconds']:.2f} seconds")
        print(f"Peak CPU: {self.results['peak_cpu']:.1f}%")
        print(f"Peak Memory: {self.results['peak_memory']:.1f}% ({self.results['peak_memory_gb']:.2f} GB)")
        
        if 'exception' in self.results:
            print(f"BROKE THE SYSTEM: {self.results['exception']}")
        else:
            print("System survived the attack")


class EmbeddingBombAttack(AttackScenario):
    """Attack: Overwhelm embedding generation with concurrent requests"""
    
    def __init__(self, num_threads=10, papers_per_thread=20):
        super().__init__("Embedding Generation Bomb")
        self.num_threads = num_threads
        self.papers_per_thread = papers_per_thread
        
    def execute_attack(self):
        """Execute embedding generation attack"""
        print(f"🎯 Launching {self.num_threads} threads with {self.papers_per_thread} papers each")
        
        # Initialize embedding generator
        config = PipelineConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        generator = EmbeddingGenerator(config)
        
        # Create sample papers
        papers = []
        for i in range(self.papers_per_thread):
            paper = ProcessedPaper(
                arxiv_id=f"attack.{i}",
                cleaned_title="Attack Paper: Very Long Title " * 10,  # Make it long
                cleaned_abstract="This is a very long abstract " * 50,  # 50x repetition
                extracted_keywords=["attack", "test", "embedding"],
                word_count=500,
                readability_score=12.0
            )
            papers.append(paper)
        
        def embedding_worker(thread_id):
            """Worker function for each thread"""
            print(f"   Thread {thread_id}: Starting embedding generation")
            results = []
            
            for i, paper in enumerate(papers):
                try:
                    start = time.time()
                    embedding = generator.generate_embeddings(paper)
                    duration = time.time() - start
                    results.append({
                        'paper_id': paper.arxiv_id,
                        'duration': duration,
                        'success': True
                    })
                    
                    if i % 5 == 0:  # Progress update every 5 papers
                        print(f"   Thread {thread_id}: Processed {i+1}/{len(papers)} papers")
                        
                except Exception as e:
                    results.append({
                        'paper_id': paper.arxiv_id,
                        'duration': 0,
                        'success': False,
                        'error': str(e)
                    })
                    
            print(f"   Thread {thread_id}: Completed")
            return results
        
        # Launch concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(embedding_worker, i) 
                for i in range(self.num_threads)
            ]
            
            # Wait for completion
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"Thread failed: {e}")
        
        # Calculate statistics
        successful = [r for r in all_results if r['success']]
        failed = [r for r in all_results if not r['success']]
        
        if successful:
            avg_duration = sum(r['duration'] for r in successful) / len(successful)
            max_duration = max(r['duration'] for r in successful)
        else:
            avg_duration = 0
            max_duration = 0
            
        self.results.update({
            'total_papers': len(all_results),
            'successful_embeddings': len(successful),
            'failed_embeddings': len(failed),
            'avg_embedding_time': avg_duration,
            'max_embedding_time': max_duration,
            'success_rate': len(successful) / len(all_results) if all_results else 0
        })
        
        print(f"✅ Processed {len(successful)}/{len(all_results)} papers successfully")
        print(f"⚡ Average embedding time: {avg_duration:.3f}s")
        print(f"⏰ Max embedding time: {max_duration:.3f}s")


class MemoryExhaustionAttack(AttackScenario):
    """Attack: Exhaust available system memory"""
    
    def __init__(self, target_gb=2):
        super().__init__(f"Memory Exhaustion ({target_gb}GB target)")
        self.target_gb = target_gb
        
    def execute_attack(self):
        """Execute memory exhaustion attack"""
        print(f"🎯 Attempting to allocate {self.target_gb}GB of memory")
        
        memory_hogs = []
        allocated_gb = 0
        chunk_size_mb = 100  # Allocate 100MB chunks
        
        try:
            while allocated_gb < self.target_gb:
                # Allocate 100MB array
                chunk = np.random.random((chunk_size_mb * 1024 * 1024 // 8,))  # 8 bytes per float64
                memory_hogs.append(chunk)
                
                allocated_gb += chunk_size_mb / 1024
                print(f"   Allocated: {allocated_gb:.2f}GB / {self.target_gb}GB")
                
                # Check if we're approaching system limits
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 90:
                    print(f"🚨 Memory usage at {memory_info.percent:.1f}% - stopping attack")
                    break
                    
                time.sleep(0.1)  # Brief pause
                
        except MemoryError:
            print(f"💥 MemoryError reached at {allocated_gb:.2f}GB")
            raise
            
        self.results.update({
            'allocated_gb': allocated_gb,
            'chunks_allocated': len(memory_hogs),
            'target_reached': allocated_gb >= self.target_gb
        })


class CPUExhaustionAttack(AttackScenario):
    """Attack: Exhaust CPU resources"""
    
    def __init__(self, num_threads=None, duration_seconds=30):
        if num_threads is None:
            num_threads = psutil.cpu_count()
        super().__init__(f"CPU Exhaustion ({num_threads} threads)")
        self.num_threads = num_threads
        self.duration_seconds = duration_seconds
        
    def execute_attack(self):
        """Execute CPU exhaustion attack"""
        print(f"🎯 Launching {self.num_threads} CPU-intensive threads for {self.duration_seconds}s")
        
        def cpu_worker(thread_id):
            """CPU-intensive worker function"""
            print(f"   Thread {thread_id}: Starting CPU burn")
            
            start_time = time.time()
            iterations = 0
            
            while time.time() - start_time < self.duration_seconds:
                # CPU-intensive computation - matrix multiplication
                a = np.random.random((200, 200))
                b = np.random.random((200, 200))
                c = np.dot(a, b)  # Expensive matrix multiplication
                iterations += 1
                
            print(f"   Thread {thread_id}: Completed {iterations} iterations")
            return iterations
        
        # Launch CPU threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(cpu_worker, i) 
                for i in range(self.num_threads)
            ]
            
            # Wait for completion
            total_iterations = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    iterations = future.result()
                    total_iterations += iterations
                except Exception as e:
                    print(f"CPU thread failed: {e}")
        
        self.results.update({
            'total_iterations': total_iterations,
            'threads_used': self.num_threads,
            'avg_iterations_per_thread': total_iterations / self.num_threads
        })
        
        print(f"✅ Completed {total_iterations} total matrix operations")


class IndexBuildingBombAttack(AttackScenario):
    """Attack: Overwhelm FAISS index building"""
    
    def __init__(self, num_papers=1000):
        super().__init__(f"Index Building Bomb ({num_papers} papers)")
        self.num_papers = num_papers
        
    def execute_attack(self):
        """Execute index building attack"""
        print(f"🎯 Building FAISS index with {self.num_papers} papers")
        
        # Initialize components
        config = PipelineConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            faiss_index_type="IVF",
            n_clusters=min(1024, self.num_papers // 10)  # Reasonable clusters
        )
        
        generator = EmbeddingGenerator(config)
        indexer = VectorIndexer(config)
        
        # Generate papers and embeddings
        print("   Generating papers and embeddings...")
        embeddings = []
        
        for i in range(self.num_papers):
            paper = ProcessedPaper(
                arxiv_id=f"index.{i:05d}",
                cleaned_title=f"Index Test Paper {i}",
                cleaned_abstract=f"Abstract for paper {i} with lots of content to embed",
                extracted_keywords=["test", "index", "faiss"],
                word_count=200,
                readability_score=12.0
            )
            
            try:
                embedding = generator.generate_embeddings(paper)
                embeddings.append(embedding)
                
                if i % 100 == 0:  # Progress every 100 papers
                    print(f"   Generated {i+1}/{self.num_papers} embeddings")
                    
            except Exception as e:
                print(f"Failed to generate embedding for paper {i}: {e}")
        
        print(f"   Building FAISS index with {len(embeddings)} embeddings...")
        
        # Build the index (this should be the breaking point)
        start_time = time.time()
        indexer.build_index(embeddings)
        build_time = time.time() - start_time
        
        self.results.update({
            'papers_processed': len(embeddings),
            'index_build_time': build_time,
            'papers_per_second': len(embeddings) / build_time if build_time > 0 else 0
        })
        
        print(f"✅ Successfully built index in {build_time:.2f} seconds")
        print(f"⚡ Rate: {self.results['papers_per_second']:.1f} papers/second")


def run_all_attacks():
    """Run all attack scenarios systematically"""
    
    print("ARXPLORER BREAKING POINT ANALYSIS")
    print("Assignment A5 Part 2 - Red Team vs Blue Team")
    print("=" * 60)
    
    attacks = [
        EmbeddingBombAttack(num_threads=5, papers_per_thread=10),  # Start conservative
        MemoryExhaustionAttack(target_gb=1),  # 1GB memory attack
        CPUExhaustionAttack(duration_seconds=20),  # 20 second CPU attack  
        IndexBuildingBombAttack(num_papers=500),  # 500 paper index
    ]
    
    all_results = {}
    
    for attack in attacks:
        try:
            result = attack.run_attack()
            all_results[attack.name] = result
            
            # Cool-down period between attacks
            print("\nCool-down period (5 seconds)...")
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\nAttack sequence interrupted by user")
            break
        except Exception as e:
            print(f"\nAttack failed: {e}")
            all_results[attack.name] = {'exception': str(e)}
    
    # Generate final report
    print("\n" + "=" * 60)
    print("FINAL BREAKING POINT ANALYSIS")
    print("=" * 60)
    
    for name, results in all_results.items():
        print(f"\n{name}:")
        if 'exception' in results:
            print(f"   BROKE THE SYSTEM: {results['exception']}")
        else:
            print(f"   System survived")
            if 'peak_cpu' in results:
                print(f"   Peak CPU: {results['peak_cpu']:.1f}%")
                print(f"   Peak Memory: {results['peak_memory']:.1f}%")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"breaking_point_analysis_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    # Check system capabilities first
    print("SYSTEM INFORMATION:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"   Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    run_all_attacks()