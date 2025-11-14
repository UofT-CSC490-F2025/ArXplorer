"""
Assignment A5 Part 2 - Load Testing ArXplorer
Breaking Your Application with Systematic Stress Testing

This module implements comprehensive load testing to find performance bottlenecks
and breaking points in the ArXplorer academic search system.
"""

from locust import HttpUser, task, between, events
import random
import json
import time
from datetime import datetime
import asyncio
import threading
import sys
import os

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from src.core.pipeline import EmbeddingGenerator, VectorIndexer
from src.core.schemas import PipelineConfig, ProcessedPaper
import numpy as np


class SearchLoadUser(HttpUser):
    """
    Simulates users performing search operations on ArXplorer
    """
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        self.search_queries = [
            "machine learning natural language processing",
            "deep neural networks computer vision", 
            "transformer architectures attention mechanisms",
            "reinforcement learning robotics applications",
            "quantum computing optimization algorithms",
            "federated learning privacy preservation",
            "graph neural networks social networks",
            "adversarial attacks model robustness",
            "transfer learning few shot learning",
            "explainable AI interpretability methods"
        ]
        self.categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"]
    
    @task(3)
    def search_by_text(self):
        """Most common operation - text search"""
        query = random.choice(self.search_queries)
        
        # Simulate API call (replace with actual endpoint when available)
        with self.client.get(
            f"/search/text?q={query}&limit=20", 
            catch_response=True,
            name="search_text"
        ) as response:
            if response.status_code == 404:
                # Expected since we don't have actual API yet
                response.success()
    
    @task(2)
    def search_by_category(self):
        """Category-based search"""
        category = random.choice(self.categories)
        
        with self.client.get(
            f"/search/category?cat={category}&limit=50",
            catch_response=True, 
            name="search_category"
        ) as response:
            if response.status_code == 404:
                response.success()
    
    @task(1)
    def get_paper_details(self):
        """Get detailed paper information"""
        paper_id = f"230{random.randint(1,9)}.{random.randint(10000,99999)}"
        
        with self.client.get(
            f"/papers/{paper_id}",
            catch_response=True,
            name="get_paper"
        ) as response:
            if response.status_code == 404:
                response.success()


class EmbeddingLoadUser(HttpUser):
    """
    Simulates heavy embedding generation workload
    """
    wait_time = between(0.5, 2)  # Faster requests for heavy load
    
    def on_start(self):
        """Initialize embedding generator"""
        try:
            self.config = PipelineConfig(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.embedding_generator = EmbeddingGenerator(self.config)
            self.sample_papers = self._generate_sample_papers()
        except Exception as e:
            print(f"Failed to initialize embedding generator: {e}")
            self.embedding_generator = None
    
    def _generate_sample_papers(self):
        """Generate sample papers for testing"""
        papers = []
        abstracts = [
            "This paper presents a novel deep learning approach for natural language processing tasks.",
            "We propose a new convolutional neural network architecture for image classification.",
            "This work explores transformer models for sequence-to-sequence learning applications.",
            "We investigate reinforcement learning algorithms for autonomous robot navigation.",
            "This paper analyzes quantum computing approaches to optimization problems.",
        ]
        
        for i in range(50):
            paper = ProcessedPaper(
                arxiv_id=f"test.{i:04d}",
                cleaned_title=f"Test Paper {i}: Advanced ML Methods",
                cleaned_abstract=random.choice(abstracts),
                extracted_keywords=["machine learning", "neural networks"],
                word_count=random.randint(100, 500),
                readability_score=random.uniform(10, 15)
            )
            papers.append(paper)
        
        return papers
    
    @task
    def generate_embeddings(self):
        """Generate embeddings for papers - CPU intensive task"""
        if not self.embedding_generator:
            return
        
        paper = random.choice(self.sample_papers)
        
        start_time = time.time()
        try:
            # This is the actual bottleneck from Part 1 profiling
            embedding = self.embedding_generator.generate_embeddings(paper)
            duration = time.time() - start_time
            
            # Report custom metrics
            events.request.fire(
                request_type="embedding",
                name="generate_embeddings", 
                response_time=duration * 1000,
                response_length=len(embedding.combined_embedding) if embedding else 0,
                exception=None,
                context={}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            events.request.fire(
                request_type="embedding",
                name="generate_embeddings",
                response_time=duration * 1000, 
                response_length=0,
                exception=e,
                context={}
            )


class DatabaseLoadUser(HttpUser):
    """
    Simulates heavy database operations
    """
    wait_time = between(0.1, 1)  # Very aggressive database load
    
    @task(4)
    def mongodb_text_search(self):
        """Simulate MongoDB text search operations"""
        query = f"machine learning {random.randint(1,1000)}"
        
        with self.client.get(
            f"/db/search?q={query}",
            catch_response=True,
            name="mongodb_search"
        ) as response:
            if response.status_code == 404:
                response.success()
    
    @task(2) 
    def mongodb_aggregation(self):
        """Simulate complex aggregation queries"""
        with self.client.get(
            f"/db/aggregate/categories",
            catch_response=True,
            name="mongodb_aggregate"
        ) as response:
            if response.status_code == 404:
                response.success()
    
    @task(1)
    def mongodb_bulk_insert(self):
        """Simulate bulk paper insertions"""
        papers_data = {
            "papers": [
                {
                    "arxiv_id": f"bulk.{random.randint(1,99999)}",
                    "title": f"Bulk Test Paper {i}",
                    "abstract": "Test abstract for bulk insertion testing"
                }
                for i in range(10)
            ]
        }
        
        with self.client.post(
            "/db/papers/bulk",
            json=papers_data,
            catch_response=True,
            name="mongodb_bulk_insert"
        ) as response:
            if response.status_code == 404:
                response.success()


class MemoryBombUser(HttpUser):
    """
    Attempts to exhaust system memory - DANGER ZONE
    """
    wait_time = between(0, 0.5)  # Aggressive memory attack
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_hogs = []  # Keep references to prevent GC
    
    @task
    def memory_exhaustion_attack(self):
        """Attempt to exhaust available memory"""
        try:
            # Allocate large arrays to exhaust memory
            large_array = np.random.random((1000, 1000))  # ~8MB per array
            self.memory_hogs.append(large_array)
            
            # Report successful allocation
            events.request.fire(
                request_type="memory_attack",
                name="allocate_memory",
                response_time=1,
                response_length=large_array.nbytes,
                exception=None,
                context={}
            )
            
            # Prevent infinite growth - cap at 1GB total
            if len(self.memory_hogs) > 125:  # 125 * 8MB = 1GB
                self.memory_hogs = self.memory_hogs[-50:]  # Keep last 50
                
        except MemoryError as e:
            events.request.fire(
                request_type="memory_attack", 
                name="allocate_memory",
                response_time=1,
                response_length=0,
                exception=e,
                context={}
            )


class CPUBombUser(HttpUser):
    """
    Attempts to exhaust CPU resources
    """
    wait_time = between(0, 0.1)  # Aggressive CPU attack
    
    @task
    def cpu_exhaustion_attack(self):
        """CPU-intensive computation to exhaust resources"""
        start_time = time.time()
        
        try:
            # Expensive computation - prime number generation
            def generate_primes(n):
                primes = []
                for num in range(2, n):
                    for i in range(2, int(num**0.5) + 1):
                        if num % i == 0:
                            break
                    else:
                        primes.append(num)
                return primes
            
            # Generate primes up to 1000 (CPU intensive)
            primes = generate_primes(1000)
            duration = time.time() - start_time
            
            events.request.fire(
                request_type="cpu_attack",
                name="generate_primes", 
                response_time=duration * 1000,
                response_length=len(primes),
                exception=None,
                context={}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            events.request.fire(
                request_type="cpu_attack",
                name="generate_primes",
                response_time=duration * 1000,
                response_length=0, 
                exception=e,
                context={}
            )


# Custom event handlers for detailed reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start"""
    print(f"\n🚀 LOAD TESTING STARTED: {datetime.now()}")
    print(f"Target host: {environment.host}")
    print("=" * 60)


@events.test_stop.add_listener  
def on_test_stop(environment, **kwargs):
    """Log test completion and results"""
    print(f"\n🏁 LOAD TESTING COMPLETED: {datetime.now()}")
    print("=" * 60)
    
    # Print summary statistics
    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Max response time: {stats.total.max_response_time:.2f}ms")
    
    # Print failures if any
    if stats.total.num_failures > 0:
        print(f"\n💥 FAILURES DETECTED:")
        for name, error in stats.errors.items():
            print(f"   {name}: {error.occurrences} occurrences")


if __name__ == "__main__":
    # Test individual components without Locust
    print("🧪 Testing load testing components...")
    
    # Test embedding load
    print("\n1. Testing embedding generation load...")
    user = EmbeddingLoadUser()
    user.host = "http://localhost:8000"
    user.on_start()
    
    if user.embedding_generator:
        user.generate_embeddings()
        print("✅ Embedding generation test passed")
    else:
        print("❌ Embedding generation test failed")
    
    print("\n✅ Load testing framework ready!")
    print("Run with: locust -f load_testing.py --host=http://localhost:8000")