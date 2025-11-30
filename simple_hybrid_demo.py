"""
Simple Hybrid Search Demo
A simplified demonstration of combining SciBERT + FAISS with A4 LLM Judge
"""

import asyncio
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class SearchMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"  
    PRECISE = "precise"


@dataclass
class MockSearchResult:
    """Mock search result for demonstration"""
    arxiv_id: str
    title: str
    abstract: str
    faiss_score: float
    llm_score: Optional[float] = None
    combined_score: Optional[float] = None
    confidence: float = 0.0


class MockHybridSearch:
    """Mock implementation to demonstrate hybrid search concepts"""
    
    def __init__(self):
        # Mock paper database
        self.papers = [
            {
                "id": "2301.12345",
                "title": "Deep Learning Approaches for Natural Language Processing",
                "abstract": "This paper presents novel deep learning architectures for NLP tasks including sentiment analysis, machine translation, and text summarization. Our approach achieves state-of-the-art results on multiple benchmarks.",
                "embedding": [random.random() for _ in range(10)]  # Mock 10D embedding
            },
            {
                "id": "2302.67890", 
                "title": "Computer Vision with Convolutional Neural Networks",
                "abstract": "We propose a new CNN architecture for image classification and object detection. The model demonstrates superior performance on ImageNet and COCO datasets with reduced computational requirements.",
                "embedding": [random.random() for _ in range(10)]
            },
            {
                "id": "2303.11111",
                "title": "Transformer Architectures for Sequence Modeling", 
                "abstract": "This work explores various transformer architectures for sequence-to-sequence tasks. We analyze attention mechanisms and propose improvements for long sequence processing.",
                "embedding": [random.random() for _ in range(10)]
            },
            {
                "id": "2304.22222",
                "title": "Quantum Computing Applications in Optimization",
                "abstract": "We investigate quantum algorithms for solving complex optimization problems. Our approach shows potential advantages over classical methods for specific problem classes.",
                "embedding": [random.random() for _ in range(10)]
            },
            {
                "id": "2305.33333",
                "title": "Machine Learning for Drug Discovery",
                "abstract": "This paper applies machine learning techniques to accelerate drug discovery processes. We demonstrate improved prediction accuracy for molecular properties and drug-target interactions.",
                "embedding": [random.random() for _ in range(10)]
            }
        ]
    
    async def search(self, query: str, mode: SearchMode = SearchMode.FAST, top_k: int = 3) -> Dict[str, Any]:
        """Perform mock hybrid search"""
        
        start_time = time.time()
        
        print(f"🔍 Searching for: '{query}' in {mode.value} mode")
        
        # Stage 1: Mock FAISS search (based on keyword similarity)
        faiss_start = time.time()
        faiss_results = self._mock_faiss_search(query)
        faiss_time = (time.time() - faiss_start) * 1000
        
        print(f"   📊 FAISS stage: {faiss_time:.1f}ms, found {len(faiss_results)} candidates")
        
        if mode == SearchMode.FAST:
            # Fast mode: just return FAISS results
            results = faiss_results[:top_k]
            total_time = (time.time() - start_time) * 1000
            
            print(f"   ⚡ Fast mode complete: {total_time:.1f}ms total")
            return {
                "mode": mode.value,
                "total_time_ms": total_time,
                "faiss_time_ms": faiss_time,
                "llm_time_ms": 0,
                "results": results
            }
        
        # Stage 2: Mock LLM enhancement
        llm_start = time.time()
        
        if mode == SearchMode.BALANCED:
            # Balanced: judge top candidates only
            candidates = faiss_results[:3]
            await asyncio.sleep(0.2)  # Simulate faster LLM processing
        else:  # PRECISE
            # Precise: judge all candidates  
            candidates = faiss_results
            await asyncio.sleep(0.8)  # Simulate thorough LLM processing
        
        enhanced_results = await self._mock_llm_judge(query, candidates)
        llm_time = (time.time() - llm_start) * 1000
        
        print(f"   🤖 LLM stage: {llm_time:.1f}ms, judged {len(candidates)} papers")
        
        # Combine scores
        final_results = self._combine_scores(enhanced_results)[:top_k]
        
        total_time = (time.time() - start_time) * 1000
        print(f"   ✅ {mode.value} mode complete: {total_time:.1f}ms total")
        
        return {
            "mode": mode.value,
            "total_time_ms": total_time, 
            "faiss_time_ms": faiss_time,
            "llm_time_ms": llm_time,
            "results": final_results
        }
    
    def _mock_faiss_search(self, query: str) -> List[MockSearchResult]:
        """Mock FAISS semantic search using keyword matching"""
        query_words = set(query.lower().split())
        
        scored_papers = []
        for paper in self.papers:
            # Mock similarity based on keyword overlap
            title_words = set(paper["title"].lower().split())
            abstract_words = set(paper["abstract"].lower().split())
            
            title_overlap = len(query_words.intersection(title_words))
            abstract_overlap = len(query_words.intersection(abstract_words))
            
            # Mock FAISS score (0-1, higher is better)
            faiss_score = min(1.0, (title_overlap * 0.3 + abstract_overlap * 0.1) + random.uniform(0.1, 0.4))
            
            result = MockSearchResult(
                arxiv_id=paper["id"],
                title=paper["title"],
                abstract=paper["abstract"],
                faiss_score=faiss_score,
                confidence=faiss_score
            )
            scored_papers.append(result)
        
        # Sort by FAISS score
        scored_papers.sort(key=lambda x: x.faiss_score, reverse=True)
        return scored_papers
    
    async def _mock_llm_judge(self, query: str, candidates: List[MockSearchResult]) -> List[MockSearchResult]:
        """Mock LLM relevance judgment"""
        
        for result in candidates:
            # Mock LLM scoring based on more sophisticated matching
            query_words = set(query.lower().split())
            content_words = set((result.title + " " + result.abstract).lower().split())
            
            # Mock semantic understanding (better than just keyword matching)
            semantic_overlap = len(query_words.intersection(content_words))
            context_bonus = 0.2 if "learning" in content_words and "learning" in query_words else 0
            domain_bonus = 0.1 if any(word in content_words for word in ["neural", "deep", "machine"]) else 0
            
            llm_score = min(1.0, semantic_overlap * 0.15 + context_bonus + domain_bonus + random.uniform(0.2, 0.6))
            result.llm_score = llm_score
        
        return candidates
    
    def _combine_scores(self, results: List[MockSearchResult]) -> List[MockSearchResult]:
        """Combine FAISS and LLM scores"""
        for result in results:
            if result.llm_score is not None:
                # Weighted combination: 70% FAISS, 30% LLM
                result.combined_score = 0.7 * result.faiss_score + 0.3 * result.llm_score
                result.confidence = min(result.faiss_score, result.llm_score)
            else:
                result.combined_score = result.faiss_score
                result.confidence = result.faiss_score * 0.8
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score or 0, reverse=True)
        return results


async def demo_modes():
    """Demonstrate different search modes"""
    
    print("🎯 ArXplorer Hybrid Search Demo")
    print("Combining SciBERT + FAISS with A4 LLM Judge")
    print("=" * 60)
    
    search = MockHybridSearch()
    query = "machine learning for natural language processing"
    
    # Test all modes
    modes = [SearchMode.FAST, SearchMode.BALANCED, SearchMode.PRECISE]
    results_by_mode = {}
    
    for mode in modes:
        print(f"\n🚀 Testing {mode.value.upper()} mode:")
        print("-" * 40)
        
        result = await search.search(query, mode=mode, top_k=3)
        results_by_mode[mode.value] = result
        
        # Show top results
        print(f"\n📋 Top results:")
        for i, paper in enumerate(result["results"][:3], 1):
            print(f"   {i}. {paper.title[:50]}...")
            print(f"      FAISS: {paper.faiss_score:.3f}", end="")
            if paper.llm_score is not None:
                print(f" | LLM: {paper.llm_score:.3f} | Combined: {paper.combined_score:.3f}")
            else:
                print()
            print(f"      Confidence: {paper.confidence:.3f}")
    
    # Performance summary
    print(f"\n\n📊 PERFORMANCE SUMMARY")
    print("=" * 40)
    
    print(f"{'Mode':<10} {'Time (ms)':<12} {'Speed Factor':<12} {'Stages'}")
    print("-" * 50)
    
    fast_time = results_by_mode["fast"]["total_time_ms"]
    
    for mode in ["fast", "balanced", "precise"]:
        result = results_by_mode[mode]
        time_ms = result["total_time_ms"] 
        speed_factor = time_ms / fast_time if fast_time > 0 else 1
        stages = "FAISS" if result["llm_time_ms"] == 0 else "FAISS+LLM"
        
        print(f"{mode:<10} {time_ms:<12.1f} {speed_factor:<12.1f}x {stages}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   ⚡ FAST: Interactive search, real-time apps (~{fast_time:.0f}ms)")
    print(f"   ⚖️  BALANCED: General purpose, good compromise (~{results_by_mode['balanced']['total_time_ms']:.0f}ms)")
    print(f"   🎯 PRECISE: Research quality, batch processing (~{results_by_mode['precise']['total_time_ms']:.0f}ms)")
    
    print(f"\n✨ INTEGRATION SUCCESS!")
    print(f"   🔬 Your SciBERT + FAISS: Production-ready semantic search")
    print(f"   🤖 A4 LLM Judge: Intelligent relevance filtering") 
    print(f"   🏆 Hybrid System: Best of both worlds!")


async def demo_explanation():
    """Demonstrate explanation capabilities"""
    
    print(f"\n\n💡 EXPLANATION DEMO")
    print("=" * 30)
    
    query = "deep learning for computer vision"
    paper = {
        "title": "Computer Vision with Convolutional Neural Networks",
        "abstract": "We propose a new CNN architecture for image classification and object detection. The model demonstrates superior performance on ImageNet and COCO datasets."
    }
    
    print(f"Query: '{query}'")
    print(f"Paper: {paper['title']}")
    print(f"\n🤖 LLM Judge Assessment:")
    
    # Mock explanation
    print(f"   Decision: YES") 
    print(f"   Relevance Score: 0.850")
    print(f"   Confidence: 0.920")
    print(f"   Reasoning: Strong semantic alignment - query mentions 'computer vision'")
    print(f"              and 'deep learning', paper discusses CNN architectures for")
    print(f"              image classification which directly addresses both concepts.")
    print(f"   Model Used: ensemble (Ollama + GRPO)")


if __name__ == "__main__":
    asyncio.run(demo_modes())
    asyncio.run(demo_explanation())