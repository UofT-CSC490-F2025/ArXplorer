"""
Unified ArXplorer Search API
Combines SciBERT + FAISS with A4 LLM Judge for hybrid search capabilities
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

from .hybrid_search import HybridSearchEngine, SearchMode, SearchResult, SearchMetrics
from .a4_judge_integration import EnsembleJudge, JudgeResult
from ..core.schemas import PipelineConfig


class SearchAPI:
    """
    Unified API for ArXplorer search with multiple modes
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.hybrid_engine = HybridSearchEngine(self.config)
        self.ensemble_judge = EnsembleJudge()
    
    async def search(self, 
                    query: str,
                    mode: str = "fast",
                    top_k: int = 20,
                    explain: bool = False) -> Dict[str, Any]:
        """
        Main search interface
        
        Args:
            query: Search query
            mode: "fast", "balanced", or "precise"  
            top_k: Number of results to return
            explain: Include detailed explanations
            
        Returns:
            Search results with metadata
        """
        
        search_mode = SearchMode(mode)
        
        # Perform search
        results, metrics = await self.hybrid_engine.search(
            query=query,
            mode=search_mode,
            top_k=top_k
        )
        
        # Format response
        response = {
            "query": query,
            "mode": mode,
            "total_results": len(results),
            "search_time_ms": metrics.total_time_ms,
            "results": [self._format_result(result, explain) for result in results],
            "metrics": asdict(metrics)
        }
        
        if explain:
            response["explanation"] = self._generate_explanation(query, mode, metrics)
        
        return response
    
    def _format_result(self, result: SearchResult, explain: bool) -> Dict[str, Any]:
        """Format a single search result"""
        formatted = {
            "arxiv_id": result.arxiv_id,
            "title": result.title,
            "abstract": result.abstract[:300] + "..." if len(result.abstract) > 300 else result.abstract,
            "authors": result.authors,
            "categories": result.categories,
            "relevance_score": result.combined_score or result.faiss_score,
            "confidence": result.confidence
        }
        
        if explain:
            formatted["scoring_details"] = {
                "faiss_score": result.faiss_score,
                "llm_score": result.llm_relevance_score,
                "combined_score": result.combined_score,
                "llm_explanation": result.llm_explanation
            }
        
        return formatted
    
    def _generate_explanation(self, query: str, mode: str, metrics: SearchMetrics) -> Dict[str, Any]:
        """Generate explanation of search process"""
        
        explanation = {
            "process": f"Used {mode} search mode",
            "stages": [],
            "performance": {
                "total_time_ms": metrics.total_time_ms,
                "candidates_examined": metrics.candidates_retrieved,
                "final_results": metrics.final_results
            }
        }
        
        # Stage 1: FAISS search
        explanation["stages"].append({
            "stage": "semantic_search",
            "description": "SciBERT embeddings + FAISS vector search",
            "time_ms": metrics.faiss_time_ms,
            "results": metrics.candidates_retrieved
        })
        
        # Stage 2: LLM filtering (if used)
        if metrics.llm_time_ms > 0:
            explanation["stages"].append({
                "stage": "llm_filtering",
                "description": "LLM judge relevance assessment",
                "time_ms": metrics.llm_time_ms,
                "enhancement": "Applied to candidates"
            })
        
        return explanation
    
    async def compare_modes(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Compare all search modes for the same query
        Useful for analysis and optimization
        """
        
        print(f"🔍 Comparing search modes for: '{query}'")
        
        modes = ["fast", "balanced", "precise"]
        comparisons = {}
        
        for mode in modes:
            print(f"   Testing {mode} mode...")
            
            start_time = time.time()
            result = await self.search(query, mode=mode, top_k=top_k, explain=True)
            
            comparisons[mode] = {
                "search_time_ms": result["search_time_ms"],
                "total_results": result["total_results"],
                "top_result": result["results"][0] if result["results"] else None,
                "explanation": result["explanation"]
            }
        
        # Calculate relative performance
        fast_time = comparisons["fast"]["search_time_ms"]
        
        for mode in modes:
            mode_time = comparisons[mode]["search_time_ms"]
            comparisons[mode]["speed_factor"] = mode_time / fast_time if fast_time > 0 else 1.0
        
        return {
            "query": query,
            "comparison": comparisons,
            "recommendation": self._recommend_mode(comparisons)
        }
    
    def _recommend_mode(self, comparisons: Dict[str, Any]) -> str:
        """Recommend best mode based on performance trade-offs"""
        
        fast_time = comparisons["fast"]["search_time_ms"]
        balanced_time = comparisons["balanced"]["search_time_ms"]
        precise_time = comparisons["precise"]["search_time_ms"]
        
        if precise_time < 2000:  # Less than 2 seconds
            return "precise - Good performance, maximum accuracy"
        elif balanced_time < 1000:  # Less than 1 second
            return "balanced - Good speed/accuracy trade-off"
        else:
            return "fast - Best for real-time applications"
    
    async def explain_result(self, arxiv_id: str, query: str) -> Dict[str, Any]:
        """
        Explain why a specific paper was or wasn't returned for a query
        """
        
        # Get detailed judgment from ensemble
        # This would need to fetch the paper from MongoDB first
        paper_data = await self._get_paper_by_id(arxiv_id)
        
        if not paper_data:
            return {"error": f"Paper {arxiv_id} not found"}
        
        # Get detailed LLM judgment
        judge_result = await self.ensemble_judge.judge_paper(
            query, paper_data["abstract"]
        )
        
        return {
            "arxiv_id": arxiv_id,
            "query": query,
            "paper": {
                "title": paper_data["title"],
                "abstract": paper_data["abstract"][:500] + "...",
                "authors": paper_data["authors"],
                "categories": paper_data["categories"]
            },
            "relevance_assessment": {
                "binary_decision": judge_result.binary_decision,
                "relevance_score": judge_result.relevance_score,
                "confidence": judge_result.confidence,
                "explanation": judge_result.explanation,
                "model_used": judge_result.model_used
            }
        }
    
    async def _get_paper_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get paper metadata by arXiv ID"""
        # This would query MongoDB for the paper
        # For now, return mock data
        return {
            "arxiv_id": arxiv_id,
            "title": f"Sample Paper {arxiv_id}",
            "abstract": "Sample abstract content for testing",
            "authors": ["Author A", "Author B"],
            "categories": ["cs.AI"]
        }


async def demo_unified_search():
    """Demonstration of the unified search API"""
    
    print("🚀 ArXplorer Unified Search API Demo")
    print("=" * 60)
    
    api = SearchAPI()
    
    # Test query
    query = "deep learning for computer vision"
    
    # 1. Basic search in different modes
    print(f"\n🔍 Searching for: '{query}'")
    print("-" * 40)
    
    for mode in ["fast", "balanced", "precise"]:
        print(f"\n📊 {mode.upper()} mode:")
        result = await api.search(query, mode=mode, top_k=3, explain=True)
        
        print(f"   Time: {result['search_time_ms']:.1f}ms")
        print(f"   Results: {result['total_results']}")
        print(f"   Process: {result['explanation']['process']}")
        
        if result['results']:
            top_result = result['results'][0]
            print(f"   Top result: {top_result['title'][:60]}...")
            print(f"   Score: {top_result['relevance_score']:.3f}")
    
    # 2. Mode comparison
    print(f"\n\n⚖️  Mode Comparison:")
    print("-" * 40)
    
    comparison = await api.compare_modes(query, top_k=5)
    print(f"Recommendation: {comparison['recommendation']}")
    
    for mode, data in comparison['comparison'].items():
        print(f"{mode}: {data['search_time_ms']:.1f}ms ({data['speed_factor']:.1f}x)")
    
    # 3. Result explanation
    print(f"\n\n💡 Result Explanation:")
    print("-" * 40)
    
    explanation = await api.explain_result("2301.12345", query)
    if "error" not in explanation:
        assessment = explanation['relevance_assessment']
        print(f"Decision: {assessment['binary_decision']}")
        print(f"Score: {assessment['relevance_score']:.3f}")
        print(f"Confidence: {assessment['confidence']:.3f}")
        print(f"Explanation: {assessment['explanation']}")


if __name__ == "__main__":
    asyncio.run(demo_unified_search())