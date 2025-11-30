"""
ArXplorer Hybrid Search Demonstration
Shows the integration of SciBERT + FAISS with A4 LLM Judge
"""

import asyncio
import time
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from search.unified_api import SearchAPI


async def main():
    """Comprehensive demonstration of hybrid search capabilities"""
    
    print("🎯 ArXplorer Hybrid Search System")
    print("Combining SciBERT + FAISS with A4 LLM Judge")
    print("=" * 60)
    
    # Initialize the search system
    print("🔧 Initializing hybrid search system...")
    api = SearchAPI()
    
    # Demo queries
    queries = [
        "machine learning for natural language processing",
        "deep neural networks for computer vision", 
        "transformer architectures for sequence modeling",
        "quantum computing applications in optimization"
    ]
    
    print("✅ System initialized!\n")
    
    # 1. Speed Comparison Demo
    print("⚡ SPEED COMPARISON DEMO")
    print("-" * 30)
    
    query = queries[0]
    print(f"Query: '{query}'\n")
    
    modes = [
        ("fast", "SciBERT + FAISS only"),
        ("balanced", "FAISS + Light LLM filtering"), 
        ("precise", "FAISS + Full LLM re-ranking")
    ]
    
    for mode, description in modes:
        start = time.time()
        result = await api.search(query, mode=mode, top_k=5)
        elapsed = time.time() - start
        
        print(f"🚀 {mode.upper():<10} | {elapsed*1000:6.1f}ms | {description}")
        print(f"   Results: {result['total_results']}")
        if result['results']:
            top_score = result['results'][0]['relevance_score']
            print(f"   Top score: {top_score:.3f}")
        print()
    
    # 2. Quality Comparison Demo
    print("\n🎯 QUALITY COMPARISON DEMO")
    print("-" * 30)
    
    comparison = await api.compare_modes(queries[1], top_k=3)
    print(f"\nQuery: '{queries[1]}'")
    print(f"Recommendation: {comparison['recommendation']}\n")
    
    # Show top result from each mode
    for mode, data in comparison['comparison'].items():
        print(f"📊 {mode.upper()} Mode:")
        print(f"   Time: {data['search_time_ms']:.1f}ms")
        
        if data['top_result']:
            result = data['top_result']
            print(f"   Title: {result['title'][:60]}...")
            print(f"   Score: {result['relevance_score']:.3f}")
            print(f"   Confidence: {result['confidence']:.3f}")
        print()
    
    # 3. Explanation Demo
    print("\n💡 EXPLANATION DEMO")
    print("-" * 30)
    
    explanation = await api.explain_result("2301.12345", queries[2])
    print(f"Query: '{queries[2]}'")
    print(f"Paper: {explanation['paper']['title']}")
    
    assessment = explanation['relevance_assessment']
    print(f"\n🤖 LLM Judge Assessment:")
    print(f"   Decision: {assessment['binary_decision']}")
    print(f"   Relevance: {assessment['relevance_score']:.3f}")
    print(f"   Confidence: {assessment['confidence']:.3f}")
    print(f"   Model: {assessment['model_used']}")
    print(f"   Reasoning: {assessment['explanation']}")
    
    # 4. Performance Summary
    print("\n\n📈 SYSTEM PERFORMANCE SUMMARY")
    print("=" * 40)
    
    print("🏆 Strengths of Hybrid Approach:")
    print("   ✅ Fast semantic search via SciBERT + FAISS")
    print("   ✅ Intelligent filtering via LLM judge")
    print("   ✅ Flexible speed/quality trade-offs")
    print("   ✅ Explainable relevance decisions")
    print("   ✅ Production-ready scalability")
    
    print("\n⚙️  Mode Recommendations:")
    print("   🚀 FAST: Interactive search, real-time applications")
    print("   ⚖️  BALANCED: General purpose, good compromise")
    print("   🎯 PRECISE: Research quality, batch processing")
    
    print("\n🔧 Technical Architecture:")
    print("   📊 Stage 1: SciBERT (768D) → FAISS → Top candidates")
    print("   🤖 Stage 2: LLM Judge → Relevance filtering")
    print("   📈 Stage 3: Hybrid scoring → Final ranking")
    
    print("\n✨ Integration Success!")
    print("   Your ArXplorer pipeline + A4 LLM judge = Best of both worlds")


if __name__ == "__main__":
    asyncio.run(main())