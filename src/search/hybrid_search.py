"""
Hybrid Search System for ArXplorer
Combines SciBERT + FAISS (fast semantic search) with LLM judge (precision filtering)
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Core ArXplorer imports  
from ..core.schemas import ArXivPaper, PipelineConfig
from ..core.pipeline import VectorIndexer, EmbeddingGenerator
from ..data.mongodb_integration import MongoDBManager, MongoDBConfig


class SearchMode(Enum):
    """Search modes with different speed/precision trade-offs"""
    FAST = "fast"           # FAISS only (sub-second)
    BALANCED = "balanced"   # FAISS + light LLM filtering
    PRECISE = "precise"     # FAISS + full LLM re-ranking


@dataclass
class SearchResult:
    """Enhanced search result with multiple scoring methods"""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    
    # Scoring components
    faiss_score: float              # Semantic similarity (0-1)
    llm_relevance_score: Optional[float] = None  # LLM judgment (0-1)
    combined_score: Optional[float] = None       # Weighted combination
    llm_explanation: Optional[str] = None        # Why it's relevant
    
    # Metadata
    search_time_ms: float = 0.0
    confidence: float = 0.0


@dataclass
class SearchMetrics:
    """Performance metrics for search operations"""
    total_time_ms: float
    faiss_time_ms: float
    llm_time_ms: float
    candidates_retrieved: int
    final_results: int
    mode: SearchMode


class LLMJudge:
    """LLM-based relevance judge integrated from A4"""
    
    def __init__(self, model_name: str = "llama3:8b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.logger = logging.getLogger(__name__)
    
    def _build_prompt(self, query: str, abstract: str) -> str:
        """Build prompt for relevance judgment (adapted from A4)"""
        return (
            "You are a relevance judge for academic papers. Rate the relevance of the abstract to the query.\n"
            "Respond with a score from 0.0 (not relevant) to 1.0 (highly relevant) followed by a brief explanation.\n\n"
            f"Query: {query}\n"
            f"Abstract: {abstract[:1200]}...\n\n"
            "Score (0.0-1.0) and explanation:"
        )
    
    async def judge_relevance(self, query: str, paper: Dict[str, Any]) -> Tuple[float, str]:
        """Judge relevance of a paper to a query"""
        try:
            prompt = self._build_prompt(query, paper.get('abstract', ''))
            
            # Simulate LLM call (replace with actual Ollama integration)
            # For now, return a mock score based on keyword overlap
            query_words = set(query.lower().split())
            abstract_words = set(paper.get('abstract', '').lower().split())
            overlap = len(query_words.intersection(abstract_words))
            mock_score = min(1.0, overlap / max(len(query_words), 1) + 0.1)
            mock_explanation = f"Found {overlap} keyword matches between query and abstract"
            
            # TODO: Replace with actual Ollama API call
            # response = await self._call_ollama(prompt)
            # score, explanation = self._parse_response(response)
            
            return mock_score, mock_explanation
            
        except Exception as e:
            self.logger.error(f"LLM judgment failed: {e}")
            return 0.5, "LLM judgment failed, using neutral score"
    
    async def batch_judge(self, query: str, papers: List[Dict[str, Any]]) -> List[Tuple[float, str]]:
        """Judge multiple papers in batch"""
        tasks = [self.judge_relevance(query, paper) for paper in papers]
        return await asyncio.gather(*tasks)


class HybridSearchEngine:
    """
    Hybrid search engine combining SciBERT + FAISS with LLM judge
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(config)
        self.vector_indexer = VectorIndexer(config)
        self.llm_judge = LLMJudge()
        
        # MongoDB for paper metadata
        mongo_config = MongoDBConfig(
            connection_string="mongodb+srv://arxplorer_db_user:E14bupBhNORll6QT@arxplorercluster.sv4wks6.mongodb.net/arxplorer"
        )
        self.mongodb_manager = MongoDBManager(mongo_config)
        
        # Load FAISS index
        self._load_index()
    
    def _load_index(self):
        """Load pre-built FAISS index"""
        try:
            index_path = f"{self.config.index_path}/papers.index"
            self.vector_indexer.load_index(index_path)
            self.logger.info("FAISS index loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load FAISS index: {e}")
            self.logger.info("Index will be built when first needed")
    
    async def search(self, 
                    query: str, 
                    mode: SearchMode = SearchMode.FAST,
                    top_k: int = 20,
                    faiss_candidates: int = 100) -> Tuple[List[SearchResult], SearchMetrics]:
        """
        Perform hybrid search with specified mode
        
        Args:
            query: Search query
            mode: Search mode (fast/balanced/precise)
            top_k: Number of final results to return
            faiss_candidates: Number of candidates to retrieve from FAISS
        """
        start_time = time.time()
        
        # Stage 1: FAISS Semantic Search
        faiss_start = time.time()
        faiss_results = await self._faiss_search(query, k=faiss_candidates)
        faiss_time = (time.time() - faiss_start) * 1000
        
        if mode == SearchMode.FAST:
            # Fast mode: just return FAISS results
            results = faiss_results[:top_k]
            total_time = (time.time() - start_time) * 1000
            
            metrics = SearchMetrics(
                total_time_ms=total_time,
                faiss_time_ms=faiss_time,
                llm_time_ms=0.0,
                candidates_retrieved=len(faiss_results),
                final_results=len(results),
                mode=mode
            )
            
            return results, metrics
        
        # Stage 2: LLM Enhancement
        llm_start = time.time()
        enhanced_results = await self._llm_enhance(query, faiss_results, mode)
        llm_time = (time.time() - llm_start) * 1000
        
        # Final ranking and selection
        final_results = self._combine_scores(enhanced_results)[:top_k]
        
        total_time = (time.time() - start_time) * 1000
        
        metrics = SearchMetrics(
            total_time_ms=total_time,
            faiss_time_ms=faiss_time,
            llm_time_ms=llm_time,
            candidates_retrieved=len(faiss_results),
            final_results=len(final_results),
            mode=mode
        )
        
        return final_results, metrics
    
    async def _faiss_search(self, query: str, k: int = 100) -> List[SearchResult]:
        """Perform FAISS semantic search"""
        try:
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query)
            
            # Search FAISS index
            scores, indices = self.vector_indexer.search(query_embedding, k)
            
            # Get paper metadata from MongoDB
            paper_ids = [self.vector_indexer.get_paper_ids_from_indices(indices[0])]
            papers_data = await self._get_papers_metadata(paper_ids[0])
            
            # Create search results
            results = []
            for i, (score, paper_data) in enumerate(zip(scores[0], papers_data)):
                if paper_data:  # Skip if paper not found
                    result = SearchResult(
                        arxiv_id=paper_data.get('arxiv_id', ''),
                        title=paper_data.get('title', ''),
                        abstract=paper_data.get('abstract', ''),
                        authors=paper_data.get('authors', []),
                        categories=paper_data.get('categories', []),
                        faiss_score=float(score),
                        search_time_ms=0.0,  # Will be filled later
                        confidence=float(score)
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"FAISS search failed: {e}")
            return []
    
    async def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for search query"""
        # Use the same embedding model as the index
        # This is a simplified version - in practice, you'd use the actual EmbeddingGenerator
        embedding = np.random.normal(0, 1, (1, self.config.embedding_dimension)).astype(np.float32)
        return embedding / np.linalg.norm(embedding)
    
    async def _get_papers_metadata(self, paper_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Retrieve paper metadata from MongoDB"""
        try:
            papers = []
            for paper_id in paper_ids:
                # Simplified - in practice, batch query MongoDB
                paper_data = {
                    'arxiv_id': paper_id,
                    'title': f'Sample Paper {paper_id}',
                    'abstract': f'Sample abstract for paper {paper_id}',
                    'authors': ['Author A', 'Author B'],
                    'categories': ['cs.AI']
                }
                papers.append(paper_data)
            return papers
        except Exception as e:
            self.logger.error(f"Failed to get papers metadata: {e}")
            return [None] * len(paper_ids)
    
    async def _llm_enhance(self, 
                          query: str, 
                          faiss_results: List[SearchResult], 
                          mode: SearchMode) -> List[SearchResult]:
        """Enhance FAISS results with LLM judgments"""
        
        if mode == SearchMode.BALANCED:
            # Balanced: Only judge top candidates
            candidates = faiss_results[:50]
        else:  # PRECISE
            # Precise: Judge all candidates
            candidates = faiss_results
        
        # Convert to format expected by LLM judge
        papers_data = [
            {
                'arxiv_id': result.arxiv_id,
                'abstract': result.abstract
            }
            for result in candidates
        ]
        
        # Get LLM judgments
        judgments = await self.llm_judge.batch_judge(query, papers_data)
        
        # Update results with LLM scores
        enhanced_results = []
        for result, (llm_score, explanation) in zip(candidates, judgments):
            result.llm_relevance_score = llm_score
            result.llm_explanation = explanation
            enhanced_results.append(result)
        
        # Add remaining results without LLM judgment
        if len(faiss_results) > len(enhanced_results):
            enhanced_results.extend(faiss_results[len(enhanced_results):])
        
        return enhanced_results
    
    def _combine_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Combine FAISS and LLM scores with weighted average"""
        
        for result in results:
            if result.llm_relevance_score is not None:
                # Weighted combination: 70% FAISS, 30% LLM
                result.combined_score = (
                    0.7 * result.faiss_score + 
                    0.3 * result.llm_relevance_score
                )
                result.confidence = min(result.faiss_score, result.llm_relevance_score)
            else:
                # No LLM score available
                result.combined_score = result.faiss_score
                result.confidence = result.faiss_score * 0.8  # Lower confidence
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score or 0, reverse=True)
        
        return results


async def demo_hybrid_search():
    """Demonstration of hybrid search capabilities"""
    
    print("🔍 ArXplorer Hybrid Search Demo")
    print("=" * 50)
    
    # Initialize search engine
    config = PipelineConfig()
    search_engine = HybridSearchEngine(config)
    
    query = "machine learning for natural language processing"
    
    # Test all search modes
    modes = [SearchMode.FAST, SearchMode.BALANCED, SearchMode.PRECISE]
    
    for mode in modes:
        print(f"\n🚀 Testing {mode.value.upper()} mode:")
        print("-" * 30)
        
        results, metrics = await search_engine.search(query, mode=mode, top_k=5)
        
        print(f"⏱️  Total time: {metrics.total_time_ms:.1f}ms")
        print(f"📊 FAISS time: {metrics.faiss_time_ms:.1f}ms")
        print(f"🤖 LLM time: {metrics.llm_time_ms:.1f}ms")
        print(f"📄 Results: {metrics.final_results}/{metrics.candidates_retrieved}")
        
        print("\n📋 Top results:")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. {result.title}")
            print(f"     FAISS: {result.faiss_score:.3f}")
            if result.llm_relevance_score:
                print(f"     LLM: {result.llm_relevance_score:.3f}")
                print(f"     Combined: {result.combined_score:.3f}")
            print(f"     Confidence: {result.confidence:.3f}")
            if result.llm_explanation:
                print(f"     Reason: {result.llm_explanation}")
            print()


if __name__ == "__main__":
    asyncio.run(demo_hybrid_search())