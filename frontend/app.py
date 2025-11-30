"""
ArXplorer Frontend - Web interface for hybrid academic paper search.

A Flask-based web application providing a modern, responsive interface
for searching academic papers using the ArXplorer backend system.
"""

import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import json
from typing import List, Dict, Optional

# Add src to path for backend imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.retrieval.encoders import DenseEncoder, SparseEncoder
from src.retrieval.searchers import MilvusHybridSearcher
from src.retrieval.query_rewriting import LLMQueryRewriter
from src.retrieval.rerankers.intent_booster import IntentBooster
from src.retrieval.rerankers.title_author_matcher import TitleAuthorMatcher
from src.retrieval.rerankers.jina_reranker import JinaReranker
from src.retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker


app = Flask(__name__)
app.config['SECRET_KEY'] = 'arxplorer-dev-key'

# Global search system components (initialized on startup)
search_system = None


class ArXplorerSearchSystem:
    """Wrapper class for the complete search pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the search system with all components."""
        self.config = Config.from_yaml(config_path)
        self.doc_texts = {}  # Will be populated as needed
        
        # Initialize encoders
        print("Initializing ArXplorer search system...")
        self.dense_encoder = DenseEncoder(
            model_name=self.config.encoder.dense_model,
            device=self.config.encoder.device,
            normalize=self.config.encoder.normalize_dense,
            use_specter2=self.config.encoder.use_specter2,
            specter2_base_adapter=self.config.encoder.specter2_base_adapter,
            specter2_query_adapter=self.config.encoder.specter2_query_adapter
        )
        
        self.sparse_encoder = SparseEncoder(
            model_name=self.config.encoder.sparse_model,
            device=self.config.encoder.device,
            max_length=self.config.encoder.max_length
        )
        
        # Initialize searcher
        self.searcher = MilvusHybridSearcher(
            dense_encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
            host=self.config.milvus.host,
            port=self.config.milvus.port,
            collection_name=self.config.milvus.collection_name,
            rrf_k=self.config.search.rrf_k
        )
        
        # Initialize query rewriter (if enabled)
        self.query_rewriter = None
        if self.config.query_rewriting.enabled:
            self.query_rewriter = LLMQueryRewriter(
                model_name=self.config.query_rewriting.model,
                device=self.config.query_rewriting.device,
                max_length=self.config.query_rewriting.max_length,
                temperature=self.config.query_rewriting.temperature,
                num_rewrites=self.config.query_rewriting.num_rewrites,
                use_vllm=self.config.query_rewriting.use_vllm,
                vllm_endpoint=self.config.query_rewriting.vllm_endpoint
            )
        
        # Initialize boosting components
        self.intent_booster = IntentBooster(
            citation_weights=self.config.intent_boosting.citation_weights,
            date_weights=self.config.intent_boosting.date_weights,
            min_year=self.config.intent_boosting.min_year
        )
        
        self.title_author_matcher = TitleAuthorMatcher(
            title_threshold=self.config.title_author_matching.title_threshold,
            author_threshold=self.config.title_author_matching.author_threshold,
            title_boost_weight=self.config.title_author_matching.title_boost_weight,
            author_boost_weight=self.config.title_author_matching.author_boost_weight
        )
        
        print("✓ ArXplorer search system initialized successfully")
    
    def search(self, query: str, top_k: int = 10, enable_reranking: bool = True) -> Dict:
        """
        Execute complete search pipeline.
        
        Args:
            query: User query string
            top_k: Number of results to return
            enable_reranking: Whether to apply reranking
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            # Step 1: Query rewriting and filter extraction
            query_variants = [query]
            filter_expr = None
            intent = "topical"
            target_title = None
            target_authors = None
            
            if self.query_rewriter:
                rewrite_result = self.query_rewriter.rewrite(
                    query, 
                    num_rewrites=self.config.query_rewriting.num_rewrites
                )
                query_variants.extend(rewrite_result['rewrites'])
                filter_expr = rewrite_result.get('filter_expr')
                intent = rewrite_result.get('intent', 'topical')
                target_title = rewrite_result.get('target_title')
                target_authors = rewrite_result.get('target_authors')
            
            # Step 2: Hybrid search with variants
            retrieval_k = self.config.search.retrieval_k if enable_reranking else top_k
            
            results = self.searcher.search(
                query=query,
                top_k=retrieval_k,
                retrieval_k=retrieval_k,
                filter_expr=filter_expr,
                query_variants=query_variants[1:]  # Exclude original query
            )
            
            # Step 3: Intent-based boosting
            if self.config.intent_boosting.enabled:
                results = self.intent_booster.boost_scores(results, intent)
            
            # Step 4: Title/author matching (for specific paper searches)
            if self.config.title_author_matching.enabled and (target_title or target_authors):
                results = self.title_author_matcher.boost_scores(
                    results, target_title, target_authors
                )
            
            # Step 5: Neural reranking (if enabled)
            if enable_reranking and self.config.reranker.enabled and len(results) > 1:
                # Load document texts for reranking (simplified for demo)
                for result in results:
                    if result.doc_id not in self.doc_texts:
                        self.doc_texts[result.doc_id] = result.metadata.get('abstract', '')
                
                # Initialize reranker based on config
                if self.config.reranker.type == "jina":
                    reranker = JinaReranker(
                        doc_texts=self.doc_texts,
                        model_name=self.config.reranker.model,
                        batch_size=min(self.config.reranker.batch_size, len(results))
                    )
                else:
                    reranker = CrossEncoderReranker(
                        doc_texts=self.doc_texts,
                        model_name=self.config.reranker.model,
                        max_length=self.config.reranker.max_length,
                        batch_size=self.config.reranker.batch_size
                    )
                
                results = reranker.rerank(
                    query=query,
                    results=results,
                    top_k=top_k
                )
            
            # Format results for frontend
            formatted_results = []
            for i, result in enumerate(results[:top_k]):
                formatted_results.append({
                    'rank': i + 1,
                    'id': result.doc_id,
                    'title': result.metadata.get('title', 'Untitled'),
                    'abstract': result.metadata.get('abstract', ''),
                    'authors': result.metadata.get('authors', []),
                    'categories': result.metadata.get('categories', []),
                    'year': result.metadata.get('year'),
                    'citation_count': result.metadata.get('citation_count', 0),
                    'score': result.score,
                    'dense_score': getattr(result, 'dense_score', None),
                    'sparse_score': getattr(result, 'sparse_score', None),
                    'cross_encoder_score': getattr(result, 'cross_encoder_score', None)
                })
            
            return {
                'success': True,
                'query': query,
                'query_variants': query_variants,
                'intent': intent,
                'filter_expr': filter_expr,
                'total_results': len(formatted_results),
                'results': formatted_results,
                'reranking_enabled': enable_reranking
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'results': []
            }


@app.route('/')
def index():
    """Main search interface."""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """API endpoint for search requests."""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'success': False, 'error': 'No query provided'}), 400
    
    query = data['query'].strip()
    if not query:
        return jsonify({'success': False, 'error': 'Empty query'}), 400
    
    # Parse search parameters
    top_k = min(int(data.get('top_k', 10)), 50)  # Limit to 50 results max
    enable_reranking = data.get('enable_reranking', True)
    
    # Execute search
    if search_system is None:
        return jsonify({'success': False, 'error': 'Search system not initialized'}), 500
    
    results = search_system.search(
        query=query,
        top_k=top_k,
        enable_reranking=enable_reranking
    )
    
    return jsonify(results)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'search_system_ready': search_system is not None
    })


def initialize_search_system():
    """Initialize the search system on startup."""
    global search_system
    try:
        config_path = Path(__file__).parent.parent / "config.yaml"
        search_system = ArXplorerSearchSystem(str(config_path))
        print("✓ Search system initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize search system: {e}")
        search_system = None


if __name__ == '__main__':
    # Initialize search system
    initialize_search_system()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )