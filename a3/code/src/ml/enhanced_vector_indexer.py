#!/usr/bin/env python3
"""
Enhanced Vector Indexer with Hyperparameter Optimization
Integrates FAISS search analysis results into the ArXplorer pipeline
"""

import os
import json
import numpy as np
import faiss
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from schemas import PaperEmbedding, PipelineConfig
from faiss_search_analysis import FAISSSearchAnalysis

class EnhancedVectorIndexer:
    """Enhanced FAISS vector indexer with automatic hyperparameter optimization"""
    
    def __init__(self, config: PipelineConfig, auto_optimize: bool = True):
        self.config = config
        self.auto_optimize = auto_optimize
        self.index = None
        self.paper_ids = []
        self.optimized_params = None
        self.logger = logging.getLogger(__name__)
        
        # Analysis results cache
        self.analysis_cache_dir = Path("./analysis_results")
        self.analysis_cache_dir.mkdir(exist_ok=True)
    
    def build_index(self, embeddings: List[PaperEmbedding]) -> None:
        """Build FAISS index with automatic parameter optimization"""
        
        try:
            if not embeddings:
                raise ValueError("No embeddings provided")
            
            # Extract embedding vectors and IDs
            vectors = np.array([emb.combined_embedding for emb in embeddings], dtype=np.float32)
            self.paper_ids = [emb.arxiv_id for emb in embeddings]
            
            dimension = vectors.shape[1]
            num_papers = len(embeddings)
            
            self.logger.info(f"Building index for {num_papers} papers with {dimension}D vectors")
            
            # Optimize parameters if requested
            if self.auto_optimize:
                self.optimized_params = self._optimize_parameters(vectors)
            else:
                self.optimized_params = self._use_config_parameters()
            
            # Create FAISS index based on optimized parameters
            self.index = self._create_optimized_index(vectors, dimension)
            
            # Add vectors to index
            self.index.add(vectors)
            
            self.logger.info(f"Built FAISS index with optimized parameters:")
            self.logger.info(f"  Index type: {self.optimized_params['index_type']}")
            self.logger.info(f"  Clusters: {self.optimized_params.get('nlist', 'N/A')}")
            self.logger.info(f"  nprobe: {self.optimized_params.get('nprobe', 'N/A')}")
            
        except Exception as e:
            self.logger.error(f"Error building index: {e}")
            raise
    
    def _optimize_parameters(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Run hyperparameter optimization for the given dataset"""
        
        num_papers = vectors.shape[0]
        cache_file = self.analysis_cache_dir / f"optimized_params_{num_papers}papers.json"
        
        # Check if we have cached results
        if cache_file.exists():
            self.logger.info(f"Loading cached optimization results from {cache_file}")
            with open(cache_file, 'r') as f:
                cached_results = json.load(f)
                return cached_results.get('optimized_params', self._use_config_parameters())
        
        # Run optimization analysis
        self.logger.info(f"Running hyperparameter optimization for {num_papers} papers...")
        
        analyzer = FAISSSearchAnalysis(num_papers=num_papers, embedding_dim=vectors.shape[1])
        
        # Use the actual embeddings instead of synthetic ones
        analyzer.embeddings = vectors.copy()
        
        # Run analysis
        results = analyzer.hyperparameter_tuning_analysis()
        optimal = analyzer.find_optimal_parameters(results)
        
        if optimal and 'best_balanced' in optimal:
            best = optimal['best_balanced']
            optimized_params = {
                'index_type': 'IVF' if num_papers >= 1000 else 'Flat',
                'nlist': best['nlist'] if num_papers >= 1000 else None,
                'nprobe': best['nprobe'] if num_papers >= 1000 else None,
                'top_k': best['k'],
                'expected_search_time_ms': best['time_ms'],
                'expected_recall': best['recall']
            }
            
            # Cache the results
            cache_data = {
                'optimized_params': optimized_params,
                'full_analysis': optimal,
                'dataset_info': {
                    'num_papers': num_papers,
                    'embedding_dim': vectors.shape[1]
                }
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.info(f"Optimization completed and cached to {cache_file}")
            return optimized_params
        
        else:
            self.logger.warning("Optimization failed, using config parameters")
            return self._use_config_parameters()
    
    def _use_config_parameters(self) -> Dict[str, Any]:
        """Use parameters from configuration file"""
        
        return {
            'index_type': self.config.faiss_index_type,
            'nlist': getattr(self.config, 'n_clusters', 1024),
            'nprobe': getattr(self.config, 'nprobe', 10),
            'top_k': getattr(self.config, 'top_k_results', 20),
            'expected_search_time_ms': None,
            'expected_recall': None
        }
    
    def _create_optimized_index(self, vectors: np.ndarray, dimension: int) -> faiss.Index:
        """Create FAISS index with optimized parameters"""
        
        num_papers = vectors.shape[0]
        
        if self.optimized_params['index_type'] == 'IVF' and num_papers >= 1000:
            # IVF (Inverted File) index for larger datasets
            nlist = self.optimized_params['nlist']
            quantizer = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Set nprobe for search
            index.nprobe = self.optimized_params['nprobe']
            
            # Train the index
            self.logger.info(f"Training IVF index with {nlist} clusters...")
            index.train(vectors)
            
            return index
        
        else:
            # Flat index for smaller datasets
            self.logger.info("Using flat index (no clustering)")
            return faiss.IndexFlatIP(dimension)
    
    def search(self, query_embedding: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index with optimized parameters"""
        
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Use optimized k if not specified
        if k is None:
            k = self.optimized_params.get('top_k', 10)
        
        # Ensure query is the right shape and type
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Perform search
        scores, indices = self.index.search(query_embedding, k)
        
        return scores, indices
    
    def get_paper_ids_from_indices(self, indices: np.ndarray) -> List[str]:
        """Convert FAISS indices to paper IDs"""
        
        paper_ids = []
        for idx_array in indices:
            batch_ids = []
            for idx in idx_array:
                if 0 <= idx < len(self.paper_ids):
                    batch_ids.append(self.paper_ids[idx])
                else:
                    batch_ids.append(None)  # Invalid index
            paper_ids.append(batch_ids)
        
        return paper_ids
    
    def save_index(self, filepath: str) -> None:
        """Save FAISS index and optimization parameters to disk"""
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, filepath)
            
            # Save paper IDs mapping
            ids_path = filepath.replace('.index', '_ids.json')
            with open(ids_path, 'w') as f:
                json.dump(self.paper_ids, f)
            
            # Save optimization parameters
            params_path = filepath.replace('.index', '_params.json')
            with open(params_path, 'w') as f:
                json.dump(self.optimized_params, f, indent=2)
            
            self.logger.info(f"Saved optimized index to {filepath}")
            self.logger.info(f"Saved paper IDs to {ids_path}")
            self.logger.info(f"Saved optimization parameters to {params_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
            raise
    
    def load_index(self, filepath: str) -> None:
        """Load FAISS index and optimization parameters from disk"""
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(filepath)
            
            # Load paper IDs mapping
            ids_path = filepath.replace('.index', '_ids.json')
            with open(ids_path, 'r') as f:
                self.paper_ids = json.load(f)
            
            # Load optimization parameters
            params_path = filepath.replace('.index', '_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    self.optimized_params = json.load(f)
                    
                    # Set nprobe if it's an IVF index
                    if hasattr(self.index, 'nprobe') and 'nprobe' in self.optimized_params:
                        self.index.nprobe = self.optimized_params['nprobe']
            
            self.logger.info(f"Loaded optimized index from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            raise
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about the optimization results"""
        
        if not self.optimized_params:
            return {"status": "not_optimized"}
        
        return {
            "status": "optimized",
            "parameters": self.optimized_params,
            "performance_estimate": {
                "search_time_ms": self.optimized_params.get('expected_search_time_ms', 'unknown'),
                "recall": self.optimized_params.get('expected_recall', 'unknown')
            }
        }

# Integration function to update existing pipeline
def upgrade_pipeline_with_optimization(pipeline_config: PipelineConfig) -> EnhancedVectorIndexer:
    """Upgrade existing pipeline to use enhanced vector indexer"""
    
    print("🚀 Upgrading ArXplorer pipeline with FAISS optimization...")
    
    enhanced_indexer = EnhancedVectorIndexer(pipeline_config, auto_optimize=True)
    
    print("✅ Enhanced vector indexer ready with automatic hyperparameter optimization")
    print("   Features added:")
    print("   • Automatic parameter optimization based on dataset size")
    print("   • Caching of optimization results")
    print("   • Performance estimates")
    print("   • Enhanced search capabilities")
    
    return enhanced_indexer

if __name__ == "__main__":
    # Demo usage
    from pipeline import PipelineConfig
    
    config = PipelineConfig()
    enhanced_indexer = upgrade_pipeline_with_optimization(config)
    
    print("\n📊 Enhanced Vector Indexer created successfully!")
    print("   Use this in place of the standard VectorIndexer for optimized performance")