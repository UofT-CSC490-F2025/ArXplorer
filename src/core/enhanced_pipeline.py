"""
Enhanced ArXplorer Cloud Pipeline with MongoDB Integration
Combines AWS S3 storage with MongoDB for structured data persistence
"""

import asyncio
import logging
import json
import uuid
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import pipeline components
from pipeline import (
    ArXivDataIngester, 
    TextProcessor, 
    EmbeddingGenerator, 
    VectorIndexer
)
from schemas import ArXivPaper, ProcessedPaper, PaperEmbedding, PipelineConfig
from s3_integration import S3StorageManager, CloudPipelineConfig
from mongodb_integration import MongoDBManager, MongoDBConfig


class EnhancedCloudPipeline:
    """
    Enhanced cloud-integrated ArXplorer pipeline with both S3 and MongoDB storage
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the enhanced cloud pipeline
        
        Args:
            config_path: Path to pipeline configuration file
        """
        # Load configuration
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logging first
        logging.basicConfig(
            level=getattr(logging, self.config.get('monitoring', {}).get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline config
        self.pipeline_config = PipelineConfig()
        
        # Initialize storage managers
        self._init_storage_managers()
        
        # Initialize pipeline components
        self.ingester = ArXivDataIngester(self.pipeline_config)
        self.processor = TextProcessor(self.pipeline_config)
        self.embedding_generator = EmbeddingGenerator(self.pipeline_config)
        self.indexer = VectorIndexer(self.pipeline_config)
        
        self.logger.info("Enhanced ArXplorer Cloud Pipeline initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.warning(f"Could not load config file {self.config_path}: {e}")
            return {}
    
    def _init_storage_managers(self):
        """Initialize S3 and MongoDB storage managers"""
        try:
            # Initialize S3 storage
            if self.config.get('storage', {}).get('mode') == 'cloud':
                self.aws_config = CloudPipelineConfig.load_aws_config()
                self.s3_storage = S3StorageManager(self.aws_config)
                self.logger.info("S3 storage manager initialized")
            else:
                self.s3_storage = None
                self.logger.info("S3 storage disabled")
            
            # Initialize MongoDB storage
            mongodb_config = self.config.get('mongodb', {})
            if mongodb_config.get('enabled', True):
                mongo_config = MongoDBConfig(
                    connection_string=mongodb_config.get('connection_string', 'mongodb://localhost:27017/'),
                    database_name=mongodb_config.get('database_name', 'arxplorer'),
                    use_async=mongodb_config.get('use_async', True)
                )
                mongo_config.collections = mongodb_config.get('collections', mongo_config.collections)
                
                self.mongodb_storage = MongoDBManager(mongo_config)
                self.logger.info("MongoDB storage manager initialized")
            else:
                self.mongodb_storage = None
                self.logger.info("MongoDB storage disabled")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize storage managers: {e}")
            raise
    
    async def initialize_databases(self) -> Dict[str, bool]:
        """Initialize databases and create necessary indexes"""
        results = {}
        
        # Initialize MongoDB
        if self.mongodb_storage:
            try:
                mongo_success = await self.mongodb_storage.initialize_database()
                results['mongodb'] = mongo_success
                if mongo_success:
                    self.logger.info("MongoDB database initialized successfully")
                else:
                    self.logger.error("MongoDB database initialization failed")
            except Exception as e:
                self.logger.error(f"MongoDB initialization error: {e}")
                results['mongodb'] = False
        else:
            results['mongodb'] = False
            
        return results
    
    async def run_full_pipeline(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete pipeline with both S3 and MongoDB storage
        
        Args:
            query_params: Query parameters for data ingestion
            
        Returns:
            Dict with comprehensive pipeline results
        """
        batch_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        results = {
            'batch_id': batch_id,
            'start_time': start_time.isoformat(),
            'status': 'running',
            'stages': {},
            'storage': {
                's3': {'enabled': self.s3_storage is not None},
                'mongodb': {'enabled': self.mongodb_storage is not None}
            }
        }
        
        try:
            # Stage 1: Data Ingestion
            self.logger.info("Stage 1: Ingesting data from arXiv API")
            raw_papers = await self._ingest_papers(query_params)
            
            if not raw_papers:
                raise Exception("No papers found for the given query")
            
            # Store raw papers
            s3_raw_success = await self._store_raw_papers(raw_papers, batch_id)
            mongo_raw_success = await self._store_raw_papers_mongo(raw_papers, batch_id)
            
            results['stages']['ingestion'] = {
                'paper_count': len(raw_papers),
                's3_upload_success': s3_raw_success,
                'mongodb_insert_success': mongo_raw_success,
                'completed_at': datetime.now().isoformat()
            }
            
            # Stage 2: Data Processing
            self.logger.info("Stage 2: Processing and cleaning papers")
            processed_papers = await self._process_papers(raw_papers)
            
            # Store processed papers
            s3_processed_success = await self._store_processed_papers(processed_papers, batch_id)
            mongo_processed_success = await self._store_processed_papers_mongo(processed_papers, batch_id)
            
            results['stages']['processing'] = {
                'processed_count': len(processed_papers),
                's3_upload_success': s3_processed_success,
                'mongodb_insert_success': mongo_processed_success,
                'completed_at': datetime.now().isoformat()
            }
            
            # Stage 3: Embedding Generation
            self.logger.info("Stage 3: Generating embeddings")
            embeddings = await self._generate_embeddings(processed_papers)
            
            # Store embeddings
            s3_embedding_success = await self._store_embeddings(embeddings, batch_id)
            mongo_embedding_success = await self._store_embeddings_mongo(embeddings, batch_id)
            
            results['stages']['embedding'] = {
                'embedding_count': len(embeddings),
                's3_upload_success': s3_embedding_success,
                'mongodb_insert_success': mongo_embedding_success,
                'completed_at': datetime.now().isoformat()
            }
            
            # Stage 4: Index Building
            self.logger.info("Stage 4: Building vector index")
            index_success = await self._build_search_index(embeddings, batch_id)
            
            results['stages']['indexing'] = {
                'index_built': index_success,
                'completed_at': datetime.now().isoformat()
            }
            
            # Final results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results.update({
                'status': 'completed',
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'total_papers': len(raw_papers),
                'successful_papers': len(processed_papers),
                'generated_embeddings': len(embeddings)
            })
            
            # Log pipeline run to MongoDB
            await self._log_pipeline_run(results)
            
            self.logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            results['duration_seconds'] = (datetime.now() - start_time).total_seconds()
            
            # Log failed run to MongoDB
            await self._log_pipeline_run(results)
            
            self.logger.error(f"Pipeline failed: {e}")
        
        return results
    
    async def _ingest_papers(self, query_params: Dict[str, Any]) -> List[ArXivPaper]:
        """Ingest papers from arXiv API"""
        start_date_str = query_params.get('start_date', '2024-01-01')
        end_date_str = query_params.get('end_date', '2024-01-02')
        max_results = query_params.get('max_results', 10)
        categories = query_params.get('categories', None)
        
        # Convert string dates to datetime objects
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        return list(self.ingester.fetch_papers(start_date, end_date, categories, max_results))
    
    async def _process_papers(self, raw_papers: List[ArXivPaper]) -> List[ProcessedPaper]:
        """Process raw papers"""
        processed_papers = []
        for paper in raw_papers:
            try:
                processed_paper = self.processor.process_paper(paper)
                if processed_paper:
                    processed_papers.append(processed_paper)
            except Exception as e:
                self.logger.warning(f"Failed to process paper {paper.arxiv_id}: {e}")
        
        return processed_papers
    
    async def _generate_embeddings(self, processed_papers: List[ProcessedPaper]) -> List[PaperEmbedding]:
        """Generate embeddings for processed papers"""
        embeddings = []
        for paper in processed_papers:
            try:
                embedding = self.embedding_generator.generate_embeddings(paper)
                if embedding:
                    embeddings.append(embedding)
            except Exception as e:
                self.logger.warning(f"Failed to generate embedding for paper {paper.arxiv_id}: {e}")
        
        return embeddings
    
    async def _build_search_index(self, embeddings: List[PaperEmbedding], batch_id: str) -> bool:
        """Build FAISS search index"""
        try:
            if not embeddings:
                return False
            
            self.indexer.build_index(embeddings)
            self.logger.info(f"Successfully built search index with {len(embeddings)} embeddings")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build search index: {e}")
            return False
    
    # S3 Storage Methods
    async def _store_raw_papers(self, papers: List[ArXivPaper], batch_id: str) -> bool:
        """Store raw papers in S3"""
        if not self.s3_storage:
            return False
        
        try:
            return self.s3_storage.upload_raw_papers(papers, batch_id)
        except Exception as e:
            self.logger.error(f"Failed to store raw papers in S3: {e}")
            return False
    
    async def _store_processed_papers(self, papers: List[ProcessedPaper], batch_id: str) -> bool:
        """Store processed papers in S3"""
        if not self.s3_storage:
            return False
        
        try:
            return self.s3_storage.upload_processed_papers(papers, batch_id)
        except Exception as e:
            self.logger.error(f"Failed to store processed papers in S3: {e}")
            return False
    
    async def _store_embeddings(self, embeddings: List[PaperEmbedding], batch_id: str) -> bool:
        """Store embeddings in S3"""
        if not self.s3_storage:
            return False
        
        try:
            return self.s3_storage.upload_embeddings(embeddings, batch_id)
        except Exception as e:
            self.logger.error(f"Failed to store embeddings in S3: {e}")
            return False
    
    # MongoDB Storage Methods
    async def _store_raw_papers_mongo(self, papers: List[ArXivPaper], batch_id: str) -> bool:
        """Store raw papers in MongoDB"""
        if not self.mongodb_storage:
            return False
        
        try:
            return await self.mongodb_storage.insert_raw_papers(papers, batch_id)
        except Exception as e:
            self.logger.error(f"Failed to store raw papers in MongoDB: {e}")
            return False
    
    async def _store_processed_papers_mongo(self, papers: List[ProcessedPaper], batch_id: str) -> bool:
        """Store processed papers in MongoDB"""
        if not self.mongodb_storage:
            return False
        
        try:
            return await self.mongodb_storage.insert_processed_papers(papers, batch_id)
        except Exception as e:
            self.logger.error(f"Failed to store processed papers in MongoDB: {e}")
            return False
    
    async def _store_embeddings_mongo(self, embeddings: List[PaperEmbedding], batch_id: str) -> bool:
        """Store embeddings in MongoDB"""
        if not self.mongodb_storage:
            return False
        
        try:
            return await self.mongodb_storage.insert_embeddings(embeddings, batch_id)
        except Exception as e:
            self.logger.error(f"Failed to store embeddings in MongoDB: {e}")
            return False
    
    async def _log_pipeline_run(self, results: Dict[str, Any]) -> bool:
        """Log pipeline run information to MongoDB"""
        if not self.mongodb_storage:
            return False
        
        try:
            return await self.mongodb_storage.log_pipeline_run(results)
        except Exception as e:
            self.logger.error(f"Failed to log pipeline run: {e}")
            return False
    
    # Query and Analytics Methods
    async def search_papers_by_category(self, categories: List[str], limit: int = 50) -> List[Dict]:
        """Search papers by category using MongoDB"""
        if not self.mongodb_storage:
            return []
        
        try:
            return await self.mongodb_storage.search_papers_by_category(categories, limit)
        except Exception as e:
            self.logger.error(f"Failed to search papers by category: {e}")
            return []
    
    async def search_papers_by_text(self, text_query: str, limit: int = 50) -> List[Dict]:
        """Search papers by text using MongoDB"""
        if not self.mongodb_storage:
            return []
        
        try:
            return await self.mongodb_storage.search_papers_by_text(text_query, limit)
        except Exception as e:
            self.logger.error(f"Failed to search papers by text: {e}")
            return []
    
    async def get_database_analytics(self) -> Dict[str, Any]:
        """Get comprehensive database analytics"""
        analytics = {
            'mongodb': {},
            's3': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # MongoDB analytics
        if self.mongodb_storage:
            try:
                analytics['mongodb'] = await self.mongodb_storage.get_database_stats()
            except Exception as e:
                self.logger.error(f"Failed to get MongoDB analytics: {e}")
                analytics['mongodb'] = {'error': str(e)}
        
        # S3 analytics
        if self.s3_storage:
            try:
                analytics['s3'] = self.s3_storage.get_bucket_info()
            except Exception as e:
                self.logger.error(f"Failed to get S3 analytics: {e}")
                analytics['s3'] = {'error': str(e)}
        
        return analytics
    
    async def process_specific_category(self, category: str, max_results: int = 50) -> Dict[str, Any]:
        """Process papers from a specific arXiv category"""
        # Get papers from the last 2 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        query_params = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'categories': [category],
            'max_results': max_results
        }
        
        return await self.run_full_pipeline(query_params)
    
    def close_connections(self):
        """Close all database connections"""
        if self.mongodb_storage:
            self.mongodb_storage.close_connections()
        
        self.logger.info("All database connections closed")


# Convenience functions
async def create_enhanced_pipeline(config_path: str = "config.yaml") -> EnhancedCloudPipeline:
    """
    Create and initialize an enhanced cloud pipeline
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        EnhancedCloudPipeline instance
    """
    pipeline = EnhancedCloudPipeline(config_path)
    
    # Initialize databases
    init_results = await pipeline.initialize_databases()
    pipeline.logger.info(f"Database initialization results: {init_results}")
    
    return pipeline


if __name__ == "__main__":
    # Test the enhanced pipeline
    async def main():
        print("🚀 Testing Enhanced ArXplorer Cloud Pipeline...")
        
        try:
            # Create pipeline
            pipeline = await create_enhanced_pipeline()
            
            # Get analytics
            analytics = await pipeline.get_database_analytics()
            print(f"📊 Database Analytics: {json.dumps(analytics, indent=2, default=str)}")
            
            # Test with a small batch
            print("\n🔬 Testing with 2 recent AI papers...")
            results = await pipeline.process_specific_category("cs.AI", max_results=2)
            
            print(f"✅ Pipeline Results:")
            print(f"   Status: {results['status']}")
            print(f"   Duration: {results.get('duration_seconds', 0):.2f} seconds")
            print(f"   Papers processed: {results.get('total_papers', 0)}")
            
            # Show storage results
            for stage_name, stage_info in results.get('stages', {}).items():
                print(f"   {stage_name}: S3={stage_info.get('s3_upload_success', False)}, MongoDB={stage_info.get('mongodb_insert_success', False)}")
            
            # Close connections
            pipeline.close_connections()
            
        except Exception as e:
            print(f"❌ Pipeline test failed: {e}")
    
    # Run the test
    asyncio.run(main())