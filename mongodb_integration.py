"""
MongoDB Integration for ArXplorer Pipeline
Provides document-based storage and advanced querying capabilities
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import json
from dataclasses import asdict

import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np

from schemas import ArXivPaper, ProcessedPaper, PaperEmbedding, SearchQuery, PipelineConfig


class MongoDBConfig:
    """Configuration for MongoDB connection and collections"""
    
    def __init__(self,
                 connection_string: str = "mongodb://localhost:27017/",
                 database_name: str = "arxplorer",
                 use_async: bool = True):
        self.connection_string = connection_string
        self.database_name = database_name
        self.use_async = use_async
        
        # Collection names
        self.collections = {
            'raw_papers': 'raw_papers',
            'processed_papers': 'processed_papers', 
            'embeddings': 'embeddings',
            'search_queries': 'search_queries',
            'pipeline_runs': 'pipeline_runs',
            'user_feedback': 'user_feedback'
        }


class MongoDBManager:
    """
    Manages MongoDB operations for ArXplorer pipeline
    Provides both sync and async operations for flexibility
    """
    
    def __init__(self, config: MongoDBConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize connections
        if config.use_async:
            self.async_client = AsyncIOMotorClient(config.connection_string)
            self.async_db = self.async_client[config.database_name]
        else:
            self.sync_client = MongoClient(config.connection_string)
            self.sync_db = self.sync_client[config.database_name]
        
        self.collections = config.collections
        
    async def initialize_database(self) -> bool:
        """
        Initialize database with proper indexes and collections
        
        Returns:
            bool: Success status
        """
        try:
            if self.config.use_async:
                db = self.async_db
            else:
                db = self.sync_db
            
            # Create indexes for efficient querying
            
            # Raw papers indexes
            raw_papers = db[self.collections['raw_papers']]
            if self.config.use_async:
                await raw_papers.create_index([("arxiv_id", ASCENDING)], unique=True)
                await raw_papers.create_index([("published", DESCENDING)])
                await raw_papers.create_index([("categories", ASCENDING)])
                await raw_papers.create_index([("title", TEXT), ("abstract", TEXT)])
            else:
                raw_papers.create_index([("arxiv_id", ASCENDING)], unique=True)
                raw_papers.create_index([("published", DESCENDING)])
                raw_papers.create_index([("categories", ASCENDING)])
                raw_papers.create_index([("title", TEXT), ("abstract", TEXT)])
            
            # Processed papers indexes
            processed_papers = db[self.collections['processed_papers']]
            if self.config.use_async:
                await processed_papers.create_index([("arxiv_id", ASCENDING)], unique=True)
                await processed_papers.create_index([("word_count", DESCENDING)])
                await processed_papers.create_index([("readability_score", DESCENDING)])
            else:
                processed_papers.create_index([("arxiv_id", ASCENDING)], unique=True)
                processed_papers.create_index([("word_count", DESCENDING)])
                processed_papers.create_index([("readability_score", DESCENDING)])
            
            # Embeddings indexes
            embeddings = db[self.collections['embeddings']]
            if self.config.use_async:
                await embeddings.create_index([("arxiv_id", ASCENDING)], unique=True)
                await embeddings.create_index([("model_name", ASCENDING)])
                await embeddings.create_index([("created_at", DESCENDING)])
            else:
                embeddings.create_index([("arxiv_id", ASCENDING)], unique=True)
                embeddings.create_index([("model_name", ASCENDING)])
                embeddings.create_index([("created_at", DESCENDING)])
            
            # Search queries indexes
            search_queries = db[self.collections['search_queries']]
            if self.config.use_async:
                await search_queries.create_index([("created_at", DESCENDING)])
                await search_queries.create_index([("raw_query", TEXT)])
            else:
                search_queries.create_index([("created_at", DESCENDING)])
                search_queries.create_index([("raw_query", TEXT)])
            
            # Pipeline runs indexes
            pipeline_runs = db[self.collections['pipeline_runs']]
            if self.config.use_async:
                await pipeline_runs.create_index([("batch_id", ASCENDING)], unique=True)
                await pipeline_runs.create_index([("start_time", DESCENDING)])
                await pipeline_runs.create_index([("status", ASCENDING)])
            else:
                pipeline_runs.create_index([("batch_id", ASCENDING)], unique=True)
                pipeline_runs.create_index([("start_time", DESCENDING)])
                pipeline_runs.create_index([("status", ASCENDING)])
            
            self.logger.info("MongoDB database initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MongoDB database: {e}")
            return False
    
    # Raw Papers Operations
    async def insert_raw_papers(self, papers: List[ArXivPaper], batch_id: str) -> bool:
        """Insert raw papers into MongoDB"""
        try:
            if not papers:
                return True
            
            documents = []
            for paper in papers:
                doc = asdict(paper)
                doc['batch_id'] = batch_id
                doc['inserted_at'] = datetime.now()
                
                # Convert datetime objects to MongoDB-compatible format
                if 'published' in doc and doc['published']:
                    doc['published'] = doc['published']
                if 'updated' in doc and doc['updated']:
                    doc['updated'] = doc['updated']
                
                # Convert enum to string for MongoDB compatibility
                if 'status' in doc and hasattr(doc['status'], 'value'):
                    doc['status'] = doc['status'].value
                
                documents.append(doc)
            
            if self.config.use_async:
                collection = self.async_db[self.collections['raw_papers']]
                result = await collection.insert_many(documents, ordered=False)
            else:
                collection = self.sync_db[self.collections['raw_papers']]
                result = collection.insert_many(documents, ordered=False)
            
            self.logger.info(f"Inserted {len(result.inserted_ids)} raw papers into MongoDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert raw papers: {e}")
            return False
    
    async def insert_processed_papers(self, papers: List[ProcessedPaper], batch_id: str) -> bool:
        """Insert processed papers into MongoDB"""
        try:
            if not papers:
                return True
            
            documents = []
            for paper in papers:
                doc = asdict(paper)
                doc['batch_id'] = batch_id
                doc['inserted_at'] = datetime.now()
                documents.append(doc)
            
            if self.config.use_async:
                collection = self.async_db[self.collections['processed_papers']]
                result = await collection.insert_many(documents, ordered=False)
            else:
                collection = self.sync_db[self.collections['processed_papers']]
                result = collection.insert_many(documents, ordered=False)
            
            self.logger.info(f"Inserted {len(result.inserted_ids)} processed papers into MongoDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert processed papers: {e}")
            return False
    
    async def insert_embeddings(self, embeddings: List[PaperEmbedding], batch_id: str) -> bool:
        """Insert embeddings into MongoDB"""
        try:
            if not embeddings:
                return True
            
            documents = []
            for embedding in embeddings:
                doc = asdict(embedding)
                doc['batch_id'] = batch_id
                doc['inserted_at'] = datetime.now()
                
                # Convert numpy arrays to lists for MongoDB storage
                if isinstance(doc.get('title_embedding'), np.ndarray):
                    doc['title_embedding'] = doc['title_embedding'].tolist()
                if isinstance(doc.get('abstract_embedding'), np.ndarray):
                    doc['abstract_embedding'] = doc['abstract_embedding'].tolist()
                if isinstance(doc.get('combined_embedding'), np.ndarray):
                    doc['combined_embedding'] = doc['combined_embedding'].tolist()
                
                documents.append(doc)
            
            if self.config.use_async:
                collection = self.async_db[self.collections['embeddings']]
                result = await collection.insert_many(documents, ordered=False)
            else:
                collection = self.sync_db[self.collections['embeddings']]
                result = collection.insert_many(documents, ordered=False)
            
            self.logger.info(f"Inserted {len(result.inserted_ids)} embeddings into MongoDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert embeddings: {e}")
            return False
    
    # Query Operations
    async def search_papers_by_category(self, categories: List[str], limit: int = 50) -> List[Dict]:
        """Search papers by arXiv categories"""
        try:
            query = {"categories": {"$in": categories}}
            
            if self.config.use_async:
                collection = self.async_db[self.collections['raw_papers']]
                cursor = collection.find(query).sort("published", DESCENDING).limit(limit)
                papers = await cursor.to_list(length=limit)
            else:
                collection = self.sync_db[self.collections['raw_papers']]
                papers = list(collection.find(query).sort("published", DESCENDING).limit(limit))
            
            return papers
            
        except Exception as e:
            self.logger.error(f"Failed to search papers by category: {e}")
            return []
    
    async def search_papers_by_text(self, text_query: str, limit: int = 50) -> List[Dict]:
        """Search papers using full-text search"""
        try:
            query = {"$text": {"$search": text_query}}
            
            if self.config.use_async:
                collection = self.async_db[self.collections['raw_papers']]
                cursor = collection.find(query).sort([("score", {"$meta": "textScore"})]).limit(limit)
                papers = await cursor.to_list(length=limit)
            else:
                collection = self.sync_db[self.collections['raw_papers']]
                papers = list(collection.find(query).sort([("score", {"$meta": "textScore"})]).limit(limit))
            
            return papers
            
        except Exception as e:
            self.logger.error(f"Failed to search papers by text: {e}")
            return []
    
    async def get_papers_by_date_range(self, 
                                     start_date: datetime, 
                                     end_date: datetime,
                                     limit: int = 100) -> List[Dict]:
        """Get papers within a date range"""
        try:
            query = {
                "published": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            if self.config.use_async:
                collection = self.async_db[self.collections['raw_papers']]
                cursor = collection.find(query).sort("published", DESCENDING).limit(limit)
                papers = await cursor.to_list(length=limit)
            else:
                collection = self.sync_db[self.collections['raw_papers']]
                papers = list(collection.find(query).sort("published", DESCENDING).limit(limit))
            
            return papers
            
        except Exception as e:
            self.logger.error(f"Failed to get papers by date range: {e}")
            return []
    
    # Analytics Operations
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}
            
            for collection_name in self.collections.values():
                if self.config.use_async:
                    collection = self.async_db[collection_name]
                    count = await collection.count_documents({})
                else:
                    collection = self.sync_db[collection_name]
                    count = collection.count_documents({})
                
                stats[collection_name] = {
                    'document_count': count
                }
            
            # Get category distribution
            if self.config.use_async:
                pipeline = [
                    {"$unwind": "$categories"},
                    {"$group": {"_id": "$categories", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 10}
                ]
                collection = self.async_db[self.collections['raw_papers']]
                cursor = collection.aggregate(pipeline)
                category_stats = await cursor.to_list(length=10)
            else:
                pipeline = [
                    {"$unwind": "$categories"},
                    {"$group": {"_id": "$categories", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 10}
                ]
                collection = self.sync_db[self.collections['raw_papers']]
                category_stats = list(collection.aggregate(pipeline))
            
            stats['top_categories'] = category_stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}
    
    async def log_pipeline_run(self, run_info: Dict[str, Any]) -> bool:
        """Log pipeline run information"""
        try:
            run_info['logged_at'] = datetime.now()
            
            if self.config.use_async:
                collection = self.async_db[self.collections['pipeline_runs']]
                await collection.insert_one(run_info)
            else:
                collection = self.sync_db[self.collections['pipeline_runs']]
                collection.insert_one(run_info)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log pipeline run: {e}")
            return False
    
    def close_connections(self):
        """Close database connections"""
        try:
            if hasattr(self, 'sync_client'):
                self.sync_client.close()
            if hasattr(self, 'async_client'):
                self.async_client.close()
            
            self.logger.info("MongoDB connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing MongoDB connections: {e}")


# Helper functions for integration with existing pipeline
def create_mongodb_manager(connection_string: str = None, database_name: str = "arxplorer") -> MongoDBManager:
    """
    Create a MongoDB manager with default configuration
    
    Args:
        connection_string: MongoDB connection string
        database_name: Database name
        
    Returns:
        MongoDBManager instance
    """
    if not connection_string:
        connection_string = "mongodb://localhost:27017/"
    
    config = MongoDBConfig(
        connection_string=connection_string,
        database_name=database_name,
        use_async=True
    )
    
    return MongoDBManager(config)


async def test_mongodb_connection(connection_string: str = "mongodb://localhost:27017/") -> bool:
    """
    Test MongoDB connection
    
    Args:
        connection_string: MongoDB connection string
        
    Returns:
        bool: Connection success status
    """
    try:
        client = AsyncIOMotorClient(connection_string)
        # Test the connection
        await client.admin.command('ping')
        client.close()
        return True
        
    except Exception as e:
        logging.error(f"MongoDB connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the MongoDB integration
    async def main():
        print("🧪 Testing MongoDB Integration...")
        
        # Test connection
        is_connected = await test_mongodb_connection()
        if is_connected:
            print("✅ MongoDB connection successful")
        else:
            print("❌ MongoDB connection failed")
            return
        
        # Create manager
        manager = create_mongodb_manager()
        
        # Initialize database
        init_success = await manager.initialize_database()
        if init_success:
            print("✅ Database initialization successful")
        else:
            print("❌ Database initialization failed")
            return
        
        # Get stats
        stats = await manager.get_database_stats()
        print(f"📊 Database Stats: {stats}")
        
        # Close connections
        manager.close_connections()
        print("✅ MongoDB integration test completed")
    
    # Run the test
    asyncio.run(main())