"""
MongoDB Client Configuration and Connection Management
Handles connection pooling, authentication, and database setup
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import os

import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from motor.motor_asyncio import AsyncIOMotorClient


class MongoDBClient:
    """
    Centralized MongoDB client with connection management
    Supports both sync and async operations
    """
    
    def __init__(self, 
                 connection_string: Optional[str] = None,
                 database_name: str = "arxplorer",
                 use_async: bool = True):
        """
        Initialize MongoDB client
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Target database name
            use_async: Whether to use async client
        """
        self.connection_string = connection_string or os.getenv(
            'MONGODB_CONNECTION_STRING', 
            'mongodb+srv://arxplorer_db_user:E14bupBhNORll6QT@arxplorercluster.sv4wks6.mongodb.net/arxplorer'
        )
        self.database_name = database_name
        self.use_async = use_async
        self.logger = logging.getLogger(__name__)
        
        # Collection names
        self.collections = {
            'raw_papers': 'raw_papers',
            'processed_papers': 'processed_papers',
            'embeddings': 'embeddings',
            'search_queries': 'search_queries',
            'search_results': 'search_results',
            'pipeline_runs': 'pipeline_runs',
            'user_feedback': 'user_feedback',
            'analytics': 'analytics'
        }
        
        # Initialize clients
        self._sync_client = None
        self._async_client = None
        self._sync_db = None
        self._async_db = None
    
    @property
    def sync_client(self) -> MongoClient:
        """Get synchronous MongoDB client"""
        if self._sync_client is None:
            self._sync_client = MongoClient(
                self.connection_string,
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000,
                connectTimeoutMS=10000,
                serverSelectionTimeoutMS=10000
            )
        return self._sync_client
    
    @property
    def async_client(self) -> AsyncIOMotorClient:
        """Get asynchronous MongoDB client"""
        if self._async_client is None:
            self._async_client = AsyncIOMotorClient(
                self.connection_string,
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000,
                connectTimeoutMS=10000,
                serverSelectionTimeoutMS=10000
            )
        return self._async_client
    
    @property
    def db(self):
        """Get database (sync or async based on configuration)"""
        if self.use_async:
            if self._async_db is None:
                self._async_db = self.async_client[self.database_name]
            return self._async_db
        else:
            if self._sync_db is None:
                self._sync_db = self.sync_client[self.database_name]
            return self._sync_db
    
    async def initialize_database(self) -> bool:
        """
        Initialize database with proper indexes and collections
        
        Returns:
            bool: Success status
        """
        try:
            db = self.db
            
            # Raw Papers Collection
            raw_papers = db[self.collections['raw_papers']]
            if self.use_async:
                await raw_papers.create_index([("arxiv_id", ASCENDING)], unique=True)
                await raw_papers.create_index([("submitted_date", DESCENDING)])
                await raw_papers.create_index([("categories", ASCENDING)])
                await raw_papers.create_index([("status", ASCENDING)])
                await raw_papers.create_index([("batch_id", ASCENDING)])
                await raw_papers.create_index([("title", TEXT), ("abstract", TEXT)])
            else:
                raw_papers.create_index([("arxiv_id", ASCENDING)], unique=True)
                raw_papers.create_index([("submitted_date", DESCENDING)])
                raw_papers.create_index([("categories", ASCENDING)])
                raw_papers.create_index([("status", ASCENDING)])
                raw_papers.create_index([("batch_id", ASCENDING)])
                raw_papers.create_index([("title", TEXT), ("abstract", TEXT)])
            
            # Processed Papers Collection
            processed_papers = db[self.collections['processed_papers']]
            if self.use_async:
                await processed_papers.create_index([("arxiv_id", ASCENDING)], unique=True)
                await processed_papers.create_index([("word_count", DESCENDING)])
                await processed_papers.create_index([("readability_score", DESCENDING)])
                await processed_papers.create_index([("language", ASCENDING)])
                await processed_papers.create_index([("extracted_keywords", ASCENDING)])
            else:
                processed_papers.create_index([("arxiv_id", ASCENDING)], unique=True)
                processed_papers.create_index([("word_count", DESCENDING)])
                processed_papers.create_index([("readability_score", DESCENDING)])
                processed_papers.create_index([("language", ASCENDING)])
                processed_papers.create_index([("extracted_keywords", ASCENDING)])
            
            # Embeddings Collection
            embeddings = db[self.collections['embeddings']]
            if self.use_async:
                await embeddings.create_index([("arxiv_id", ASCENDING)], unique=True)
                await embeddings.create_index([("model_name", ASCENDING)])
                await embeddings.create_index([("created_at", DESCENDING)])
                await embeddings.create_index([("embedding_dimension", ASCENDING)])
            else:
                embeddings.create_index([("arxiv_id", ASCENDING)], unique=True)
                embeddings.create_index([("model_name", ASCENDING)])
                embeddings.create_index([("created_at", DESCENDING)])
                embeddings.create_index([("embedding_dimension", ASCENDING)])
            
            # Search Queries Collection
            search_queries = db[self.collections['search_queries']]
            if self.use_async:
                await search_queries.create_index([("query_id", ASCENDING)], unique=True)
                await search_queries.create_index([("created_at", DESCENDING)])
                await search_queries.create_index([("user_id", ASCENDING)])
                await search_queries.create_index([("raw_query", TEXT)])
            else:
                search_queries.create_index([("query_id", ASCENDING)], unique=True)
                search_queries.create_index([("created_at", DESCENDING)])
                search_queries.create_index([("user_id", ASCENDING)])
                search_queries.create_index([("raw_query", TEXT)])
            
            # Search Results Collection
            search_results = db[self.collections['search_results']]
            if self.use_async:
                await search_results.create_index([("query_id", ASCENDING)])
                await search_results.create_index([("arxiv_id", ASCENDING)])
                await search_results.create_index([("relevance_score", DESCENDING)])
                await search_results.create_index([("created_at", DESCENDING)])
            else:
                search_results.create_index([("query_id", ASCENDING)])
                search_results.create_index([("arxiv_id", ASCENDING)])
                search_results.create_index([("relevance_score", DESCENDING)])
                search_results.create_index([("created_at", DESCENDING)])
            
            # Pipeline Runs Collection
            pipeline_runs = db[self.collections['pipeline_runs']]
            if self.use_async:
                await pipeline_runs.create_index([("batch_id", ASCENDING)], unique=True)
                await pipeline_runs.create_index([("start_time", DESCENDING)])
                await pipeline_runs.create_index([("status", ASCENDING)])
                await pipeline_runs.create_index([("pipeline_version", ASCENDING)])
            else:
                pipeline_runs.create_index([("batch_id", ASCENDING)], unique=True)
                pipeline_runs.create_index([("start_time", DESCENDING)])
                pipeline_runs.create_index([("status", ASCENDING)])
                pipeline_runs.create_index([("pipeline_version", ASCENDING)])
            
            # User Feedback Collection
            user_feedback = db[self.collections['user_feedback']]
            if self.use_async:
                await user_feedback.create_index([("query_id", ASCENDING)])
                await user_feedback.create_index([("arxiv_id", ASCENDING)])
                await user_feedback.create_index([("user_id", ASCENDING)])
                await user_feedback.create_index([("created_at", DESCENDING)])
                await user_feedback.create_index([("feedback_type", ASCENDING)])
            else:
                user_feedback.create_index([("query_id", ASCENDING)])
                user_feedback.create_index([("arxiv_id", ASCENDING)])
                user_feedback.create_index([("user_id", ASCENDING)])
                user_feedback.create_index([("created_at", DESCENDING)])
                user_feedback.create_index([("feedback_type", ASCENDING)])
            
            # Analytics Collection
            analytics = db[self.collections['analytics']]
            if self.use_async:
                await analytics.create_index([("metric_name", ASCENDING)])
                await analytics.create_index([("date", DESCENDING)])
                await analytics.create_index([("category", ASCENDING)])
            else:
                analytics.create_index([("metric_name", ASCENDING)])
                analytics.create_index([("date", DESCENDING)])
                analytics.create_index([("category", ASCENDING)])
            
            self.logger.info("MongoDB database initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MongoDB database: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check database health and return status information
        
        Returns:
            Dict with health status, connection info, and collection stats
        """
        try:
            # Test connection
            if self.use_async:
                await self.async_client.admin.command('ping')
                db = self.async_db
            else:
                self.sync_client.admin.command('ping')
                db = self.sync_db
            
            # Get collection stats
            collection_stats = {}
            for name, collection_name in self.collections.items():
                collection = db[collection_name]
                if self.use_async:
                    count = await collection.count_documents({})
                else:
                    count = collection.count_documents({})
                collection_stats[name] = count
            
            return {
                "status": "healthy",
                "database": self.database_name,
                "collections": collection_stats,
                "connection_string": self.connection_string.split('@')[1] if '@' in self.connection_string else "localhost"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database": self.database_name
            }
    
    def close(self):
        """Close all database connections"""
        if self._sync_client:
            self._sync_client.close()
        if self._async_client:
            self._async_client.close()
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.close()