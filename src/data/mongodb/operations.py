"""
MongoDB Operations - Repository Pattern Implementation
Provides high-level database operations for ArXplorer data
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Iterator, Tuple
import uuid

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, BulkWriteError

from .client import MongoDBClient
from .models import (
    RawPaperDocument, ProcessedPaperDocument, EmbeddingDocument,
    SearchQueryDocument, SearchResultDocument, PipelineRunDocument,
    UserFeedbackDocument, AnalyticsDocument
)
from ...core.schemas import ArXivPaper, ProcessedPaper, PaperEmbedding, SearchQuery, SearchResult


class BaseRepository:
    """Base repository with common operations"""
    
    def __init__(self, client: MongoDBClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.logger = logging.getLogger(f"{__name__}.{collection_name}")
    
    @property
    def collection(self):
        """Get the MongoDB collection"""
        return self.client.db[self.collection_name]
    
    async def count(self, filter_query: Dict[str, Any] = None) -> int:
        """Count documents matching filter"""
        filter_query = filter_query or {}
        if self.client.use_async:
            return await self.collection.count_documents(filter_query)
        else:
            return self.collection.count_documents(filter_query)


class PaperRepository(BaseRepository):
    """Repository for raw and processed papers"""
    
    def __init__(self, client: MongoDBClient):
        super().__init__(client, client.collections['raw_papers'])
        self.processed_collection = client.db[client.collections['processed_papers']]
    
    async def insert_raw_papers(self, papers: List[ArXivPaper], batch_id: str) -> Tuple[int, int]:
        """
        Insert raw papers with duplicate handling
        
        Returns:
            Tuple of (inserted_count, duplicate_count)
        """
        if not papers:
            return 0, 0
        
        documents = [
            RawPaperDocument.from_arxiv_paper(paper, batch_id).to_mongo()
            for paper in papers
        ]
        
        try:
            if self.client.use_async:
                result = await self.collection.insert_many(documents, ordered=False)
                inserted_count = len(result.inserted_ids)
            else:
                result = self.collection.insert_many(documents, ordered=False)
                inserted_count = len(result.inserted_ids)
                
            duplicate_count = len(documents) - inserted_count
            
            self.logger.info(f"Inserted {inserted_count} papers, {duplicate_count} duplicates skipped")
            return inserted_count, duplicate_count
            
        except BulkWriteError as e:
            inserted_count = e.details['nInserted']
            duplicate_count = len(documents) - inserted_count
            
            self.logger.info(f"Bulk insert completed with {inserted_count} inserted, {duplicate_count} duplicates")
            return inserted_count, duplicate_count
    
    async def get_raw_papers(self, 
                             status: Optional[str] = None,
                             batch_id: Optional[str] = None,
                             limit: int = 1000,
                             skip: int = 0) -> List[Dict[str, Any]]:
        """Get raw papers with filtering"""
        query = {}
        if status:
            query['status'] = status
        if batch_id:
            query['batch_id'] = batch_id
        
        if self.client.use_async:
            cursor = self.collection.find(query).skip(skip).limit(limit)
            return await cursor.to_list(length=limit)
        else:
            cursor = self.collection.find(query).skip(skip).limit(limit)
            return list(cursor)
    
    async def update_paper_status(self, arxiv_id: str, status: str, error_message: Optional[str] = None) -> bool:
        """Update paper processing status"""
        update_doc = {
            'status': status,
            'processed_at': datetime.now()
        }
        if error_message:
            update_doc['error_message'] = error_message
        
        if self.client.use_async:
            result = await self.collection.update_one(
                {'arxiv_id': arxiv_id},
                {'$set': update_doc}
            )
        else:
            result = self.collection.update_one(
                {'arxiv_id': arxiv_id},
                {'$set': update_doc}
            )
        
        return result.modified_count > 0
    
    async def insert_processed_papers(self, papers: List[ProcessedPaper], batch_id: str) -> int:
        """Insert processed papers"""
        if not papers:
            return 0
        
        documents = [
            ProcessedPaperDocument.from_processed_paper(paper, batch_id).to_mongo()
            for paper in papers
        ]
        
        try:
            if self.client.use_async:
                result = await self.processed_collection.insert_many(documents, ordered=False)
            else:
                result = self.processed_collection.insert_many(documents, ordered=False)
            
            return len(result.inserted_ids)
            
        except BulkWriteError as e:
            return e.details['nInserted']
    
    async def get_papers_by_categories(self, categories: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        """Get papers by arXiv categories"""
        query = {'categories': {'$in': categories}}
        
        if self.client.use_async:
            cursor = self.collection.find(query).limit(limit)
            return await cursor.to_list(length=limit)
        else:
            return list(self.collection.find(query).limit(limit))
    
    async def search_papers_by_text(self, search_text: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Full-text search on title and abstract"""
        query = {'$text': {'$search': search_text}}
        
        if self.client.use_async:
            cursor = self.collection.find(query).limit(limit)
            return await cursor.to_list(length=limit)
        else:
            return list(self.collection.find(query).limit(limit))


class EmbeddingRepository(BaseRepository):
    """Repository for paper embeddings"""
    
    def __init__(self, client: MongoDBClient):
        super().__init__(client, client.collections['embeddings'])
    
    async def insert_embeddings(self, embeddings: List[PaperEmbedding], batch_id: str) -> int:
        """Insert paper embeddings"""
        if not embeddings:
            return 0
        
        documents = [
            EmbeddingDocument.from_paper_embedding(embedding, batch_id).to_mongo()
            for embedding in embeddings
        ]
        
        try:
            if self.client.use_async:
                result = await self.collection.insert_many(documents, ordered=False)
            else:
                result = self.collection.insert_many(documents, ordered=False)
            
            return len(result.inserted_ids)
            
        except BulkWriteError as e:
            return e.details['nInserted']
    
    async def get_embeddings(self, 
                            arxiv_ids: Optional[List[str]] = None,
                            model_name: Optional[str] = None,
                            limit: int = 1000) -> List[Dict[str, Any]]:
        """Get embeddings with filtering"""
        query = {}
        if arxiv_ids:
            query['arxiv_id'] = {'$in': arxiv_ids}
        if model_name:
            query['model_name'] = model_name
        
        if self.client.use_async:
            cursor = self.collection.find(query).limit(limit)
            return await cursor.to_list(length=limit)
        else:
            return list(self.collection.find(query).limit(limit))
    
    async def get_embedding_by_arxiv_id(self, arxiv_id: str, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get single embedding by arXiv ID"""
        query = {'arxiv_id': arxiv_id}
        if model_name:
            query['model_name'] = model_name
        
        if self.client.use_async:
            return await self.collection.find_one(query)
        else:
            return self.collection.find_one(query)


class SearchRepository(BaseRepository):
    """Repository for search queries and results"""
    
    def __init__(self, client: MongoDBClient):
        super().__init__(client, client.collections['search_queries'])
        self.results_collection = client.db[client.collections['search_results']]
    
    async def insert_search_query(self, query: SearchQuery, execution_time_ms: Optional[int] = None) -> str:
        """Insert search query and return query_id"""
        document = SearchQueryDocument.from_search_query(query, execution_time_ms).to_mongo()
        
        if self.client.use_async:
            await self.collection.insert_one(document)
        else:
            self.collection.insert_one(document)
        
        return query.query_id
    
    async def insert_search_results(self, query_id: str, results: List[SearchResult]) -> int:
        """Insert search results for a query"""
        if not results:
            return 0
        
        documents = [
            SearchResultDocument.from_search_result(result, query_id, rank).to_mongo()
            for rank, result in enumerate(results, 1)
        ]
        
        try:
            if self.client.use_async:
                result = await self.results_collection.insert_many(documents)
            else:
                result = self.results_collection.insert_many(documents)
            
            return len(result.inserted_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to insert search results: {e}")
            return 0
    
    async def get_search_history(self, 
                                user_id: Optional[str] = None,
                                session_id: Optional[str] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Get search history"""
        query = {}
        if user_id:
            query['user_id'] = user_id
        if session_id:
            query['session_id'] = session_id
        
        if self.client.use_async:
            cursor = self.collection.find(query).sort('created_at', DESCENDING).limit(limit)
            return await cursor.to_list(length=limit)
        else:
            return list(self.collection.find(query).sort('created_at', DESCENDING).limit(limit))
    
    async def get_popular_queries(self, days: int = 7, limit: int = 20) -> List[Dict[str, Any]]:
        """Get popular search queries from recent days"""
        start_date = datetime.now() - timedelta(days=days)
        
        pipeline = [
            {'$match': {'created_at': {'$gte': start_date}}},
            {'$group': {
                '_id': '$raw_query',
                'count': {'$sum': 1},
                'avg_execution_time': {'$avg': '$execution_time_ms'},
                'last_searched': {'$max': '$created_at'}
            }},
            {'$sort': {'count': -1}},
            {'$limit': limit}
        ]
        
        if self.client.use_async:
            cursor = self.collection.aggregate(pipeline)
            return await cursor.to_list(length=limit)
        else:
            return list(self.collection.aggregate(pipeline))


class PipelineRepository(BaseRepository):
    """Repository for pipeline runs and analytics"""
    
    def __init__(self, client: MongoDBClient):
        super().__init__(client, client.collections['pipeline_runs'])
        self.analytics_collection = client.db[client.collections['analytics']]
        self.feedback_collection = client.db[client.collections['user_feedback']]
    
    async def create_pipeline_run(self, batch_id: str, config: Dict[str, Any], pipeline_version: str = "1.0") -> str:
        """Create new pipeline run record"""
        document = PipelineRunDocument(
            batch_id=batch_id,
            pipeline_version=pipeline_version,
            start_time=datetime.now(),
            end_time=None,
            status="running",
            config=config,
            stats={}
        ).to_mongo()
        
        if self.client.use_async:
            await self.collection.insert_one(document)
        else:
            self.collection.insert_one(document)
        
        return batch_id
    
    async def update_pipeline_status(self, batch_id: str, status: str, stats: Optional[Dict[str, Any]] = None, error_message: Optional[str] = None):
        """Update pipeline run status"""
        update_doc = {
            'status': status,
            'end_time': datetime.now() if status in ['completed', 'failed'] else None
        }
        
        if stats:
            update_doc['stats'] = stats
        if error_message:
            update_doc['error_message'] = error_message
        
        if self.client.use_async:
            await self.collection.update_one(
                {'batch_id': batch_id},
                {'$set': update_doc}
            )
        else:
            self.collection.update_one(
                {'batch_id': batch_id},
                {'$set': update_doc}
            )
    
    async def get_pipeline_runs(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get pipeline run history"""
        query = {}
        if status:
            query['status'] = status
        
        if self.client.use_async:
            cursor = self.collection.find(query).sort('start_time', DESCENDING).limit(limit)
            return await cursor.to_list(length=limit)
        else:
            return list(self.collection.find(query).sort('start_time', DESCENDING).limit(limit))
    
    async def record_analytics(self, metric_name: str, metric_value: Any, category: str, subcategory: Optional[str] = None):
        """Record analytics metric"""
        document = AnalyticsDocument(
            metric_name=metric_name,
            metric_value=metric_value,
            category=category,
            subcategory=subcategory,
            date=datetime.now()
        ).to_mongo()
        
        if self.client.use_async:
            await self.analytics_collection.insert_one(document)
        else:
            self.analytics_collection.insert_one(document)
    
    async def get_analytics(self, 
                           metric_name: Optional[str] = None,
                           category: Optional[str] = None,
                           days: int = 30) -> List[Dict[str, Any]]:
        """Get analytics data"""
        start_date = datetime.now() - timedelta(days=days)
        query = {'date': {'$gte': start_date}}
        
        if metric_name:
            query['metric_name'] = metric_name
        if category:
            query['category'] = category
        
        if self.client.use_async:
            cursor = self.analytics_collection.find(query).sort('date', DESCENDING)
            return await cursor.to_list(length=1000)
        else:
            return list(self.analytics_collection.find(query).sort('date', DESCENDING))
    
    async def record_user_feedback(self, query_id: str, arxiv_id: str, user_id: str, 
                                  feedback_type: str, feedback_value: Any, comments: Optional[str] = None):
        """Record user feedback"""
        feedback_id = str(uuid.uuid4())
        
        document = UserFeedbackDocument(
            feedback_id=feedback_id,
            query_id=query_id,
            arxiv_id=arxiv_id,
            user_id=user_id,
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            comments=comments,
            created_at=datetime.now()
        ).to_mongo()
        
        if self.client.use_async:
            await self.feedback_collection.insert_one(document)
        else:
            self.feedback_collection.insert_one(document)
        
        return feedback_id