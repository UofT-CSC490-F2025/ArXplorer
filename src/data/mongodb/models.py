"""
MongoDB Data Models
Document schemas and validation for ArXplorer collections
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from bson import ObjectId

from ...core.schemas import ArXivPaper, ProcessedPaper, PaperEmbedding, SearchQuery, SearchResult


class MongoDocument:
    """Base class for MongoDB documents"""
    
    def to_mongo(self) -> Dict[str, Any]:
        """Convert to MongoDB document format"""
        doc = asdict(self) if hasattr(self, '__dataclass_fields__') else self.__dict__
        
        # Handle datetime serialization
        for key, value in doc.items():
            if isinstance(value, datetime):
                doc[key] = value
            elif isinstance(value, list) and value and isinstance(value[0], datetime):
                doc[key] = value
        
        return doc
    
    @classmethod
    def from_mongo(cls, doc: Dict[str, Any]):
        """Create instance from MongoDB document"""
        # Remove MongoDB _id field if present
        if '_id' in doc:
            del doc['_id']
        
        return cls(**doc)


@dataclass
class RawPaperDocument(MongoDocument):
    """MongoDB document model for raw papers collection"""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[Dict[str, str]]
    categories: List[str]
    submitted_date: datetime
    updated_date: Optional[datetime]
    doi: Optional[str]
    journal_ref: Optional[str]
    comments: Optional[str]
    status: str
    batch_id: str
    created_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    @classmethod
    def from_arxiv_paper(cls, paper: ArXivPaper, batch_id: str) -> 'RawPaperDocument':
        """Create from ArXivPaper schema"""
        return cls(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            abstract=paper.abstract,
            authors=[author.to_dict() for author in paper.authors],
            categories=paper.categories,
            submitted_date=paper.submitted_date,
            updated_date=paper.updated_date,
            doi=paper.doi,
            journal_ref=paper.journal_ref,
            comments=paper.comments,
            status=paper.status.value,
            batch_id=batch_id,
            created_at=datetime.now(),
            processed_at=paper.processed_at,
            error_message=paper.error_message
        )


@dataclass
class ProcessedPaperDocument(MongoDocument):
    """MongoDB document model for processed papers collection"""
    arxiv_id: str
    cleaned_title: str
    cleaned_abstract: str
    extracted_keywords: List[str]
    citation_count: int
    references: List[str]
    full_text_url: Optional[str]
    word_count: int
    language: str
    readability_score: Optional[float]
    batch_id: str
    created_at: datetime
    processing_time_ms: Optional[int] = None
    
    @classmethod
    def from_processed_paper(cls, paper: ProcessedPaper, batch_id: str, processing_time_ms: Optional[int] = None) -> 'ProcessedPaperDocument':
        """Create from ProcessedPaper schema"""
        return cls(
            arxiv_id=paper.arxiv_id,
            cleaned_title=paper.cleaned_title,
            cleaned_abstract=paper.cleaned_abstract,
            extracted_keywords=paper.extracted_keywords,
            citation_count=paper.citation_count,
            references=paper.references,
            full_text_url=paper.full_text_url,
            word_count=paper.word_count,
            language=paper.language,
            readability_score=paper.readability_score,
            batch_id=batch_id,
            created_at=datetime.now(),
            processing_time_ms=processing_time_ms
        )


@dataclass
class EmbeddingDocument(MongoDocument):
    """MongoDB document model for embeddings collection"""
    arxiv_id: str
    title_embedding: List[float]
    abstract_embedding: List[float]
    combined_embedding: List[float]
    model_name: str
    model_version: str
    embedding_dimension: int
    batch_id: str
    created_at: datetime
    generation_time_ms: Optional[int] = None
    
    @classmethod
    def from_paper_embedding(cls, embedding: PaperEmbedding, batch_id: str, generation_time_ms: Optional[int] = None) -> 'EmbeddingDocument':
        """Create from PaperEmbedding schema"""
        return cls(
            arxiv_id=embedding.arxiv_id,
            title_embedding=embedding.title_embedding,
            abstract_embedding=embedding.abstract_embedding,
            combined_embedding=embedding.combined_embedding,
            model_name=embedding.model_name,
            model_version=embedding.model_version,
            embedding_dimension=embedding.embedding_dimension,
            batch_id=batch_id,
            created_at=embedding.created_at,
            generation_time_ms=generation_time_ms
        )


@dataclass
class SearchQueryDocument(MongoDocument):
    """MongoDB document model for search queries collection"""
    query_id: str
    raw_query: str
    processed_query: str
    query_embedding: List[float]
    filters: Dict[str, Any]
    user_id: Optional[str]
    session_id: Optional[str]
    created_at: datetime
    execution_time_ms: Optional[int] = None
    result_count: Optional[int] = None
    
    @classmethod
    def from_search_query(cls, query: SearchQuery, execution_time_ms: Optional[int] = None, result_count: Optional[int] = None) -> 'SearchQueryDocument':
        """Create from SearchQuery schema"""
        return cls(
            query_id=query.query_id,
            raw_query=query.raw_query,
            processed_query=query.processed_query,
            query_embedding=query.query_embedding,
            filters=query.filters,
            user_id=query.user_id,
            session_id=query.session_id,
            created_at=query.timestamp,
            execution_time_ms=execution_time_ms,
            result_count=result_count
        )


@dataclass
class SearchResultDocument(MongoDocument):
    """MongoDB document model for search results collection"""
    query_id: str
    arxiv_id: str
    relevance_score: float
    similarity_score: float
    boost_factors: Dict[str, float]
    rank: int
    created_at: datetime
    
    @classmethod
    def from_search_result(cls, result: SearchResult, query_id: str, rank: int) -> 'SearchResultDocument':
        """Create from SearchResult schema"""
        return cls(
            query_id=query_id,
            arxiv_id=result.arxiv_id,
            relevance_score=result.relevance_score,
            similarity_score=result.similarity_score,
            boost_factors=result.boost_factors,
            rank=rank,
            created_at=datetime.now()
        )


@dataclass
class PipelineRunDocument(MongoDocument):
    """MongoDB document model for pipeline runs collection"""
    batch_id: str
    pipeline_version: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # "running", "completed", "failed"
    config: Dict[str, Any]
    stats: Dict[str, Any]
    error_message: Optional[str] = None
    
    # Processing stages
    ingestion_stats: Optional[Dict[str, Any]] = None
    processing_stats: Optional[Dict[str, Any]] = None
    embedding_stats: Optional[Dict[str, Any]] = None
    indexing_stats: Optional[Dict[str, Any]] = None


@dataclass
class UserFeedbackDocument(MongoDocument):
    """MongoDB document model for user feedback collection"""
    feedback_id: str
    query_id: str
    arxiv_id: str
    user_id: str
    feedback_type: str  # "relevant", "not_relevant", "bookmark", "rating"
    feedback_value: Any  # boolean for relevance, 1-5 for rating, etc.
    comments: Optional[str]
    created_at: datetime
    session_id: Optional[str] = None


@dataclass
class AnalyticsDocument(MongoDocument):
    """MongoDB document model for analytics collection"""
    metric_name: str
    metric_value: Any
    category: str
    subcategory: Optional[str]
    date: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    # Common analytics metrics:
    # - paper_ingestion_rate
    # - search_query_count
    # - average_search_time
    # - popular_categories
    # - user_engagement_metrics