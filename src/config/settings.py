"""Configuration management with YAML and CLI overrides."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class EncoderConfig:
    """Configuration for encoders."""
    dense_model: str = "sentence-transformers/allenai-specter"
    sparse_model: str = "naver/splade-v3"
    max_length: int = 512
    normalize_dense: bool = True
    device: Optional[str] = None  # None = auto-detect
    use_specter2: bool = False
    specter2_base_adapter: str = "allenai/specter2"  # For document embeddings
    specter2_query_adapter: str = "allenai/specter2_adhoc_query"  # For query embeddings


@dataclass
class IndexConfig:
    """Configuration for indexing."""
    batch_size: int = 16
    checkpoint_enabled: bool = True
    chunk_size: int = 10000  # Number of docs per chunk for sparse indexing
    sparse_encoder_batch_size: int = 4  # SPLADE encoder batch size (to manage VRAM)


@dataclass
class RerankerConfig:
    """Configuration for reranking."""
    enabled: bool = True
    type: str = "cross-encoder"  # 'cross-encoder' or 'jina'
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 100  # Number of candidates to rerank
    max_length: int = 512
    batch_size: int = 32
    instruction: Optional[str] = None  # Custom instruction for reranker (Jina only)
    # Score fusion weights (for two-stage reranking)
    pre_rerank_weight: float = 0.7  # Weight for pre-reranking scores (RRF + intent boosting)
    rerank_weight: float = 0.3  # Weight for reranker scores (cross-encoder)


@dataclass
class SearchConfig:
    """Configuration for search."""
    top_k: int = 10
    retrieval_k: int = 100  # For hybrid: retrieve this many from each before fusion
    rrf_k: int = 60  # RRF constant


@dataclass
class IntentBoostingConfig:
    """Configuration for intent-based boosting."""
    enabled: bool = True
    citation_weights: dict = field(default_factory=lambda: {
        'topical': 0.1,
        'sota': 0.2,
        'foundational': 0.3,
        'comparison': 0.1,
        'method_lookup': 0.1,
        'default': 0.1
    })
    date_weights: dict = field(default_factory=lambda: {
        'sota': 0.1,  # Favor recent papers
        'foundational': 0.1  # Favor older papers
    })
    min_year: int = 1990  # Minimum year in corpus for normalization


@dataclass
class TitleAuthorMatchingConfig:
    """Configuration for title and author fuzzy matching."""
    enabled: bool = True
    title_threshold: float = 0.5  # Jaccard similarity threshold for title match (token-based)
    author_threshold: float = 0.7  # Token overlap threshold for author match
    title_boost_weight: float = 1.0  # Boost to add for title match
    author_boost_weight: float = 1.0  # Boost to add for author match


@dataclass
class QueryRewritingConfig:
    """Configuration for LLM-based query rewriting and filter extraction."""
    enabled: bool = False
    model: str = "google/flan-t5-base"
    max_length: int = 128
    temperature: float = 0.3
    num_rewrites: int = 1  # Number of query rewrites to generate (1-5 recommended)
    device: Optional[str] = None  # None = auto-detect (local mode only)
    filter_confidence_threshold: float = 0.7  # Minimum confidence to apply filters (0-1)
    enable_citation_filters: bool = False  # Enable citation_count filters (requires citation data)
    enable_year_filters: bool = True  # Enable year filters
    # vLLM API settings
    use_vllm: bool = False  # Use vLLM API instead of loading model locally
    vllm_endpoint: str = "http://localhost:8000/v1"  # vLLM OpenAI-compatible endpoint
    vllm_timeout: int = 30  # Timeout for vLLM API calls (seconds)
    # AWS Bedrock API settings
    use_bedrock: bool = False  # Use AWS Bedrock API instead of local/vLLM
    bedrock_model_id: str = "mistral.mistral-7b-instruct-v0:2"  # Bedrock model ID
    bedrock_region: str = "ca-central-1"  # AWS region for Bedrock
    bedrock_max_tokens: int = 512  # Max tokens for Bedrock response


@dataclass
class DataConfig:
    """Configuration for data loading."""
    jsonl_file: str = "data/arxiv_1k.jsonl"
    text_key: str = "abstract"
    id_key: str = "id"
    title_key: Optional[str] = "title"
    # Metadata enhancement
    use_metadata: bool = False
    categories_key: Optional[str] = "categories"
    authors_key: Optional[str] = "authors"
    year_key: Optional[str] = "published_date"  # Field for year/date extraction
    metadata_template: str = "Title: {title}\n\nAuthors: {authors}\n\nYear: {year}\n\nCategories: {categories}\n\nAbstract: {abstract}"


@dataclass
class MilvusConfig:
    """Configuration for Milvus vector database."""
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "arxplorer_papers"
    # Index parameters
    dense_index_type: str = "IVF_FLAT"  # IVF_FLAT, HNSW, or FLAT
    dense_nlist: int = 1024  # Number of clusters for IVF
    dense_nprobe: int = 64  # Number of clusters to search
    sparse_index_type: str = "SPARSE_INVERTED_INDEX"
    # Connection settings
    connection_timeout: int = 30
    # Batch insert settings
    batch_size: int = 1000  # Insert documents in batches


@dataclass
class Config:
    """Main configuration container."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    query_rewriting: QueryRewritingConfig = field(default_factory=QueryRewritingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    intent_boosting: IntentBoostingConfig = field(default_factory=IntentBoostingConfig)
    title_author_matching: TitleAuthorMatchingConfig = field(default_factory=TitleAuthorMatchingConfig)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """Load configuration from YAML file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(
            encoder=EncoderConfig(**data.get('encoder', {})),
            index=IndexConfig(**data.get('index', {})),
            search=SearchConfig(**data.get('search', {})),
            reranker=RerankerConfig(**data.get('reranker', {})),
            query_rewriting=QueryRewritingConfig(**data.get('query_rewriting', {})),
            data=DataConfig(**data.get('data', {})),
            milvus=MilvusConfig(**data.get('milvus', {})),
            intent_boosting=IntentBoostingConfig(**data.get('intent_boosting', {})),
            title_author_matching=TitleAuthorMatchingConfig(**data.get('title_author_matching', {}))
        )
    
    @classmethod
    def default(cls) -> 'Config':
        """Create default configuration."""
        return cls()
    
    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        data = {
            'encoder': {
                'dense_model': self.encoder.dense_model,
                'sparse_model': self.encoder.sparse_model,
                'max_length': self.encoder.max_length,
                'normalize_dense': self.encoder.normalize_dense,
                'device': self.encoder.device
            },
            'index': {
                'batch_size': self.index.batch_size,
                'dense_output_dir': self.index.dense_output_dir,
                'sparse_output_dir': self.index.sparse_output_dir,
                'use_gpu_faiss': self.index.use_gpu_faiss,
                'checkpoint_enabled': self.index.checkpoint_enabled
            },
            'search': {
                'top_k': self.search.top_k,
                'retrieval_k': self.search.retrieval_k,
                'rrf_k': self.search.rrf_k
            },
            'reranker': {
                'enabled': self.reranker.enabled,
                'model': self.reranker.model,
                'rerank_top_k': self.reranker.rerank_top_k,
                'max_length': self.reranker.max_length,
                'batch_size': self.reranker.batch_size
            },
            'data': {
                'jsonl_file': self.data.jsonl_file,
                'text_key': self.data.text_key,
                'id_key': self.data.id_key,
                'title_key': self.data.title_key
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
