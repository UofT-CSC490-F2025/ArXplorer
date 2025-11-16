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
    specter2_adapter: str = "allenai/specter2"


@dataclass
class IndexConfig:
    """Configuration for indexing."""
    batch_size: int = 16
    dense_output_dir: str = "data/dense_index"
    sparse_output_dir: str = "data/sparse_index"
    use_gpu_faiss: bool = False
    checkpoint_enabled: bool = True
    chunk_size: int = 10000  # Number of docs per chunk for sparse indexing
    sparse_encoder_batch_size: int = 4  # SPLADE encoder batch size (to manage VRAM)


@dataclass
class RerankerConfig:
    """Configuration for reranking."""
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 100  # Number of candidates to rerank
    max_length: int = 512
    batch_size: int = 32


@dataclass
class SearchConfig:
    """Configuration for search."""
    top_k: int = 10
    retrieval_k: int = 100  # For hybrid: retrieve this many from each before fusion
    rrf_k: int = 60  # RRF constant


@dataclass
class QueryRewritingConfig:
    """Configuration for LLM-based query rewriting."""
    enabled: bool = False
    model: str = "google/flan-t5-base"
    max_length: int = 128
    temperature: float = 0.3
    num_rewrites: int = 1  # Number of query rewrites to generate (1-5 recommended)
    device: Optional[str] = None  # None = auto-detect


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
    metadata_template: str = "Title: {title}\n\nAuthors: {authors}\n\nCategories: {categories}\n\nAbstract: {abstract}"


@dataclass
class Config:
    """Main configuration container."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    query_rewriting: QueryRewritingConfig = field(default_factory=QueryRewritingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
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
            data=DataConfig(**data.get('data', {}))
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
