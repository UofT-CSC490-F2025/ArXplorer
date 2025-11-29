# ArXplorer - Academic Paper Retrieval with Milvus

![Test Coverage](coverage.svg)

A production-grade academic paper retrieval system using Milvus vector database for unified hybrid search (dense + sparse), with advanced query rewriting and multiple reranking options.

## Quick Start

### 1. Setup Milvus with Docker

**Install Docker Desktop** (if not already installed):
- Windows/Mac: Download from [docker.com](https://www.docker.com/products/docker-desktop/)
- Linux: Follow [official installation guide](https://docs.docker.com/engine/install/)

**Start Milvus standalone instance**:

```powershell
# Using Docker Compose (recommended)
docker-compose -f docker-compose.milvus.yml up -d

# Verify Milvus is running
docker-compose ps

# Expected output:
# NAME           IMAGE                PORTS
# milvus-milvus  milvusdb/milvus      0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp
# milvus-etcd    quay.io/coreos/etcd  2379/tcp, 2380/tcp
# milvus-minio   minio/minio          9000/tcp, 9001/tcp

# Check Milvus health
curl http://localhost:9091/healthz
# Should return: OK
```

**Milvus management** (optional):
```powershell
# Stop Milvus
docker-compose down

# Stop and remove all data (fresh start)
docker-compose down -v

# View logs
docker-compose logs -f milvus

# Restart Milvus
docker-compose restart
```

**Alternative: Docker run command** (if docker-compose not available):
```powershell
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -e ETCD_ENDPOINTS=etcd:2379 \
  -e MINIO_ADDRESS=minio:9000 \
  milvusdb/milvus:v2.4.15 milvus run standalone
```

### 2. Setup Python Environment

```powershell
# Create conda environment
conda env create -f environment.yml -n arxplorer-env
conda activate arxplorer-env
```

### 3. Prepare Dataset

Download the Kaggle arXiv dataset and extract papers to JSONL format:

```powershell
# 1. Download arXiv dataset from Kaggle
# Visit: https://www.kaggle.com/datasets/Cornell-University/arxiv
# Download and extract to: data/kaggle_arxiv/

# 2. Create filtered JSONL dataset (default: AI/ML categories)
python scripts/create_arxiv_dataset.py --limit 300000

# Or with custom categories
python scripts/create_arxiv_dataset.py --categories cs.ai,cs.lg,stat.ml --limit 100000

# Or include all categories
python scripts/create_arxiv_dataset.py --no-filter --limit 500000
```

This creates `data/arxiv_300k.jsonl` with standardized fields (id, title, abstract, categories, authors, year).

### 4. Build Milvus Index

```powershell
# Encode documents and load into Milvus (dense + sparse + metadata)
python scripts/encode.py --data-file data/arxiv_300k.jsonl

# Or use smaller dataset for testing
python scripts/encode.py --data-file data/arxiv_1k.jsonl
```

This will:
- Encode documents with SPECTER (dense) and SPLADE (sparse)
- Create Milvus collection with hybrid indexes
- Store all metadata (title, authors, year, categories, etc.)

### 5. Query the System

```powershell
# Interactive hybrid search with reranking (default)
python scripts/query.py

# With query rewriting and filter extraction
python scripts/query.py --rewrite-query

# Single query (non-interactive)
python scripts/query.py --query "attention is all you need"

# Override top-k results
python scripts/query.py --top-k 20

# Disable reranking for faster queries
python scripts/query.py --no-rerank
```

## Architecture

**Milvus-Only Architecture** (Production-Ready):

```
src/
├── config/          # YAML configuration management
├── data/            # Document dataclass and streaming JSONL loader
└── retrieval/
    ├── encoders/    # Dense (SPECTER/SPECTER2) and Sparse (SPLADE) encoders
    ├── indexers/    # MilvusIndexer - unified hybrid indexer
    ├── searchers/   # MilvusHybridSearcher - unified search with RRF
    ├── rerankers/   # CrossEncoderReranker, QwenReranker, JinaReranker
    └── query_rewriting/  # LLM-based filter extraction + query rewriting (Qwen)

scripts/
├── encode.py        # Build Milvus index (dense + sparse + metadata)
├── query.py         # Query Milvus with hybrid search + reranking
└── create_arxiv_dataset.py  # Extract papers from Kaggle dataset
```

**Milvus Unified Storage**:
- Dense vectors (SPECTER): IVF_FLAT index
- Sparse vectors (SPLADE): SPARSE_INVERTED_INDEX
- Metadata: title, abstract, authors, categories, year, citation_count
- Replaces: FAISS, scipy, doc_map.json, MongoDB

**Key Benefits**:
- Single source of truth for all data
- Built-in hybrid search with RRF fusion
- Scalar filtering (year, citations) integrated with vector search
- Production-ready with backup/restore capabilities

## Configuration

Edit `config.yaml` to customize:

```yaml
encoder:
  dense_model: "sentence-transformers/allenai-specter"
  sparse_model: "naver/splade-v3"
  use_specter2: true  # Use SPECTER2 with adapters
  specter2_base_adapter: "allenai/specter2"  # For documents
  specter2_query_adapter: "allenai/specter2_adhoc_query"  # For queries
  
index:
  batch_size: 16
  sparse_encoder_batch_size: 4  # SPLADE uses more VRAM

milvus:
  host: "localhost"
  port: 19530
  collection_name: "arxplorer_papers"
  dense_index_type: "IVF_FLAT"  # Or HNSW for >1M docs
  dense_nlist: 1024  # Number of IVF clusters
  sparse_index_type: "SPARSE_INVERTED_INDEX"
  batch_size: 1000  # Insert batch size
  
search:
  top_k: 10          # Final results to return
  rrf_k: 60          # RRF constant (default from literature)

reranker:
  enabled: true
  type: "jina"       # Options: cross-encoder, qwen, jina
  model: "jinaai/jina-reranker-v3"
  rerank_top_k: 50   # Number of candidates to rerank
  batch_size: 50     # Jina: MUST match rerank_top_k for listwise comparison!
  
query_rewriting:
  enabled: true
  model: "Qwen/Qwen3-4B-Instruct-2507-FP8"
  num_rewrites: 1    # Number of query variants
  filter_confidence_threshold: 0.7  # Minimum confidence for filters
  enable_year_filters: true   # Extract year constraints
  enable_citation_filters: false  # Extract citation constraints

data:
  jsonl_file: "data/arxiv_300k.jsonl"
  use_metadata: true  # Encode title + authors + year + categories
  metadata_template: 'Title: {title}\n\nAuthors: {authors}\n\nYear: {year}\n\nCategories: {categories}\n\nAbstract: {abstract}'
```

CLI arguments override config values:

```powershell
python scripts/encode.py --batch-size 32 --data-file data/custom.jsonl
python scripts/query.py --top-k 20 --no-rerank
```

## Key Features

### Milvus Vector Database
- **Unified storage**: Dense + sparse vectors + metadata in single collection
- **Hybrid search**: Built-in RRF fusion with configurable k parameter
- **Scalar filtering**: Year and citation filtering integrated with vector search
- **Production-ready**: Distributed architecture, backup/restore, high availability
- **Efficient indexing**: IVF_FLAT for dense (300k docs), SPARSE_INVERTED_INDEX for sparse

### LLM-Based Query Rewriting & Filter Extraction
- **Model**: Qwen3-4B-Instruct (instruction-tuned causal LM)
- **Dual functionality**:
  1. **Filter extraction**: Detects canonical intent ("original", "seminal") and extracts year/citation constraints
  2. **Query rewriting**: Generates alternative phrasings for better recall
- **Dynamic filtering**: `year >= 2017 and citation_count >= 5000` applied to Milvus search
- **Multi-query fusion**: Combines results from original + rewritten queries using RRF
- **Use case**: Especially effective for finding foundational papers with varied citation styles

### Advanced Reranking Options
Three reranker types available via config:

1. **Cross-Encoder** (fastest, 33M params)
   - Model: `cross-encoder/ms-marco-MiniLM-L-12-v2`
   - Best for: General queries, speed-critical applications
   - Latency: ~200-300ms for 50 docs

2. **Qwen Reranker** (powerful, 0.6B params, pairwise)
   - Model: `Qwen/Qwen3-Reranker-0.6B`
   - Best for: Semantic understanding, customizable instructions
   - Latency: ~800-1200ms for 50 docs (FP16 optimized)
   - Limitation: Pairwise scoring, can't compare years across documents

3. **Jina Reranker** (SOTA, 0.6B params, listwise)
   - Model: `jinaai/jina-reranker-v3`
   - Best for: Canonical queries, year-based ranking, multilingual
   - Latency: ~200-400ms for 50 docs
   - **Key advantage**: Listwise reranking - sees all documents at once, can compare years for "original" queries

### Multi-Stage Retrieval Pipeline
1. **Filter extraction (optional)**: LLM extracts year/citation constraints from query
2. **Query rewriting (optional)**: Generate diverse query variants with Qwen LLM
3. **Hybrid search**: Milvus RRF fusion of dense (SPECTER) + sparse (SPLADE) with filters
4. **Reranking**: Refine top candidates with cross-encoder/Qwen/Jina
5. **Citation boost (optional)**: Apply citation score adjustment to final results

### Metadata-Enhanced Embeddings
Documents encoded with full context for better retrieval:
```
Title: U-Net: Convolutional Networks for Biomedical Image Segmentation

Authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

Year: 2015

Categories: cs.CV

Abstract: [full abstract text...]
```

This enables:
- Exact title matching in dense/sparse search
- Author name queries
- Temporal context for rerankers (especially Jina's listwise comparison)

### Scalability
- **Streaming data loader**: Handles large JSONL files without loading all into memory
- **Batch processing**: Configurable batch sizes for encoding
- **Milvus distributed**: Can scale to billions of vectors (cluster mode)
- **Memory-efficient**: Sparse inverted index, IVF clustering for dense vectors
- **GPU acceleration**: CUDA support for all encoders and rerankers

## Performance Notes (300k corpus on RTX 4070 Super)

### Encoding Time
- **Dense (SPECTER)**: ~30-60 min for 300k docs (batch_size=16, GPU)
- **Sparse (SPLADE)**: ~60-120 min for 300k docs (batch_size=4, GPU)
- **Total encoding**: ~90-180 min for full hybrid index
- **GPU memory**: ~6-8 GB peak

### Query Latency
| Configuration | Latency (ms) | Use Case |
|--------------|--------------|----------|
| Hybrid (RRF only) | 100-200 | Fast exploration |
| + Cross-Encoder | 300-500 | General queries |
| + Qwen Reranker | 800-1200 | Semantic understanding |
| + Jina Reranker | 300-500 | Canonical queries |
| + Query Rewriting | +200-400 | Robustness boost |
| + Filters | -50-100 | Narrows search space |

### Milvus Performance
- **Collection size**: ~5-8 GB for 300k papers (vectors + metadata)
- **Index build**: ~2-5 min for 300k docs
- **Search latency**: ~50-100ms for hybrid search without reranking
- **Scalability**: Tested up to 300k docs, supports 10M+ with HNSW index

## Advanced Usage

### Query Rewriting with Filters

```powershell
# Enable query rewriting with filter extraction
python scripts/query.py --rewrite-query

# Example: "original unet paper"
# → Extracts filters: year <= 2016, citation_count >= 1000
# → Rewrites: "seminal U-Net segmentation architecture"
# → Searches with filters applied to Milvus

# Disable citation filters (if no citation data in Milvus)
# Edit config.yaml: enable_citation_filters: false

# Override filter confidence threshold
# Edit config.yaml: filter_confidence_threshold: 0.8
```

### Reranker Selection

```yaml
# In config.yaml

# Option 1: Cross-Encoder (fastest, general purpose)
reranker:
  type: cross-encoder
  model: cross-encoder/ms-marco-MiniLM-L-12-v2
  batch_size: 32

# Option 2: Qwen (powerful semantic understanding)
reranker:
  type: qwen
  model: Qwen/Qwen3-Reranker-0.6B
  batch_size: 8  # Lower due to long sequences
  instruction: null  # Custom instruction or null for default

# Option 3: Jina (best for canonical queries, listwise)
reranker:
  type: jina
  model: jinaai/jina-reranker-v3
  batch_size: 50  # MUST match rerank_top_k for listwise comparison!
```

### Custom Metadata Template

```yaml
# In config.yaml
data:
  use_metadata: true
  metadata_template: 'Title: {title}\n\nAuthors: {authors}\n\nYear: {year}\n\nCategories: {categories}\n\nAbstract: {abstract}'
```

After changing template, re-encode documents:
```powershell
python scripts/encode.py --data-file data/arxiv_300k.jsonl
```

## Troubleshooting

### Milvus Connection Issues

```powershell
# Check if Milvus is running
docker-compose ps

# Check Milvus health
curl http://localhost:9091/healthz

# View Milvus logs
docker-compose logs -f milvus

# Restart Milvus
docker-compose restart

# Fresh start (removes all data)
docker-compose down -v
docker-compose up -d
```

### Collection Not Found

```powershell
# Run encoding first to create collection
python scripts/encode.py --data-file data/arxiv_1k.jsonl

# Or check if collection exists
# In Python:
from pymilvus import connections, utility
connections.connect(host='localhost', port='19530')
print(utility.list_collections())
```

### Import Errors

Ensure you're running from repo root and conda env is active:
```powershell
conda activate arxplorer-env
cd "C:\Users\Kyle Vavasour\Desktop\CSC490-3"
python scripts/query.py
```

### CUDA Out of Memory

Reduce batch sizes in `config.yaml`:
```yaml
index:
  batch_size: 8  # Reduce from 16
  sparse_encoder_batch_size: 2  # Reduce from 4

reranker:
  batch_size: 4  # Reduce from 8 (Qwen) or 16 (others)
```

### Slow Queries

**Optimization strategies**:
1. Reduce `rerank_top_k` from 50 to 30
2. Disable reranking: `--no-rerank`
3. Use Cross-Encoder instead of Qwen for speed
4. Enable query rewriting only when needed
5. For >1M docs: Switch to HNSW index in `config.yaml`

### Missing Dependencies

```powershell
# Reinstall environment
conda env remove -n arxplorer-env
conda env create -f environment.yml -n arxplorer-env
conda activate arxplorer-env

# For SPECTER2 support
pip uninstall peft -y
pip install transformers==4.38.2 adapters
```

## Data Format

Input JSONL format (one JSON object per line):
```json
{"id": "arxiv:2004.07180", "title": "SPECTER: Document-level Representation Learning...", "abstract": "...", "authors": ["Arman Cohan", "..."], "categories": ["cs.CL"], "published_date": "2020-04-15"}
{"id": "arxiv:1706.03762", "title": "Attention Is All You Need", "abstract": "...", "authors": ["Ashish Vaswani", "..."], "categories": ["cs.CL", "cs.LG"], "published_date": "2017-06-12"}
```

Required fields (configurable via `config.yaml`):
- `id`: Document identifier
- `abstract` (or custom `text_key`): Text to encode
- `title`: Paper title
- `published_date`: Publication date (year extracted automatically)

Optional fields:
- `authors`: List of author names
- `categories`: List of categories (e.g., cs.AI, cs.CV)

## Milvus Schema

The Milvus collection stores:
- `id` (VARCHAR): Document ID
- `title` (VARCHAR): Paper title  
- `abstract` (VARCHAR): Paper abstract (truncated to 8192 chars)
- `authors` (ARRAY): List of author names
- `categories` (ARRAY): List of categories
- `year` (INT64): Publication year
- `citation_count` (INT64): Citation count (default 0)
- `dense_vector` (FLOAT_VECTOR): SPECTER embedding (768D)
- `sparse_vector` (SPARSE_FLOAT_VECTOR): SPLADE embedding (~30k dims, sparse)

Indexes:
- Dense: IVF_FLAT (nlist=1024) for ~300k docs, HNSW for >1M docs
- Sparse: SPARSE_INVERTED_INDEX (memory-efficient)

## Backup and Restore

**Backup Milvus data**:
```powershell
# Using Milvus Backup tool (requires separate installation)
# See: https://milvus.io/docs/milvus_backup_overview.md

# Or backup Docker volumes
docker-compose down
docker run --rm -v milvus-standalone:/data -v $(pwd)/backups:/backup ubuntu tar czf /backup/milvus-backup.tar.gz /data
```

**Restore from backup**:
```powershell
docker run --rm -v milvus-standalone:/data -v $(pwd)/backups:/backup ubuntu tar xzf /backup/milvus-backup.tar.gz -C /
docker-compose up -d
```

## References

- **Milvus**: [milvus.io](https://milvus.io/) - Open-source vector database
- **SPECTER**: [Cohan et al., 2020](https://arxiv.org/abs/2004.07180) - Document-level embeddings
- **SPECTER2**: [Singh et al., 2022](https://arxiv.org/abs/2209.07930) - Adapters for specialized embeddings
- **SPLADE**: [Formal et al., 2021](https://arxiv.org/abs/2107.05720) - Sparse lexical and expansion model
- **RRF**: Cormack et al., 2009 - Reciprocal Rank Fusion
- **Qwen 3**: [Qwen Team, 2025](https://arxiv.org/abs/2505.09388) - Multilingual LLM family
- **Jina Reranker v3**: [Wang et al., 2025](https://arxiv.org/abs/2509.25085) - Listwise document reranker 
