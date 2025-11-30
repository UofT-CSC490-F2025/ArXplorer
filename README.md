# ArXplorer - Academic Paper Retrieval with Milvus

Academic paper retrieval system using Milvus vector database for unified hybrid search (dense + sparse), with LLM-based query analysis, intent-aware boosting, and advanced reranking.

## Deployment Options

### Local Development
See [Quick Start](#quick-start) below for local Docker setup.

### AWS Production Deployment
Deploy to AWS EC2 with Terraform for production workloads:
- **vLLM**: GPU instance (g5.xlarge) for fast LLM inference
- **Milvus**: CPU instance (c5.2xlarge) with EBS storage and S3 backups
- **Cost**: ~$1,035/month (24/7) or ~$350-400/month (spot instances)

See [AWS Deployment Guide](infrastructure/AWS_DEPLOYMENT.md) for detailed setup.

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
    ├── encoders/    # Dense (SPECTER2) and Sparse (SPLADE) encoders
    ├── indexers/    # MilvusIndexer - unified hybrid indexer
    ├── searchers/   # MilvusHybridSearcher - unified search with RRF
    ├── rerankers/   # JinaReranker (default), CrossEncoderReranker (fallback)
    │                # IntentBooster, TitleAuthorMatcher
    └── query_rewriting/  # LLMQueryRewriter - intent detection + query expansion (Qwen)

scripts/
├── encode.py                    # Build Milvus index (dense + sparse + metadata)
├── query.py                     # Query with hybrid search + reranking
├── fetch_citations_openalex.py  # Fetch citation data from OpenAlex API
└── create_arxiv_dataset.py      # Extract papers from Kaggle dataset
```

**Milvus Unified Storage**:
- Dense vectors (SPECTER2): IVF_FLAT index
- Sparse vectors (SPLADE): SPARSE_INVERTED_INDEX
- Metadata: title, abstract, authors, categories, year, citation_count

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
  retrieval_k: 200   # Number of candidates to retrieve for reranking

intent_boosting:
  enabled: true
  # Citation boost weights per intent (topical: 0.1, foundational: 0.3, etc.)
  # Date boost weights per intent (foundational: 0.1 favor older, sota: 0.1 favor recent)

title_author_matching:
  enabled: true      # Fuzzy title/author matching (specific_paper, foundational intents)
  title_threshold: 0.5   # Jaccard similarity (token-based)
  author_threshold: 0.7  # Token overlap threshold
  title_boost_weight: 1.0
  author_boost_weight: 1.0

reranker:
  enabled: true
  type: "jina"       # Options: jina (default), cross-encoder (fallback)
  model: "jinaai/jina-reranker-v3"
  rerank_top_k: 50   # Number of candidates to rerank
  batch_size: 50     # Jina: MUST match rerank_top_k for listwise comparison!
  pre_rerank_weight: 0.7  # Weight for boosted RRF scores
  rerank_weight: 0.3      # Weight for reranker scores
  
query_rewriting:
  enabled: true
  model: "Qwen/Qwen3-4B-AWQ"  # Quantized for faster inference
  max_length: 128
  temperature: 0.3
  num_rewrites: 1    # Number of query variants
  use_vllm: false    # Use vLLM API for 3-10x speedup
  vllm_endpoint: http://localhost:8000/v1

data:
  jsonl_file: "data/arxiv_1k.jsonl"
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

### LLM-Based Query Analysis
- **Model**: Qwen3-4B-AWQ (quantized, causal LM)
- **Single-call extraction**: Intent detection, year/citation filters, title/author extraction, query rewrites
- **Intent types**: topical, sota, foundational, comparison, method_lookup, specific_paper
- **Structured output**: JSON format with target_title, target_authors, year_constraint, citation_threshold
- **Multi-query search**: Original + rewrites + extracted title/authors as additional queries
- **Use case**: "original unet paper" → extracts title, authors, filters, generates semantic variants

### Intent-Based Boosting
- **Post-RRF scoring adjustment**: Citation and date weights applied per intent
- **Citation weighting**: Topical (0.1), foundational (0.3), specific_paper (0.15)
- **Date weighting**: SOTA favors recent (0.1), foundational favors older (0.1)
- **Normalized scoring**: All scores normalized to [0, 1] before fusion

### Title/Author Fuzzy Matching
- **Token-based Jaccard similarity**: Fast, accurate matching for specific paper queries
- **Dual matching**: Title (threshold 0.5) + Authors (threshold 0.7)
- **Intent-aware**: Only applies to specific_paper and foundational intents
- **Fallback**: If LLM doesn't extract title/authors, uses original query for matching
- **No double-boosting**: Mutual exclusivity between LLM-based and query-based matching

### Advanced Reranking
Two reranker types available:

1. **Jina Reranker** (default, listwise)
   - Model: `jinaai/jina-reranker-v3` (0.6B params)
   - **Listwise ranking**: Sees all documents at once, can compare across results
   - Best for: Canonical queries, year-based ranking, multilingual
   - Latency: ~200-400ms for 50 docs

2. **Cross-Encoder** (fallback, pairwise)
   - Model: `cross-encoder/ms-marco-MiniLM-L-12-v2` (33M params)
   - Best for: General queries, speed-critical applications
   - Latency: ~200-300ms for 50 docs

### Multi-Stage Retrieval Pipeline
1. **LLM query analysis**: Extract intent, filters, title/authors, generate rewrites
2. **Multi-query hybrid search**: Milvus RRF fusion across original + rewrites + title/authors
3. **Intent-based boosting**: Citation + date weighting per intent
4. **Title/author matching**: Fuzzy matching boost (if applicable)
5. **Jina reranking**: Listwise refinement of top candidates
6. **Weighted fusion**: 0.7 pre-rerank + 0.3 rerank scores
7. **Return top-k**: Final ranked results

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
- Temporal context for rerankers

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
| + Intent Boosting | +10-20 | Minimal overhead |
| + Title/Author Matching | +20-50 | Specific paper queries |
| + Jina Reranker | 300-500 | Default configuration |
| + Cross-Encoder | 300-500 | Alternative reranker |
| + Query Rewriting | +300-1500 | Robustness boost (local model) |
| + Query Rewriting (vLLM) | +100-300 | 3-10x faster with vLLM server |
| + Filters | -50-100 | Narrows search space |

### Milvus Performance
- **Collection size**: ~5-8 GB for 300k papers (vectors + metadata)
- **Index build**: ~2-5 min for 300k docs
- **Search latency**: ~50-100ms for hybrid search without reranking
- **Scalability**: Tested up to 300k docs, supports 10M+ with HNSW index

## Advanced Usage

### Query Rewriting with LLM Analysis

```powershell
# Enable query rewriting with full LLM analysis
python scripts/query.py --rewrite-query

# Example: "original unet paper"
# → Intent: foundational
# → Extracts: target_title="U-Net: Convolutional Networks...", target_authors=["Ronneberger", "Fischer", "Brox"]
# → Rewrites: "seminal U-Net segmentation architecture"
# → Searches: original + rewrite + title + authors (all fused with RRF)
# → Applies: foundational boosting + title/author matching + Jina reranking
```

### Performance Optimization: vLLM

For faster query rewriting, use vLLM server:

```powershell
# Install vLLM
pip install vllm

# Start vLLM server (one-time, runs in background)
vllm serve Qwen/Qwen3-4B-AWQ `
  --port 8000 `
  --dtype auto `
  --max-model-len 2048 `
  --gpu-memory-utilization 0.6
```

```yaml
# In config.yaml
query_rewriting:
  use_vllm: true
  vllm_endpoint: http://localhost:8000/v1
```

### Reranker Selection

```yaml
# In config.yaml

# Option 1: Jina (default, listwise)
reranker:
  type: jina
  model: jinaai/jina-reranker-v3
  batch_size: 50  # MUST match rerank_top_k for listwise comparison!

# Option 2: Cross-Encoder (fallback, pairwise)
reranker:
  type: cross-encoder
  model: cross-encoder/ms-marco-MiniLM-L-12-v2
  batch_size: 32
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
1. **Enable vLLM** for query rewriting (3-10x speedup)
2. Reduce `rerank_top_k` from 50 to 30
3. Disable reranking: `--no-rerank`
4. Disable query rewriting if not needed
5. For >1M docs: Switch to HNSW index in `config.yaml`

### Missing Dependencies

```powershell
# Reinstall environment
conda env remove -n arxplorer-env
conda env create -f environment.yml -n arxplorer-env
conda activate arxplorer-env

# For vLLM support (optional, for fast query rewriting)
pip install vllm
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

### Local Docker Backup

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

### AWS Production Backup

For AWS deployments:
- **Automated daily backups** to S3 (scheduled at 2 AM UTC)
- **EBS snapshots** for disaster recovery
- **30-day retention** (configurable)

See [AWS Deployment Guide](infrastructure/AWS_DEPLOYMENT.md) for backup management.

## Production Deployment

### AWS Infrastructure

Deploy production-ready infrastructure on AWS:

```bash
cd infrastructure/terraform

# Configure
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your AWS key pair and IP

# Deploy
terraform init
terraform apply

# Get connection info
terraform output
```

This deploys:
- **vLLM Instance** (g5.xlarge): GPU inference server for Qwen3-4B-AWQ
- **Milvus Instance** (c5.2xlarge): Vector database with 500GB EBS
- **S3 Backups**: Automated daily backups with 30-day retention
- **Security Groups**: Restricted access to API ports
- **IAM Roles**: Least-privilege access for services

**Cost Estimates**:
- 24/7 operation: ~$1,035/month
- With spot instances: ~$350-400/month
- 12hr/day usage: ~$500/month

See [AWS_DEPLOYMENT.md](infrastructure/AWS_DEPLOYMENT.md) for:
- Detailed setup instructions
- Cost optimization strategies
- Monitoring and alerting
- Scaling guidelines
- Security best practices

### Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│                  AWS VPC                         │
│                                                  │
│  ┌──────────────────┐    ┌─────────────────┐   │
│  │  vLLM Instance   │    │ Milvus Instance │   │
│  │  g5.xlarge       │◄───┤  c5.2xlarge     │   │
│  │  1x A10G GPU     │    │  8 vCPU         │   │
│  │  Port 8000       │    │  Port 19530     │   │
│  └──────────────────┘    └─────────────────┘   │
│         │                         │              │
│         │                  ┌──────▼──────┐       │
│         │                  │   EBS 500GB │       │
│         │                  └──────┬──────┘       │
│         │                         │              │
│         │                  ┌──────▼──────┐       │
│  Client │                  │  S3 Bucket  │       │
│  Queries│                  │  (Backups)  │       │
│         │                  └─────────────┘       │
└─────────┼─────────────────────────────────────-──┘
          │
    ┌─────▼──────┐
    │   Local    │
    │   Scripts  │
    └────────────┘
```

## References

- **Milvus**: [milvus.io](https://milvus.io/) - Open-source vector database
- **SPECTER**: [Cohan et al., 2020](https://arxiv.org/abs/2004.07180) - Document-level embeddings
- **SPECTER2**: [Singh et al., 2022](https://arxiv.org/abs/2209.07930) - Adapters for specialized embeddings
- **SPLADE**: [Formal et al., 2021](https://arxiv.org/abs/2107.05720) - Sparse lexical and expansion model
- **RRF**: Cormack et al., 2009 - Reciprocal Rank Fusion
- **Qwen 3**: [Qwen Team, 2024](https://qwenlm.github.io/) - Multilingual LLM family
- **Jina Reranker v3**: [Jina AI, 2024](https://huggingface.co/jinaai/jina-reranker-v3) - Listwise document reranker

## Recent Updates (November 2025)
