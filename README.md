# ArXplorer - Academic Paper Retrieval System

![Coverage](.github/badges/coverage.svg)
![Tests](https://github.com/UofT-CSC490-F2025/turtleneck/actions/workflows/test.yml/badge.svg?branch=hybrid-pipeline)

ArXplorer is a production-ready academic paper retrieval system combining **Milvus vector database** with **multi-stage hybrid search** (dense + sparse vectors), **LLM-based query analysis**, **intent-aware boosting**, and **advanced reranking**.

## Table of Contents

- [Quick Start (Local)](#quick-start-local)
- [AWS Deployment](#aws-deployment)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Running Evaluations](#running-evaluations)
- [Configuration](#configuration)
- [Architecture](#architecture)

---

## Quick Start (Local)

### Prerequisites
- **Docker Desktop** ([download](https://www.docker.com/products/docker-desktop/))
- **Python 3.10+** with conda
- **8GB+ RAM**, **6GB+ GPU VRAM** (for encoding)

### 1. Start Milvus Vector Database

```bash
# Start Milvus standalone with Docker Compose
docker-compose -f docker-compose.milvus.yml up -d

# Verify Milvus is running (wait ~30 seconds for startup)
curl http://localhost:9091/healthz
# Should return: OK

# Check container status
docker-compose -f docker-compose.milvus.yml ps
```

**Expected output:**
```
NAME              IMAGE                     PORTS
milvus-milvus     milvusdb/milvus:v2.4.15   0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp
milvus-etcd       quay.io/coreos/etcd       2379/tcp, 2380/tcp
milvus-minio      minio/minio               9000/tcp, 9001/tcp
```

### 2. Setup Python Environment

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate arxplorer-env
```

### 3. Encode Documents into Milvus

You have two options for data:

#### Option A: Use Demo Dataset (Fast - 1k papers)

```bash
# Encode the included 1k demo dataset (~2-5 minutes)
python scripts/encode.py --data-file data/arxiv_1k.jsonl
```

#### Option B: Create Full Dataset from Kaggle (300k+ papers)

```bash
# 1. Download arXiv dataset from Kaggle
# Visit: https://www.kaggle.com/datasets/Cornell-University/arxiv
# Download arxiv-metadata-oai-snapshot.json and place in data/kaggle_arxiv/

# 2. Extract papers to JSONL format
python scripts/create_arxiv_dataset.py --limit 300000

# 3. (Optional) Fetch citation counts from OpenAlex API
# ⚠️ WARNING: This takes 10+ hours for 300k papers
python scripts/fetch_citations_openalex.py

# 4. Encode full dataset (~90-180 minutes with GPU)
python scripts/encode.py --data-file data/arxiv_300k.jsonl
```

**What encoding does:**
- Generates **dense vectors** (SPECTER2, 768-dim) and **sparse vectors** (SPLADE, ~30k-dim)
- Creates Milvus collection with hybrid indexes
- Stores metadata (title, authors, year, categories, citations)

### 4. Query the System

```bash
# Interactive mode (default)
python scripts/query.py

# Single query with full pipeline
python scripts/query.py --query "attention is all you need" --rewrite-query

# Fast queries (no reranking)
python scripts/query.py --no-rerank

# Override number of results
python scripts/query.py --top-k 20
```

**Example query:**
```
Query: original unet paper

Intent: specific_paper
Extracted Title: U-Net
Extracted Authors: Ronneberger, Fischer, Brox

Results:
1. U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)
   Authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
   Score: 0.95 | Citations: 45,234
   
2. SegNet: A Deep Convolutional Encoder-Decoder Architecture... (2015)
   Score: 0.73 | Citations: 12,891
```

---

## AWS Deployment

Deploy production infrastructure with **GPU inference (vLLM)**, **scalable Milvus**, and **FastAPI query endpoint**.

### Prerequisites

1. **AWS Account** with programmatic access
2. **AWS CLI** configured: `aws configure`
3. **Terraform** installed ([download](https://www.terraform.io/downloads))
4. **SSH key pair** for EC2 instances

### Step 1: Configure AWS Credentials

```bash
# Configure AWS CLI with your credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (ca-central-1), Output format (json)
```

### Step 2: Setup Terraform Variables

```bash
cd terraform

# Copy example variables
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your settings
nano terraform.tfvars
```

**Required changes in `terraform.tfvars`:**

```hcl
# 1. Generate SSH key pair
key_name = "arxplorer-key"  # Will be created at ~/.ssh/arxplorer-key.pem

# 2. Set your IP for SSH access (find with: curl ifconfig.me)
allowed_ip = "123.456.789.0/32"  # Replace with YOUR_IP/32

# 3. Get HuggingFace token for model downloads
# Visit: https://huggingface.co/settings/tokens
# Create read token and paste below
huggingface_token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 4. Enable/configure components
enable_vllm = false                   # Note: this setting is a legacy version that used GPU ec2 instance
enable_query_api = true               # FastAPI endpoint
milvus_instance_type = "c5.2xlarge"   # 8 vCPU, 16GB RAM
```

### Step 3: Generate SSH Key

```bash
# Generate SSH key pair (if not exists)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/arxplorer-key -N ""
chmod 400 ~/.ssh/arxplorer-key
```

### Step 4: Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Deploy (takes ~5-10 minutes)
terraform apply
# Type 'yes' to confirm

# Save outputs for later use
terraform output > ../aws-endpoints.txt
```

**What gets deployed:**
- **Milvus Instance** (c5.2xlarge): Vector database with 500GB EBS storage
- **vLLM Instance** (g5.xlarge, optional): GPU server for Qwen3-4B-AWQ inference
- **Query API Instance** (t3.xlarge, optional): FastAPI server with full query pipeline
- **S3 Bucket**: Automated backups with 30-day retention
- **Security Groups**: Restricted access to your IP

### Step 5: Wait for Initialization (~10 minutes)

After `terraform apply`, instances need time to:
- Download and install dependencies
- Pull Docker images
- Download ML models from HuggingFace
- Start services

```bash
# Check Milvus instance readiness
MILVUS_IP=$(terraform output -raw milvus_public_ip)
ssh -i ~/.ssh/arxplorer-key.pem ubuntu@$MILVUS_IP 'docker ps'
# Should show: milvus-standalone, etcd, minio containers running

# Check API instance readiness (if enabled)
API_IP=$(terraform output -raw query_api_public_ip)
ssh -i ~/.ssh/arxplorer-key.pem ubuntu@$API_IP 'sudo systemctl status arxplorer-api'
# Should show: active (running)
```

### Step 6: Load Data into Milvus

You have **two options**:

#### Option A: Restore from S3 Backup (2-5 minutes)

If you have a previous Milvus backup on S3:

```bash
# List available backups
aws s3 ls s3://arxplorer-backups-prod/volumes/

# Restore backup
MILVUS_IP=$(terraform output -raw milvus_public_ip) \
  ./scripts/backup_restore_aws.sh restore milvus-volumes-backup-YYYYMMDD-HHMMSS.tar.gz
```

#### Option B: Encode Documents from Scratch (90-180 minutes)

Encode documents locally, then upload to AWS Milvus:

```bash
# 1. Update config.yaml with AWS Milvus IP
MILVUS_IP=$(cd terraform && terraform output -raw milvus_public_ip)
echo "Milvus IP: $MILVUS_IP"

# Edit config.yaml:
nano config.yaml
# Change: milvus.host: "localhost" → milvus.host: "<MILVUS_IP>"

# 2. Encode documents (uses your local GPU)
python scripts/encode.py --data-file data/arxiv_300k.jsonl

# Note: This uploads vectors to AWS Milvus over the network
# Encoding: ~90-180 min with GPU
# Upload: ~10-20 min for 300k docs
```

### Step 7: Deploy Query API

```bash
# Deploy code to API instance
./scripts/deploy_query_api.sh

# Wait for deployment (~2-3 minutes)
# This uploads code, installs dependencies, and restarts service
```

### Step 8: Test API

```bash
# Get API endpoint
cd terraform
API_ENDPOINT=$(terraform output -raw query_api_endpoint)

# Health check
curl $API_ENDPOINT/health

# Test query
curl -X POST "$API_ENDPOINT/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "attention is all you need",
    "top_k": 10,
    "use_reranking": true
  }'
```

### Step 9: Run Cloud Frontend (Optional)

```bash
# Start local frontend connected to cloud API
./scripts/run_cloud_frontend.sh
# Opens browser at http://localhost:5001
```

### AWS Management Commands

```bash
# View Milvus logs
ssh -i ~/.ssh/arxplorer-key.pem ubuntu@$MILVUS_IP 'docker logs -f milvus-standalone'

# View API logs
ssh -i ~/.ssh/arxplorer-key.pem ubuntu@$API_IP 'sudo journalctl -u arxplorer-api -f'

# Backup Milvus data to S3
MILVUS_IP=$MILVUS_IP ./scripts/backup_restore_aws.sh backup

# Stop instances (data persists on EBS)
aws ec2 stop-instances --instance-ids $(cd terraform && terraform output -raw milvus_instance_id)

# Start instances
aws ec2 start-instances --instance-ids $(cd terraform && terraform output -raw milvus_instance_id)

# Destroy infrastructure (DELETES ALL DATA)
cd terraform
terraform destroy
```

---

## Project Structure

```
turtleneck/
├── src/                          # Core library code
│   ├── config/                   # YAML configuration management
│   │   ├── settings.py           # Config dataclasses
│   │   └── __init__.py
│   ├── data/                     # Data models and loaders
│   │   ├── document.py           # Document dataclass
│   │   ├── loader.py             # Streaming JSONL loader
│   │   └── __init__.py
│   └── retrieval/                # Retrieval pipeline components
│       ├── encoders/             # Dense (SPECTER2) and Sparse (SPLADE) encoders
│       │   ├── dense.py          # SPECTER2 with adapter switching
│       │   ├── sparse.py         # SPLADE encoder
│       │   ├── base.py           # Base encoder interface
│       │   └── __init__.py
│       ├── indexers/             # Milvus indexing
│       │   ├── milvus_indexer.py # Unified hybrid indexer
│       │   ├── base.py           # Base indexer interface
│       │   └── __init__.py
│       ├── searchers/            # Milvus search
│       │   ├── milvus_hybrid_searcher.py  # Hybrid search with RRF fusion
│       │   ├── base.py           # Base searcher interface
│       │   └── __init__.py
│       ├── rerankers/            # Score refinement
│       │   ├── jina_reranker.py          # Jina AI listwise reranker (default)
│       │   ├── cross_encoder_reranker.py # CrossEncoder pairwise reranker
│       │   ├── intent_booster.py         # Intent-based score boosting
│       │   ├── title_author_matcher.py   # Fuzzy title/author matching
│       │   ├── base.py           # Base reranker interface
│       │   └── __init__.py
│       ├── query_rewriting/      # LLM query analysis
│       │   ├── llm_rewriter.py   # LLM-based intent detection + expansion
│       │   ├── base.py           # Base rewriter interface
│       │   └── __init__.py
│       └── __init__.py
├── scripts/                      # Executable scripts
│   ├── encode.py                 # Build Milvus index from JSONL
│   ├── query.py                  # Interactive/single query search
│   ├── create_arxiv_dataset.py   # Extract papers from Kaggle dataset
│   ├── fetch_citations_openalex.py  # Fetch citation counts (10+ hours)
│   ├── run_tests.py              # Run test suite with coverage
│   ├── deploy_query_api.sh       # Deploy code to AWS API instance
│   ├── backup_restore_aws.sh     # Backup/restore AWS Milvus
│   └── run_cloud_frontend.sh     # Start frontend for cloud API
├── tests/                        # Test suite (96% coverage)
│   ├── test_encoders.py          # Encoder unit tests
│   ├── test_searchers.py         # Searcher unit tests
│   ├── test_rerankers.py         # Reranker unit tests
│   ├── test_intent_booster.py    # Intent boosting tests
│   ├── test_title_author_matcher.py  # Title/author matching tests
│   ├── test_query_rewriting.py   # LLM query analysis tests
│   ├── test_indexers.py          # Indexer unit tests
│   ├── test_loader.py            # Data loader tests
│   └── conftest.py               # Pytest fixtures
├── evaluation/                   # Evaluation framework
│   ├── scripts/                  # Evaluation runner scripts
│   ├── data/                     # Benchmark datasets
│   ├── results/                  # Evaluation results
│   └── README.md                 # See evaluation/README.md for details
├── terraform/                    # AWS infrastructure as code
│   ├── main.tf                   # Main infrastructure definition
│   ├── outputs.tf                # Terraform outputs (IPs, endpoints)
│   ├── terraform.tfvars.example  # Example configuration
│   ├── user_data_milvus.sh       # Milvus instance init script
│   ├── user_data_vllm.sh         # vLLM instance init script
│   └── user_data_query_api.sh    # Query API instance init script
├── data/                         # Datasets
│   ├── arxiv_1k.jsonl            # Demo dataset (1k papers)
│   ├── arxiv_300k.jsonl          # Full dataset (generated)
│   ├── citations.json            # Citation counts (optional)
│   └── kaggle_arxiv/             # Downloaded Kaggle dataset
├── config.yaml                   # Main configuration file
├── config.api.yaml               # API-specific configuration
├── environment.yml               # Conda environment definition
├── docker-compose.milvus.yml     # Local Milvus Docker setup
├── pyproject.toml                # Python package configuration
├── README.md                     # This file
├── COVERAGE_REPORT.md            # Test coverage report
└── .github/
    ├── workflows/
    │   └── test.yml              # CI/CD: tests + coverage badge
    └── badges/
        └── coverage.svg          # Auto-generated coverage badge
```

### Key Directories Explained

- **`src/`**: Core library containing all retrieval pipeline components. Fully tested (96% coverage).
- **`scripts/`**: Entry points for indexing, querying, deployment, and management.
- **`tests/`**: Comprehensive test suite with 163 tests. Run with `pytest tests/`.
- **`evaluation/`**: Benchmark framework for evaluating retrieval quality. See `evaluation/README.md`.
- **`terraform/`**: Infrastructure as Code for AWS deployment. Creates GPU/CPU instances, S3 backups.
- **`data/`**: Local storage for datasets. `.gitignore`'d except demo files.

---

## Running Tests

ArXplorer has **96% test coverage** with **163 passing tests**.

```bash
# Run all tests with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Or use the test runner script
python scripts/run_tests.py

# Run specific test file
pytest tests/test_searchers.py -v

# Run with verbose output
pytest tests/ -vv

# Run fast tests only (skip slow integration tests)
pytest tests/ -m "not slow"
```

**Coverage breakdown:**
- Overall: **96.09%**
- Dense/Sparse encoders: **100%**
- Milvus indexer: **99.28%**
- Milvus searcher: **99.11%**
- Intent booster: **98.61%**
- Title/author matcher: **97.48%**
- LLM rewriter: **81.43%** (remaining 18% is optional vLLM/Bedrock cloud APIs)

---

## Running Evaluations

Evaluate retrieval quality against benchmark datasets:

```bash
cd evaluation

# See evaluation/README.md for:
# - Available benchmarks (TREC-COVID, NFCorpus, etc.)
# - How to run evaluations
# - Interpreting results (NDCG@10, Recall@100, MRR)
# - Comparing configurations
```

The evaluation framework supports:
- **Automated benchmarking** against standard IR datasets
- **Comparison** between different configurations (e.g., with/without reranking)
- **Metrics**: NDCG@k, Recall@k, MRR, Precision@k
- **Baseline comparisons**: BM25, dense-only, sparse-only

---

## Configuration

ArXplorer is configured via `config.yaml`. All settings can be overridden with CLI arguments.

### Key Configuration Sections

```yaml
# Encoders
encoder:
  use_specter2: true                           # Use SPECTER2 with adapters
  specter2_base_adapter: "allenai/specter2"    # For documents
  specter2_query_adapter: "allenai/specter2_adhoc_query"  # For queries
  
# Milvus connection
milvus:
  host: "localhost"                            # Change to AWS IP for cloud
  port: 19530
  collection_name: "arxplorer_papers"
  
# Search parameters
search:
  top_k: 10                                    # Final results
  retrieval_k: 200                             # Candidates for reranking
  rrf_k: 60                                    # RRF fusion constant
  
# Reranker
reranker:
  enabled: true
  type: "jina"                                 # Options: jina, cross-encoder
  rerank_top_k: 50                             # How many to rerank
  batch_size: 50                               # MUST match rerank_top_k for Jina
  
# LLM query analysis
query_rewriting:
  enabled: true
  model: "Qwen/Qwen3-4B-AWQ"
  use_vllm: false                              # Set true for 3-10x speedup
  vllm_endpoint: "http://localhost:8000/v1"
```

### CLI Overrides

```bash
# Override top-k
python scripts/query.py --top-k 20

# Override Milvus host
python scripts/query.py --milvus-host 3.96.123.45

# Disable reranking
python scripts/query.py --no-rerank

# Enable query rewriting
python scripts/query.py --rewrite-query
```

---

## Architecture

### Multi-Stage Retrieval Pipeline

```
User Query: "original unet paper"
    │
    ├──► 1. LLM Query Analysis (Qwen3-4B-AWQ)
    │       • Intent: specific_paper
    │       • Extracted: title="U-Net", authors=["Ronneberger"]
    │       • Rewrites: "seminal U-Net segmentation architecture"
    │
    ├──► 2. Multi-Query Hybrid Search (Milvus)
    │       • Original query → Dense + Sparse vectors
    │       • Rewrite query → Dense + Sparse vectors
    │       • Extracted title → Dense + Sparse vectors
    │       • RRF Fusion across all queries
    │       • Retrieves top 200 candidates
    │
    ├──► 3. Intent-Based Boosting
    │       • specific_paper: boost citations (0.15), no date bias
    │       • Normalize scores to [0, 1]
    │
    ├──► 4. Title/Author Fuzzy Matching
    │       • Jaccard similarity on title tokens (threshold: 0.5)
    │       • Author overlap (threshold: 0.7)
    │       • Boost matching papers
    │
    ├──► 5. Jina Reranking (Listwise)
    │       • Listwise comparison of top 50 candidates
    │       • Captures cross-document relevance
    │
    ├──► 6. Score Fusion
    │       • 0.7 × boosted_score + 0.3 × rerank_score
    │
    └──► Results:
            1. U-Net: Convolutional Networks... (0.95)
            2. SegNet: A Deep Convolutional... (0.73)
```

### Milvus Unified Storage

All data stored in a single Milvus collection:

| Field | Type | Description |
|-------|------|-------------|
| `id` | VARCHAR | ArXiv ID (e.g., "arxiv:1706.03762") |
| `title` | VARCHAR | Paper title |
| `abstract` | VARCHAR | Paper abstract (truncated to 8192 chars) |
| `authors` | ARRAY | List of author names |
| `categories` | ARRAY | ArXiv categories (e.g., ["cs.AI", "cs.LG"]) |
| `year` | INT64 | Publication year |
| `citation_count` | INT64 | Citation count (from OpenAlex) |
| `dense_vector` | FLOAT_VECTOR | SPECTER2 embedding (768-dim) |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | SPLADE embedding (~30k-dim, sparse) |

**Indexes:**
- Dense: `IVF_FLAT` (nlist=1024) for <1M docs, `HNSW` for >1M docs
- Sparse: `SPARSE_INVERTED_INDEX` (memory-efficient)

---

## Key Features

### 1. Unified Milvus Architecture
- **Single collection** stores dense + sparse vectors + metadata
- **Built-in hybrid search** with RRF (Reciprocal Rank Fusion)
- **Scalar filtering** integrated with vector search (year, citations)
- **Production-ready**: Distributed architecture, backups, high availability

### 2. LLM Query Analysis
- **Model**: Qwen3-4B-AWQ (quantized for speed)
- **Extraction**: Intent, title, authors, year/citation filters, query rewrites
- **6 Intent types**: topical, sota, foundational, comparison, method_lookup, specific_paper
- **Multi-query search**: Uses original + rewrites + extracted title/authors

### 3. Intent-Aware Boosting
- **Citation weighting**: foundational (0.3) > specific_paper (0.15) > topical (0.1)
- **Date weighting**: SOTA favors recent (+0.1), foundational favors older (+0.1)
- **Normalized scoring**: All boosts normalized to [0, 1] before fusion

### 4. Advanced Reranking
- **Jina Reranker** (default): Listwise ranking, sees all candidates simultaneously
- **CrossEncoder** (fallback): Pairwise ranking, faster but less context-aware
- **Weighted fusion**: 0.7 × boosted_score + 0.3 × rerank_score

### 5. Metadata-Enhanced Embeddings
Documents encoded with full context:
```
Title: U-Net: Convolutional Networks for Biomedical Image Segmentation
Authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
Year: 2015
Categories: cs.CV
Abstract: [full text...]
```

This enables exact title matching, author queries, and temporal context for rerankers.

---

## References

- **Milvus**: [milvus.io](https://milvus.io/) - Open-source vector database
- **SPECTER2**: [Singh et al., 2022](https://arxiv.org/abs/2209.07930) - Document embeddings with adapters
- **SPLADE**: [Formal et al., 2021](https://arxiv.org/abs/2107.05720) - Sparse lexical expansion
- **RRF**: Cormack et al., 2009 - Reciprocal Rank Fusion
- **Qwen3**: [Qwen Team, 2024](https://qwenlm.github.io/) - Multilingual LLM family
- **Jina Reranker v3**: [Jina AI, 2024](https://huggingface.co/jinaai/jina-reranker-v3) - Listwise reranker

---
