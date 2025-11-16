# Retrieval System - Dense, Sparse, and Hybrid Search with Query Rewriting

A modular retrieval system supporting dense (SPECTER), sparse (SPLADE), and hybrid search with Reciprocal Rank Fusion. Includes LLM-based query rewriting with Qwen for improved retrieval robustness.

## Quick Start

### 1. Setup Environment

```powershell
# Create conda environment from provided environment.yml
conda env create -f environment.yml -n arxplorer-env
conda activate arxplorer-env
```

### 2. Prepare Dataset

Download the Kaggle arXiv dataset and extract papers to JSONL format:

```powershell
# 1. Download the arXiv dataset from Kaggle
# Visit: https://www.kaggle.com/datasets/Cornell-University/arxiv
# Download and extract to: data/kaggle_arxiv/

# 2. Create filtered JSONL dataset (default: AI/ML categories)
python scripts/create_arxiv_dataset.py --limit 300000

# Or with custom categories
python scripts/create_arxiv_dataset.py --categories cs.ai,cs.lg,stat.ml --limit 100000

# Or include all categories
python scripts/create_arxiv_dataset.py --no-filter --limit 500000
```

This creates `data/arxiv_300k.jsonl` with standardized fields (id, title, abstract, categories, authors, etc.).

### 3. Build Indexes

```powershell
# Build both dense and sparse indexes (required for hybrid search)
python scripts/encode.py --method both --data-file data/arxiv_300k.jsonl

# Or build individually
python scripts/encode.py --method dense --data-file data/arxiv_1k.jsonl
python scripts/encode.py --method sparse --data-file data/arxiv_1k.jsonl
```

### 4. Query the Indexes

```powershell
# Hybrid search with reranking (default: enabled)
python scripts/query.py --method hybrid

# Hybrid search with query rewriting (generates multiple query variants)
python scripts/query.py --method hybrid --rewrite-query

# Hybrid search without reranking
python scripts/query.py --method hybrid --no-rerank

# Dense or sparse search with reranking
python scripts/query.py --method dense --rerank
python scripts/query.py --method sparse --rerank

# Single query (non-interactive)
python scripts/query.py --method hybrid --query "neural networks for NLP"

# Override reranking parameters
python scripts/query.py --method hybrid --rerank-top-k 50 --top-k 5
```

## Architecture

```
src/
├── config/          # YAML configuration management
├── data/            # Document dataclass and streaming JSONL loader
└── retrieval/
    ├── encoders/    # Dense (SPECTER) and Sparse (SPLADE) encoders
    ├── indexers/    # FAISS and scipy sparse indexers
    ├── searchers/   # Dense, sparse, and hybrid (RRF) searchers
    ├── rerankers/   # Cross-encoder reranking for result refinement
    └── query_rewriting/  # LLM-based query rewriting (Qwen)

scripts/
├── encode.py        # Unified encoding CLI
├── query.py         # Unified query CLI with multi-query fusion
└── create_arxiv_dataset.py  # Extract papers from Kaggle dataset

data/
├── dense_index/     # FAISS index + embeddings + doc_map
└── sparse_index/    # Scipy sparse matrix + doc_map
```

## Configuration

Edit `config.yaml` to customize:

```yaml
encoder:
  dense_model: "sentence-transformers/allenai-specter"
  sparse_model: "naver/splade-v3"
  
index:
  batch_size: 16
  dense_output_dir: "data/dense_index"
  sparse_output_dir: "data/sparse_index"
  
search:
  top_k: 10          # Final results to return
  retrieval_k: 100   # Retrieve from each system before fusion
  rrf_k: 60          # RRF constant (default from literature)

reranker:
  enabled: true      # Enable cross-encoder reranking
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  rerank_top_k: 100  # Number of candidates to rerank
  batch_size: 32

query_rewriting:
  enabled: false     # Enable LLM-based query rewriting (requires --rewrite-query flag)
  model: "Qwen/Qwen2.5-3B-Instruct"  # Instruction-tuned causal LM
  num_rewrites: 2    # Number of query variants to generate
  device: "cuda"     # Use GPU for fast inference
```

CLI arguments override config values:

```powershell
python scripts/encode.py --method both --batch-size 32 --device cuda
python scripts/query.py --method hybrid --top-k 20 --retrieval-k 200
```

## Key Features

### LLM-Based Query Rewriting
- **Model**: Qwen/Qwen3-4B-Instruct (instruction-tuned causal LM)
- **Multi-query generation**: Creates N diverse query variants (configurable via `num_rewrites`)
- **Multi-query fusion**: Combines results from original + rewritten queries using RRF
- **Robustness**: Improves recall by capturing alternative phrasings and terminology
- **Use case**: Especially effective for finding canonical papers with varied citation styles

### Three-Stage Retrieval Pipeline
1. **Query rewriting (optional)**: Generate diverse query variants with Qwen LLM
2. **First-stage retrieval**: Dense (SPECTER) + Sparse (SPLADE) with configurable retrieval_k (default 100)
3. **Fusion**: Reciprocal Rank Fusion (RRF) to combine rankings (multi-query if rewriting enabled)
4. **Reranking**: Cross-encoder (ms-marco-MiniLM) for precise relevance scoring on top candidates

### Hybrid Search with RRF
- Combines dense semantic search (SPECTER) with sparse lexical search (SPLADE)
- Uses Reciprocal Rank Fusion (RRF) - no score normalization needed
- Formula: `RRF(doc) = Σ 1/(k + rank_i(doc))` where k=60

### Cross-Encoder Reranking
- Refines top candidates with query-document relevance modeling
- Default: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, accurate)
- Optional: enable/disable via config or CLI flags
- Batch inference for efficiency

### Scalability
- **Streaming data loader**: Handles large JSONL files without loading all into memory
- **Batch processing**: Configurable batch sizes for encoding
- **Memory-efficient sparse storage**: CSR format for SPLADE vectors
- **FAISS optimization**: Fast similarity search for 300k+ documents
- **Checkpoint support**: Resume interrupted indexing (framework in place)

### Modular Design
- Abstract base classes for easy extension
- Pluggable encoders, indexers, and searchers
- YAML + CLI configuration
- Type-annotated for IDE support

## Performance Notes (300k corpus on RTX 4070 Super)

### Dense Encoding (SPECTER)
- ~1-2 min for 1k docs
- ~30-60 min for 300k docs
- GPU memory: ~4-6 GB
- Batch size 16 recommended

### Sparse Encoding (SPLADE)
- ~2-3 min for 1k docs  
- ~60-120 min for 300k docs
- GPU memory: ~6-8 GB
- Memory: ~2-4 GB RAM for sparse matrix construction

### Query Latency
- Dense: ~50-100ms
- Sparse: ~100-200ms
- Hybrid (RRF only): ~200-400ms
- Hybrid + Reranking (100 candidates): ~400-700ms
  - Cross-encoder adds ~200-300ms for batch scoring
- Hybrid + Query Rewriting (2 rewrites): ~600-900ms
  - Qwen inference adds ~200-400ms (FP16, GPU)
  - Multi-query retrieval adds ~100-200ms

## Advanced Usage

### Query Rewriting Options

```powershell
# Enable query rewriting with default settings (2 rewrites)
python scripts/query.py --method hybrid --rewrite-query

# Generate more query variants for better coverage
python scripts/query.py --method hybrid --rewrite-query --num-rewrites 4

# Query rewriting with reranking disabled (faster)
python scripts/query.py --method hybrid --rewrite-query --no-rerank

# Single-query mode with rewriting
python scripts/query.py --method hybrid --rewrite-query \
  --query "attention mechanism for transformers" \
  --top-k 10
```

### Custom Config File

```powershell
python scripts/encode.py --config my_custom_config.yaml --method both
```

### Override Data Keys

```powershell
python scripts/encode.py --method dense \
  --data-file data/custom.jsonl \
  --text-key "content" \
  --id-key "paper_id"
```

### Non-Interactive Query

```powershell
python scripts/query.py --method hybrid \
  --query "transformer attention mechanisms" \
  --top-k 20 \
  --rerank

# Disable reranking for faster queries
python scripts/query.py --method dense \
  --query "neural networks" \
  --no-rerank
```

## Incremental Indexing (Future)

The indexer framework supports incremental updates. To add new documents:

```python
# Load existing index
indexer = DenseIndexer(encoder, "data/dense_index")
indexer.load()

# Add new documents
new_docs = [Document(id="...", text="..."), ...]
indexer.add_documents(new_docs)

# Save updated index
indexer.save()
```

*Note: Full incremental CLI support coming soon.*

## Troubleshooting

### Import errors
Ensure you're running from the repo root and the conda environment is activated:
```powershell
conda activate a4-precision
python scripts/encode.py --method dense
```

### CUDA out of memory
Reduce batch size:
```powershell
python scripts/encode.py --method sparse --batch-size 8
```

### Missing index files
Run encoding first:
```powershell
python scripts/encode.py --method both
```

### Slow queries
- For dense: Use GPU FAISS (`use_gpu_faiss: true` in config)
- For sparse: Ensure index is in CSR format (automatic on save)
- For hybrid: Reduce `retrieval_k` (default 100)
- For reranking: Reduce `rerank_top_k` or disable with `--no-rerank`

## Data Format

Input JSONL format:
```json
{"id": "arxiv:1234", "title": "...", "abstract": "...", "authors": [...]}
{"id": "arxiv:5678", "title": "...", "abstract": "...", "authors": [...]}
```

Required fields (configurable via `config.yaml`):
- `id`: Document identifier
- `abstract` (or custom `text_key`): Text to encode

## References

- **SPECTER**: [Cohan et al., 2020](https://arxiv.org/abs/2004.07180)
- **SPLADE**: [Formal et al., 2021](https://arxiv.org/abs/2107.05720)
- **RRF**: Cormack et al., 2009 - "Reciprocal Rank Fusion"
- **MS MARCO Cross-Encoders**: [Bajaj et al., 2018](https://arxiv.org/abs/1611.09268)
- **Qwen 3** [Qwen Team, 2025](https://arxiv.org/abs/2505.09388) 
