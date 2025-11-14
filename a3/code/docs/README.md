# ArXplorer Data Processing Pipeline

## Overview

ArXplorer is an academic search assistant for arXiv that processes research papers using state-of-the-art NLP techniques. This pipeline handles data ingestion, text processing, semantic embedding generation, and vector search indexing.

## Pipeline Architecture

```
Kaggle arXiv Dataset → Data Ingestion → Text Processing → Embedding Generation → Vector Index → Search Ready
```

**Core Components:**
- **Data Ingestion**: Load papers from Kaggle arXiv dataset (2M+ papers)
- **Text Processing**: Clean text, extract keywords, analyze readability  
- **Embedding Generation**: Create semantic vectors using SciBERT
- **Vector Indexing**: Build FAISS search index for similarity queries

## Data Flow Stages

### 1. Data Ingestion (`ArXivDataIngester`)
- **Input**: arXiv API queries with date ranges and categories
- **Process**: 
  - Fetches papers via arXiv REST API
  - Parses XML responses
  - Extracts metadata (title, abstract, authors, categories, dates)
  - Handles pagination and rate limiting
- **Output**: `ArXivPaper` objects with raw metadata
- **Technologies**: `requests`, `xml.etree.ElementTree`

### 2. Text Processing (`TextProcessor`)
- **Input**: Raw `ArXivPaper` objects
- **Process**:
  - Cleans and normalizes text (remove special chars, URLs)
  - Extracts keywords using TF-IDF
  - Calculates readability scores
  - Tokenizes and processes for downstream tasks
- **Output**: `ProcessedPaper` objects with cleaned text and features
- **Technologies**: `nltk`, `scikit-learn`, `regex`

### 3. Embedding Generation (`EmbeddingGenerator`)
- **Input**: `ProcessedPaper` objects with cleaned text
- **Process**:
  - Loads pre-trained SciBERT model
  - Generates semantic embeddings for titles and abstracts
  - Creates combined embeddings for search
  - Handles batching and GPU acceleration
- **Output**: `PaperEmbedding` objects with vector representations
- **Technologies**: `transformers`, `torch`, `SciBERT`

### 4. Vector Indexing (`VectorIndexer`)
- **Input**: `PaperEmbedding` objects
- **Process**:
  - Builds FAISS vector index for similarity search
  - Supports IVF (Inverted File) clustering for large datasets
  - Optimizes for cosine similarity search
  - Persists index to disk
- **Output**: Searchable FAISS index
- **Technologies**: `faiss`, `numpy`

## Data Schemas

### Core Entities

1. **ArXivPaper**: Raw paper metadata from arXiv
2. **ProcessedPaper**: Cleaned text and extracted features
3. **PaperEmbedding**: Semantic vector representations
4. **SearchQuery**: User queries and their embeddings
5. **SearchResult**: Ranked results with relevance scores

### Database Design (Data Warehouse)

```sql
-- Papers table (fact table)
papers(arxiv_id, title, abstract, authors, categories, dates, status)

-- Processed papers (derived facts)
processed_papers(arxiv_id, cleaned_text, keywords, metrics)

-- Embeddings (vector storage)
embeddings(arxiv_id, title_vector, abstract_vector, combined_vector)

-- Search queries (analytics)
search_queries(query_id, text, embeddings, filters, timestamp)
```

## Pipeline Execution

### Batch Processing Schedule

1. **Daily Updates** (02:00 UTC):
   - Fetch papers from last 24 hours
   - Process new submissions in AI/ML categories
   - Update search index incrementally
   - Max 500 papers per run

2. **Weekly Full Refresh** (Sunday 01:00 UTC):
   - Full reprocessing of last week's papers
   - Rebuild vector index from scratch
   - Update citation counts and metadata
   - Max 2000 papers per run

### Use Cases

1. **Real-time Search**: Query processing and similarity search
2. **Data Analytics**: Track research trends and topics
3. **Recommendation System**: Suggest related papers
4. **Citation Analysis**: Build academic knowledge graphs

## Technologies Used

### Open Source Frameworks

- **Data Processing**: Pandas, NumPy, SciPy
- **NLP**: NLTK, spaCy, transformers
- **Machine Learning**: scikit-learn, PyTorch
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Database**: SQLAlchemy, SQLite/PostgreSQL
- **Web Framework**: FastAPI (future API)
- **Task Queue**: Celery + Redis (async processing)

### Infrastructure

- **Storage**: File system (JSON) + SQL database
- **Compute**: CPU/GPU for embedding generation
- **Monitoring**: Prometheus + custom metrics
- **Deployment**: Docker containers (future)

## Implementation Status

### Completed Features

1. Core data schemas and models
2. arXiv API integration and parsing
3. Text processing and cleaning pipeline
4. SciBERT embedding generation
5. FAISS vector indexing
6. Configuration management
7. Logging and error handling
8. Batch processing framework

### Next Steps (Not Yet Implemented)

1. **Citation Analysis**:
   - Parse references from full-text PDFs
   - Build citation networks
   - Calculate impact metrics

2. **Full-Text Processing**:
   - Download and parse PDF content
   - Extract figures and equations
   - Section-based embeddings

3. **Author Disambiguation**:
   - Resolve author identities
   - Build author profiles
   - Collaboration networks

4. **Real-time API**:
   - FastAPI web service
   - Query processing endpoint
   - Result ranking and filtering

5. **Web Interface**:
   - React-based search UI
   - Interactive result exploration
   - Advanced filtering options

6. **Advanced Search Features**:
   - Semantic clustering
   - Topic modeling
   - Trend analysis

7. **Performance Optimization**:
   - Distributed processing
   - Caching layer
   - Index compression

8. **Data Quality**:
   - Duplicate detection
   - Content validation
   - Metadata enrichment

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Analyze Dataset
```bash
python static_pipeline.py --dataset-path /path/to/kaggle/arxiv --analyze-only
```

#### 2. Process Papers (Fast)
```bash
python static_pipeline.py \
  --dataset-path /path/to/kaggle/arxiv \
  --categories cs.AI cs.CL cs.LG \
  --max-papers 500 \
  --skip-embeddings
```

#### 3. Full Processing with Embeddings
```bash
python static_pipeline.py \
  --dataset-path /path/to/kaggle/arxiv \
  --categories cs.AI cs.CL \
  --max-papers 100
```

### Configuration

Edit `config.yaml` to customize:
- Processing batch sizes and worker counts
- Embedding model selection (SciBERT)
- Storage paths and output formats
- Vector search parameters

## Performance

- **Text Processing**: ~50-100 papers/second
- **Embedding Generation**: ~10-20 papers/second (SciBERT)
- **Index Building**: ~1-5 minutes for 10K papers
- **Memory Usage**: <2GB for 1K papers

## Output

The pipeline generates:
- **Processed Papers**: Cleaned text with extracted keywords
- **Semantic Embeddings**: 768-dimensional vectors (SciBERT)
- **Search Index**: FAISS vector similarity index
- **Analytics**: Dataset statistics and processing reports