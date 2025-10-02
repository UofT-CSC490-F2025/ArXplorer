# ArXplorer Pipeline - Project Structure

## Essential Files

```
turtleneck/
├── schemas.py              # Data models and database schemas
├── pipeline.py             # Core processing components
├── kaggle_loader.py        # Kaggle dataset integration
├── static_pipeline.py      # Main pipeline orchestrator
├── config.yaml            # Configuration settings
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── CSC490_FINAL_SUMMARY.md # Assignment deliverable summary
```

## File Descriptions

### Core Pipeline Files

**`schemas.py`**
- Data models: `ArXivPaper`, `ProcessedPaper`, `PaperEmbedding`
- Database schemas for SQL storage
- Type-safe dataclasses with validation

**`pipeline.py`** 
- `TextProcessor`: Text cleaning and keyword extraction
- `EmbeddingGenerator`: SciBERT semantic embeddings
- `VectorIndexer`: FAISS search index creation
- `ArXivDataIngester`: arXiv API integration (alternative data source)

**`kaggle_loader.py`**
- `KaggleArXivLoader`: Load papers from Kaggle dataset
- JSON parsing and data validation
- Category and date filtering

**`static_pipeline.py`**
- `StaticArXplorerPipeline`: Main orchestrator
- End-to-end processing workflow
- CLI interface and batch processing

### Configuration

**`config.yaml`**
- Dataset paths and processing parameters
- Model configuration (SciBERT settings)
- Vector search and indexing options
- Storage and output settings

**`requirements.txt`**
- Core ML/NLP: PyTorch, transformers, scikit-learn
- Vector search: FAISS
- Data processing: pandas, numpy, nltk
- Utilities: requests, pyyaml, tqdm

## Usage Workflow

1. **Download** Kaggle arXiv dataset
2. **Configure** settings in `config.yaml`
3. **Run** `static_pipeline.py` with desired parameters
4. **Analyze** results in output directory

## Output Structure

```
output_directory/
├── processed/              # Cleaned papers with features
├── embeddings/            # Semantic vector representations
├── index/                 # FAISS search indices
└── stats/                 # Analysis reports and statistics
```

## Key Features

- ✅ **Static Dataset Processing**: No complex scheduling
- ✅ **Semantic Embeddings**: SciBERT for scientific text
- ✅ **Vector Search**: FAISS indexing for similarity queries
- ✅ **Batch Processing**: Memory-efficient large dataset handling
- ✅ **Error Handling**: Robust processing with comprehensive logging
- ✅ **Flexible Filtering**: By category, year, and paper count

Perfect for CSC490 project requirements with clean, production-ready code.