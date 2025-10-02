# ArXplorer Pipeline - Setup Guide for Team Members

## Prerequisites

### 1. Python Version
- **Required**: Python 3.8 or higher
- **Recommended**: Python 3.10+ 
- **Current development version**: Python 3.12.10

Check your Python version:
```bash
python --version
```

### 2. Operating System
- **Windows**: ✅ Tested and working
- **macOS**: ✅ Should work (all dependencies support macOS)
- **Linux**: ✅ Should work (all dependencies support Linux)

## Installation Steps

### Step 1: Clone the Repository
```bash
git clone [YOUR_REPOSITORY_URL]
cd turtleneck
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv arxplorer_env

# Activate virtual environment
# On Windows:
arxplorer_env\Scripts\activate
# On macOS/Linux:
source arxplorer_env/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

**Note**: This will install ~2-3GB of packages including PyTorch and transformer models.

### Step 4: Download NLTK Data (Required)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

## Essential Dependencies

### Core Machine Learning
- **PyTorch** (`torch>=1.12.0`) - Deep learning framework
- **Transformers** (`transformers>=4.21.0`) - Hugging Face models (SciBERT)
- **scikit-learn** (`scikit-learn>=1.1.0`) - TF-IDF and PCA
- **NLTK** (`nltk>=3.7.0`) - Text processing

### Vector Search
- **FAISS** (`faiss-cpu>=1.7.2`) - Facebook's similarity search
  - Use `faiss-gpu` if you have CUDA GPU available

### Data Processing
- **pandas** (`pandas>=1.5.0`) - Data manipulation
- **numpy** (`numpy>=1.21.0`) - Numerical computing
- **requests** (`requests>=2.28.0`) - HTTP requests for arXiv API

### Utilities
- **PyYAML** (`pyyaml>=6.0`) - Configuration file parsing
- **tqdm** (`tqdm>=4.64.0`) - Progress bars

## Optional Dependencies

### For GPU Acceleration (Recommended)
If you have an NVIDIA GPU with CUDA:
```bash
# Uninstall CPU version
pip uninstall faiss-cpu

# Install GPU version
pip install faiss-gpu
```

### For Database Storage
Only needed if you want to use SQL databases instead of JSON files:
- **SQLAlchemy** - SQL toolkit
- **psycopg2-binary** - PostgreSQL adapter
- **pymysql** - MySQL adapter

## Quick Test

After installation, test if everything works:
```bash
python -c "
import torch
import transformers
import faiss
import pandas as pd
import nltk
from schemas import ArXivPaper
print('✅ All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## Data Requirements

### Kaggle arXiv Dataset
1. **Download** the arXiv dataset from Kaggle:
   - Go to https://www.kaggle.com/datasets/Cornell-University/arxiv
   - Download `arxiv-metadata-oai-snapshot.json` (~3.5GB)

2. **Extract** to a directory (e.g., `./data/kaggle_arxiv/`)

## Usage Examples

### Basic Usage (Fast - No Embeddings)
```bash
python static_pipeline.py \
  --dataset-path ./data/kaggle_arxiv \
  --categories cs.AI cs.CL cs.LG \
  --max-papers 100 \
  --skip-embeddings
```

### Full Pipeline (With Embeddings)
```bash
python static_pipeline.py \
  --dataset-path ./data/kaggle_arxiv \
  --categories cs.AI \
  --max-papers 50
```

## System Requirements

### Minimum Requirements
- **RAM**: 8GB (for text processing only)
- **Storage**: 5GB free space (for dependencies + data)
- **CPU**: Multi-core recommended for batch processing

### Recommended Requirements  
- **RAM**: 16GB+ (for embedding generation)
- **Storage**: 10GB+ free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for faster embeddings)

## Common Issues & Solutions

### 1. Import Errors
```bash
# If you get import errors, upgrade pip first:
pip install --upgrade pip setuptools wheel
```

### 2. NLTK Data Missing
```bash
# Download required NLTK data:
python -c "import nltk; nltk.download('all')"
```

### 3. Memory Issues
- Reduce `--max-papers` parameter
- Use `--skip-embeddings` flag
- Close other applications

### 4. GPU Not Detected
```bash
# Check CUDA installation:
python -c "import torch; print(torch.cuda.is_available())"
```

## Development Mode

For development/testing with smaller datasets:
```bash
# Quick test with minimal dependencies (no embeddings)
python static_pipeline.py \
  --dataset-path ./data/sample \
  --max-papers 10 \
  --skip-embeddings \
  --analyze-only
```

## Estimated Installation Time
- **Dependencies**: 5-10 minutes (depending on internet speed)
- **NLTK data**: 1-2 minutes
- **Dataset download**: 10-30 minutes (3.5GB file)
- **Total setup time**: 15-45 minutes

## Support

If team members encounter issues:
1. Check Python version compatibility
2. Ensure virtual environment is activated
3. Verify all dependencies installed correctly
4. Test with smaller datasets first
5. Check available RAM/storage space

The pipeline is designed to be robust and should work on most modern systems with Python 3.8+!