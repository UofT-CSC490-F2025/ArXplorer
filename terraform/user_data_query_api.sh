#!/bin/bash
# User data script for ArXplorer Query API EC2 instance

set -e

# Log all output
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "Starting ArXplorer Query API instance setup..."

# Update system
apt-get update
apt-get upgrade -y

# Install system dependencies
apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    ca-certificates

# Install AWS CLI
apt-get install -y awscli

# Create application directory
mkdir -p /opt/arxplorer
cd /opt/arxplorer

# Clone repository or copy code (for production, use git clone from private repo)
# For now, we'll create the structure and copy files via terraform

# Create Python virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install Python dependencies
cat > requirements.txt <<EOF
# Core dependencies
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3

# ML/NLP
torch==2.3.0
transformers==4.51.3
sentence-transformers==5.1.2
adapters
accelerate

# Vector search
pymilvus>=2.4.0

# AWS
boto3
botocore

# Utilities
numpy
scipy
python-multipart
EOF

pip install --upgrade pip
pip install -r requirements.txt

# Download models (pre-cache to avoid first-request delay)
echo "Pre-downloading models..."
python3 -c "
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# SPECTER2 base model
print('Downloading SPECTER2...')
AutoTokenizer.from_pretrained('allenai/specter2_base')
from adapters import AutoAdapterModel
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
model.load_adapter('allenai/specter2', source='hf')
model.load_adapter('allenai/specter2_adhoc_query', source='hf')

# SPLADE
print('Downloading SPLADE...')
AutoTokenizer.from_pretrained('naver/splade-v3-distilbert')
AutoModel.from_pretrained('naver/splade-v3-distilbert')

print('✓ Models downloaded')
"

# Create systemd service for API
# Using direct substitution (not heredoc with quotes) so terraform variables are expanded
echo "Creating systemd service file..."
cat > /etc/systemd/system/arxplorer-api.service <<SYSTEMD_EOF
[Unit]
Description=ArXplorer Query API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/arxplorer
Environment="PATH=/opt/arxplorer/venv/bin"
Environment="MILVUS_HOST=${MILVUS_HOST}"
Environment="MILVUS_PORT=${MILVUS_PORT}"
Environment="BEDROCK_REGION=${BEDROCK_REGION}"
Environment="BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID}"
ExecStart=/opt/arxplorer/venv/bin/python scripts/api_server.py --host 0.0.0.0 --port ${QUERY_API_PORT} --workers ${QUERY_API_WORKERS}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SYSTEMD_EOF

# Reload systemd
systemctl daemon-reload

echo "✓ Systemd service file created"
cat /etc/systemd/system/arxplorer-api.service

# Create config.yaml with correct Milvus host
echo "Creating config.yaml with Milvus host: ${MILVUS_HOST}..."
cat > /opt/arxplorer/config.yaml <<CONFIG_EOF
# ArXplorer Configuration for AWS Query API Deployment

encoder:
  dense_model: sentence-transformers/allenai-specter
  sparse_model: naver/splade-v3-distilbert
  max_length: 512
  normalize_dense: true
  device: null
  use_specter2: true
  specter2_base_adapter: allenai/specter2
  specter2_query_adapter: allenai/specter2_adhoc_query

index:
  batch_size: 16
  chunk_size: 10000
  sparse_encoder_batch_size: 4

search:
  top_k: 10
  rrf_k: 60
  retrieval_k: 200

intent_boosting:
  enabled: true
  citation_weights:
    topical: 0.1
    comparison: 0.1
    method_lookup: 0.1
    default: 0.1
    sota: 0.2
    foundational: 0.3
    specific_paper: 0.15
  date_weights:
    topical: 0.0
    comparison: 0.0
    method_lookup: 0.0
    default: 0.0
    sota: 0.1
    foundational: 0.1
    specific_paper: 0.0
  min_year: 1990

title_author_matching:
  enabled: true
  title_threshold: 0.5
  author_threshold: 0.7
  title_boost_weight: 1.0
  author_boost_weight: 1.0

citation:
  enabled: false
  data_file: data/citations.json
  score_formula: log
  normalize: true
  missing_default: 0.0

query_rewriting:
  enabled: true
  model: Qwen/Qwen3-4B-AWQ
  max_length: 128
  temperature: 0.3
  num_rewrites: 1
  device: null
  use_vllm: false
  vllm_endpoint: http://localhost:8000/v1
  vllm_timeout: 30
  use_bedrock: true
  bedrock_model_id: ${BEDROCK_MODEL_ID}
  bedrock_region: ${BEDROCK_REGION}
  bedrock_max_tokens: 512

reranker:
  enabled: false
  type: jina
  model: jinaai/jina-reranker-v3
  rerank_top_k: 50
  max_length: 512
  batch_size: 50
  pre_rerank_weight: 0.7
  rerank_weight: 0.3
  instruction: null

data:
  jsonl_file: data/arxiv_1k.jsonl
  text_key: abstract
  id_key: id
  title_key: title
  use_metadata: true
  categories_key: categories
  authors_key: authors
  year_key: published_date
  metadata_template: 'Title: \{title\}\n\nAuthors: \{authors\}\n\nYear: \{year\}\n\nCategories: \{categories\}\n\nAbstract: \{abstract\}'

milvus:
  host: ${MILVUS_HOST}
  port: 19530
  collection_name: arxplorer_papers
  dense_index_type: IVF_FLAT
  dense_nlist: 1024
  dense_nprobe: 64
  sparse_index_type: SPARSE_INVERTED_INDEX
  connection_timeout: 30
  batch_size: 1000
CONFIG_EOF

echo "✓ config.yaml created with Milvus host: ${MILVUS_HOST}"
cat /opt/arxplorer/config.yaml | head -20

# Set permissions
chown -R ubuntu:ubuntu /opt/arxplorer

# Install CloudWatch agent (optional)
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb

echo ""
echo "✓ Query API infrastructure setup complete!"
echo ""
echo "======================================"
echo "NEXT STEP: Deploy application code"
echo "======================================"
echo ""
echo "From your local machine, run:"
echo "  bash scripts/deploy_query_api.sh"
echo ""
echo "This will:"
echo "  1. Upload src/ and scripts/ directories"
echo "  2. Enable and start the systemd service"
echo "  3. Verify the API is running"
echo ""
