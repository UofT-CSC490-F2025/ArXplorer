# AWS Region
aws_region = "ca-central-1"

# Environment
environment = "prod"

# Project Name
project_name = "arxplorer"

# Networking
vpc_cidr            = "10.0.0.0/16"
public_subnet_cidr  = "10.0.1.0/24"
allowed_ssh_cidr    = "0.0.0.0/0"  # CHANGE THIS to your IP for security!
allowed_api_cidr    = "0.0.0.0/0"

# EC2 Configuration
key_name = "arxplorer-key"  # CHANGE THIS to your key pair name

# Instance Types
vllm_instance_type      = "g5.xlarge"   # ~$1/hr, 1x A10G GPU (24GB VRAM)
milvus_instance_type    = "c5.2xlarge"  # ~$0.34/hr, 8 vCPU, 16GB RAM
query_api_instance_type = "t3.xlarge"   # ~$0.17/hr, 4 vCPU, 16GB RAM (CPU inference)

# Use Elastic IPs for static addressing
use_elastic_ip = false

# vLLM Configuration (requires GPU quota - optional)
enable_vllm             = false  # for running llm rewriting on a gpu ec2; legacy
vllm_model              = "Qwen/Qwen3-4B-AWQ"
vllm_port               = 8000
gpu_memory_utilization  = 0.5
max_model_len           = 512
quantization            = "awq"
kv_cache_dtype          = "fp8"
enforce_eager           = true

# Query API Configuration
enable_query_api  = true
query_api_port    = 8080
query_api_workers = 2

# Milvus Configuration
milvus_version  = "v2.4.15"
milvus_ebs_size = 500  # GB

# Backup Configuration
backup_retention_days = 30

# Monitoring
enable_alarms = false  # Set to true to enable CloudWatch alarms

# HuggingFace Token (optional, for gated models like SPLADE v3)
# Get your token from: https://huggingface.co/settings/tokens
# hf_token = "hf_..."  # Uncomment and add your token to use full SPLADE v3 model
