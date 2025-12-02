# AWS Region
aws_region = "ca-central-1"

# Environment
environment = "prod"

# Project Name
project_name = "arxplorer"

# Networking
vpc_cidr            = "10.0.0.0/16"
public_subnet_cidr  = "10.0.1.0/24"
allowed_ssh_cidr    = "0.0.0.0/32"  # CHANGE THIS to your IP 
allowed_api_cidr    = "0.0.0.0/32" # CHANGE THIS to your IP 

# EC2 Configuration
key_name = "arxplorer-key"  # CHANGE THIS to your key pair name

# Instance Types
vllm_instance_type     = "g5.xlarge"   
milvus_instance_type   = "c5.2xlarge"  
query_api_instance_type = "t3.xlarge" 

# Use Elastic IPs for static addressing
use_elastic_ip = false

# vLLM Configuration
vllm_model              = "Qwen/Qwen3-4B-AWQ"
vllm_port               = 8000
gpu_memory_utilization  = 0.5
max_model_len           = 512
quantization            = "awq"
kv_cache_dtype          = "fp8"
enforce_eager           = true

# Milvus Configuration
milvus_version  = "v2.4.15"
milvus_ebs_size = 500  # GB

# Backup Configuration
backup_retention_days = 30

# vLLM Instance (requires GPU quota approval)
enable_vllm = false  # keep false because no gpu quota just use aws bedrock

# Query API Configuration
enable_query_api  = true
query_api_port    = 8080
query_api_workers = 2

# Monitoring
enable_alarms = false  # Set to true to enable CloudWatch alarms
