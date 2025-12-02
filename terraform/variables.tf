# Terraform Variables for ArXplorer AWS Infrastructure

# AWS Configuration
variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "arxplorer"
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR block for public subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH to instances"
  type        = string
}

variable "allowed_api_cidr" {
  description = "CIDR block allowed to access APIs (vLLM, Milvus)"
  type        = string
}

# EC2 Configuration
variable "key_name" {
  description = "Name of SSH key pair for EC2 instances"
  type        = string
}

# Instance Types
variable "vllm_instance_type" {
  description = "EC2 instance type for vLLM (GPU required)"
  type        = string
  default     = "g5.xlarge"
}

variable "milvus_instance_type" {
  description = "EC2 instance type for Milvus"
  type        = string
  default     = "c5.2xlarge"
}

variable "query_api_instance_type" {
  description = "EC2 instance type for Query API (t3.xlarge for CPU, g4dn.xlarge for GPU)"
  type        = string
  default     = "t3.xlarge"
}

# Elastic IP
variable "use_elastic_ip" {
  description = "Whether to use Elastic IPs for static addressing"
  type        = bool
  default     = false
}

# vLLM Configuration
variable "enable_vllm" {
  description = "Whether to deploy vLLM instance (requires GPU quota)"
  type        = bool
  default     = true
}

variable "vllm_model" {
  description = "HuggingFace model to serve with vLLM"
  type        = string
  default     = "Qwen/Qwen3-4B-AWQ"
}

variable "vllm_port" {
  description = "Port for vLLM API server"
  type        = number
  default     = 8000
}

variable "gpu_memory_utilization" {
  description = "GPU memory utilization (0.0-1.0)"
  type        = number
  default     = 0.5
}

variable "max_model_len" {
  description = "Maximum model context length"
  type        = number
  default     = 512
}

variable "quantization" {
  description = "Quantization method (awq, gptq, etc.)"
  type        = string
  default     = "awq"
}

variable "kv_cache_dtype" {
  description = "Data type for KV cache (auto, fp8, fp16)"
  type        = string
  default     = "fp8"
}

variable "enforce_eager" {
  description = "Whether to enforce eager execution in vLLM"
  type        = bool
  default     = true
}

# Milvus Configuration
variable "milvus_version" {
  description = "Milvus Docker image version"
  type        = string
  default     = "v2.4.15"
}

variable "milvus_ebs_size" {
  description = "Size of EBS volume for Milvus data (GB)"
  type        = number
  default     = 500
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Number of days to retain S3 backups"
  type        = number
  default     = 30
}

# Monitoring
variable "enable_alarms" {
  description = "Whether to enable CloudWatch alarms"
  type        = bool
  default     = false
}

# Query API Configuration
variable "enable_query_api" {
  description = "Whether to deploy Query API instance"
  type        = bool
  default     = true
}

variable "query_api_port" {
  description = "Port for Query API server"
  type        = number
  default     = 8080
}

variable "query_api_workers" {
  description = "Number of uvicorn workers for Query API"
  type        = number
  default     = 2
}

variable "allowed_ip" {
  description = "IP address allowed to access Query API (CIDR notation)"
  type        = string
  default     = "0.0.0.0/0"
}

# Bedrock Configuration
variable "bedrock_region" {
  description = "AWS region for Bedrock API"
  type        = string
  default     = "us-east-1"
}

variable "bedrock_model_id" {
  description = "Bedrock model ID for LLM query rewriting"
  type        = string
  default     = "mistral.mistral-7b-instruct-v0:2"
}

# HuggingFace Configuration
variable "hf_token" {
  description = "HuggingFace API token for model downloads (optional, for gated models)"
  type        = string
  default     = ""
  sensitive   = true
}
