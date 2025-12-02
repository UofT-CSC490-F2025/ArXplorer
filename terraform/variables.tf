# Core settings
variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
}

variable "project_name" {
  description = "Prefix for tagging and resource naming"
  type        = string
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "public_subnet_cidr" {
  description = "CIDR block for the public subnet (ALB / bastion)"
  type        = string
}

variable "public_subnet_cidr_b" {
  description = "Second public subnet CIDR (ALB requires two AZs)"
  type        = string
}

variable "private_app_subnet_cidr" {
  description = "CIDR block for the application layer private subnet"
  type        = string
}

variable "private_ml_subnet_cidr" {
  description = "CIDR block for the ML services private subnet"
  type        = string
}

variable "private_data_subnet_cidr" {
  description = "CIDR block for the data layer private subnet"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into EC2 instances"
  type        = string
}

variable "key_name" {
  description = "EC2 key pair name for SSH access"
  type        = string
}

# Search API (Application layer)
variable "search_api_instance_type" {
  description = "Instance type for Search API autoscaling group"
  type        = string
  default     = "t3.medium"
}

variable "search_api_min_size" {
  description = "Minimum number of Search API instances"
  type        = number
  default     = 1
}

variable "search_api_max_size" {
  description = "Maximum number of Search API instances"
  type        = number
  default     = 3
}

variable "search_api_desired_capacity" {
  description = "Desired number of Search API instances"
  type        = number
  default     = 2
}

variable "search_api_port" {
  description = "Container/listener port for the Search API"
  type        = number
  default     = 8000
}

variable "search_api_health_path" {
  description = "HTTP path used by the ALB health check for the Search API"
  type        = string
  default     = "/"
}

variable "search_api_image" {
  description = "Container image for the Search API (used in the user data bootstrap)"
  type        = string
  default     = "nginx:stable-alpine"
}

# Milvus / data layer
variable "milvus_instance_type" {
  description = "Instance type for Milvus"
  type        = string
}

variable "milvus_version" {
  description = "Milvus version to install"
  type        = string
}

variable "milvus_ebs_size" {
  description = "EBS volume size (GB) for Milvus data"
  type        = number
}

# Backups / S3
variable "use_elastic_ip" {
  description = "Use Elastic IPs for static addressing"
  type        = bool
  default     = false
}

variable "backup_retention_days" {
  description = "Retention in days for backup lifecycle"
  type        = number
  default     = 30
}

# Monitoring
variable "enable_alarms" {
  description = "Toggle to enable CloudWatch alarms"
  type        = bool
  default     = false
}

# Offline workers (ECS Fargate)
variable "offline_task_image" {
  description = "Container image for offline pipeline worker (embedding/ingest jobs)"
  type        = string
  default     = "public.ecr.aws/amazonlinux/amazonlinux:2023"
}

variable "offline_task_cpu" {
  description = "CPU units for offline worker task (Fargate)"
  type        = number
  default     = 1024
}

variable "offline_task_memory" {
  description = "Memory (MiB) for offline worker task (Fargate)"
  type        = number
  default     = 2048
}

variable "offline_desired_count" {
  description = "Desired count for offline ECS service (set 0 to keep idle and run tasks on-demand)"
  type        = number
  default     = 0
}
