# AWS Region
aws_region = "ca-central-1"

# Environment
environment = "prod"

# Project Name
project_name = "arxplorer-jjb08"

# Networking
vpc_cidr             = "10.0.0.0/16"
public_subnet_cidr   = "10.0.1.0/24"
public_subnet_cidr_b = "10.0.11.0/24"
private_app_subnet_cidr  = "10.0.2.0/24"
private_ml_subnet_cidr   = "10.0.3.0/24"
private_data_subnet_cidr = "10.0.4.0/24"
allowed_ssh_cidr     = "0.0.0.0/0"  # adjust to your IP for security

# EC2 Configuration
key_name = "arxplorer-jjb08-key"

# Instance Types
milvus_instance_type        = "c5.2xlarge"
search_api_instance_type    = "t3.medium"
search_api_min_size         = 1
search_api_max_size         = 3
search_api_desired_capacity = 2

# Elastic IP usage
use_elastic_ip = false

# Search API (ALB + ASG)
search_api_port        = 8000
search_api_health_path = "/"
search_api_image       = "nginx:stable-alpine"

# Milvus
milvus_version  = "v2.4.15"
milvus_ebs_size = 500

# Backups
backup_retention_days = 30

# Monitoring
enable_alarms = false

# Offline workers (ECS Fargate)
offline_task_image    = "472730591037.dkr.ecr.ca-central-1.amazonaws.com/arxplorer-offline:latest"
offline_task_cpu      = 2048
offline_task_memory   = 8192
offline_desired_count = 0
