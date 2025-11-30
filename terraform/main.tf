# Main Terraform configuration for ArXplorer AWS infrastructure

terraform {
  required_version = ">= 1.5"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Optional: Use S3 backend for state management
  # backend "s3" {
  #   bucket = "arxplorer-terraform-state"
  #   key    = "prod/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "ArXplorer"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "ubuntu_gpu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# VPC and Networking
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-subnet"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# Security Groups
resource "aws_security_group" "vllm" {
  name        = "${var.project_name}-vllm-sg"
  description = "Security group for vLLM instance"
  vpc_id      = aws_vpc.main.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "SSH access"
  }

  # vLLM API access
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_api_cidr]
    description = "vLLM API access"
  }

  # Outbound internet access
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = {
    Name = "${var.project_name}-vllm-sg"
  }
}

resource "aws_security_group" "milvus" {
  name        = "${var.project_name}-milvus-sg"
  description = "Security group for Milvus instance"
  vpc_id      = aws_vpc.main.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "SSH access"
  }

  # Milvus gRPC
  ingress {
    from_port   = 19530
    to_port     = 19530
    protocol    = "tcp"
    cidr_blocks = [var.allowed_api_cidr]
    description = "Milvus gRPC API"
  }

  # Milvus HTTP (optional, for web UI)
  ingress {
    from_port   = 9091
    to_port     = 9091
    protocol    = "tcp"
    cidr_blocks = [var.allowed_api_cidr]
    description = "Milvus HTTP API"
  }

  # Outbound internet access
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = {
    Name = "${var.project_name}-milvus-sg"
  }
}

# IAM Roles and Policies
resource "aws_iam_role" "vllm" {
  name = "${var.project_name}-vllm-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-vllm-role"
  }
}

resource "aws_iam_role_policy" "vllm_cloudwatch" {
  name = "${var.project_name}-vllm-cloudwatch-policy"
  role = aws_iam_role.vllm.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "ec2:DescribeVolumes",
          "ec2:DescribeTags",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "vllm" {
  name = "${var.project_name}-vllm-profile"
  role = aws_iam_role.vllm.name
}

resource "aws_iam_role" "milvus" {
  name = "${var.project_name}-milvus-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-milvus-role"
  }
}

resource "aws_iam_role_policy" "milvus_s3" {
  name = "${var.project_name}-milvus-s3-policy"
  role = aws_iam_role.milvus.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "ec2:DescribeVolumes",
          "ec2:DescribeTags",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "milvus" {
  name = "${var.project_name}-milvus-profile"
  role = aws_iam_role.milvus.name
}

# S3 Bucket for Milvus Backups
resource "aws_s3_bucket" "backups" {
  bucket = "${var.project_name}-backups-${var.environment}"

  tags = {
    Name = "${var.project_name}-backups"
  }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "delete-old-backups"
    status = "Enabled"

    filter {
      prefix = ""
    }

    expiration {
      days = var.backup_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# EBS Volume for Milvus
resource "aws_ebs_volume" "milvus_data" {
  availability_zone = data.aws_availability_zones.available.names[0]
  size              = var.milvus_ebs_size
  type              = "gp3"
  iops              = 3000
  throughput        = 125
  encrypted         = true

  tags = {
    Name = "${var.project_name}-milvus-data"
  }
}

# EC2 Instance for vLLM (optional - requires GPU quota)
resource "aws_instance" "vllm" {
  count                  = var.enable_vllm ? 1 : 0
  ami                    = data.aws_ami.ubuntu_gpu.id
  instance_type          = var.vllm_instance_type
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.vllm.id]
  iam_instance_profile   = aws_iam_instance_profile.vllm.name

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data_vllm.sh", {
    vllm_model           = var.vllm_model
    vllm_port            = var.vllm_port
    gpu_memory_util      = var.gpu_memory_utilization
    max_model_len        = var.max_model_len
    quantization         = var.quantization
    kv_cache_dtype       = var.kv_cache_dtype
    enforce_eager        = var.enforce_eager
  })

  tags = {
    Name = "${var.project_name}-vllm"
  }

  lifecycle {
    ignore_changes = [user_data]
  }
}

# EC2 Instance for Milvus
resource "aws_instance" "milvus" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.milvus_instance_type
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.milvus.id]
  iam_instance_profile   = aws_iam_instance_profile.milvus.name

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data_milvus.sh", {
    milvus_version = var.milvus_version
    s3_bucket      = aws_s3_bucket.backups.id
    aws_region     = var.aws_region
  })

  tags = {
    Name = "${var.project_name}-milvus"
  }

  lifecycle {
    ignore_changes = [user_data]
  }
}

# Attach EBS volume to Milvus instance
resource "aws_volume_attachment" "milvus_data" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.milvus_data.id
  instance_id = aws_instance.milvus.id

  # Wait for instance to be running
  depends_on = [aws_instance.milvus]
}

# Elastic IPs (optional, for static IPs)
resource "aws_eip" "vllm" {
  count    = var.use_elastic_ip && var.enable_vllm ? 1 : 0
  instance = aws_instance.vllm[0].id
  domain   = "vpc"

  tags = {
    Name = "${var.project_name}-vllm-eip"
  }

  depends_on = [aws_internet_gateway.main]
}

resource "aws_eip" "milvus" {
  count    = var.use_elastic_ip ? 1 : 0
  instance = aws_instance.milvus.id
  domain   = "vpc"

  tags = {
    Name = "${var.project_name}-milvus-eip"
  }

  depends_on = [aws_internet_gateway.main]
}

# CloudWatch Alarms (optional)
resource "aws_cloudwatch_metric_alarm" "vllm_cpu" {
  count               = var.enable_alarms && var.enable_vllm ? 1 : 0
  alarm_name          = "${var.project_name}-vllm-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "This metric monitors vLLM EC2 CPU utilization"

  dimensions = {
    InstanceId = aws_instance.vllm[0].id
  }
}

resource "aws_cloudwatch_metric_alarm" "milvus_cpu" {
  count               = var.enable_alarms ? 1 : 0
  alarm_name          = "${var.project_name}-milvus-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "This metric monitors Milvus EC2 CPU utilization"

  dimensions = {
    InstanceId = aws_instance.milvus.id
  }
}
