variable "project_name" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }
variable "vpc_id" { type = string }
variable "data_subnet_ids" { type = list(string) }
variable "allowed_ssh_cidr" { type = string }
variable "use_elastic_ip" { type = bool }
variable "key_name" { type = string }
variable "milvus_instance_type" { type = string }
variable "milvus_version" { type = string }
variable "milvus_ebs_size" { type = number }
variable "backup_retention_days" { type = number }

variable "app_sg_id" { type = string }
variable "offline_sg_id" { type = string }

locals {
  data_buckets = {
    raw_data   = "raw-data"
    processed  = "processed"
    metadata   = "metadata"
    embeddings = "embeddings"
  }
}

resource "aws_security_group" "milvus" {
  name        = "${var.project_name}-milvus-sg"
  description = "Security group for Milvus instance"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "SSH access"
  }

  ingress {
    from_port       = 19530
    to_port         = 19530
    protocol        = "tcp"
    security_groups = [var.app_sg_id, var.offline_sg_id]
    description     = "Milvus gRPC API from app/offline"
  }

  ingress {
    from_port       = 9091
    to_port         = 9091
    protocol        = "tcp"
    security_groups = [var.app_sg_id, var.offline_sg_id]
    description     = "Milvus HTTP API from app/offline"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = { Name = "${var.project_name}-milvus-sg" }
}

resource "aws_iam_role" "milvus" {
  name = "${var.project_name}-milvus-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = { Service = "ec2.amazonaws.com" }
      }
    ]
  })

  tags = { Name = "${var.project_name}-milvus-role" }
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

resource "aws_s3_bucket" "data" {
  for_each = local.data_buckets
  bucket   = "${var.project_name}-${var.environment}-${each.value}"
  tags = {
    Name        = "${var.project_name}-${each.value}"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "data" {
  for_each = aws_s3_bucket.data
  bucket   = each.value.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  for_each = aws_s3_bucket.data
  bucket   = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  for_each = aws_s3_bucket.data
  bucket   = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "backups" {
  bucket = "${var.project_name}-backups-${var.environment}"
  tags   = { Name = "${var.project_name}-backups" }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  versioning_configuration { status = "Enabled" }
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
    filter { prefix = "" }
    expiration { days = var.backup_retention_days }
    noncurrent_version_expiration { noncurrent_days = 30 }
  }
}

resource "aws_ebs_volume" "milvus_data" {
  availability_zone = data.aws_availability_zones.available.names[0]
  size              = var.milvus_ebs_size
  type              = "gp3"
  iops              = 3000
  throughput        = 125
  encrypted         = true
  tags              = { Name = "${var.project_name}-milvus-data" }
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_instance" "milvus" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.milvus_instance_type
  key_name               = var.key_name
  subnet_id              = var.use_elastic_ip ? var.data_subnet_ids[0] : var.data_subnet_ids[0]
  vpc_security_group_ids = [aws_security_group.milvus.id]
  iam_instance_profile   = aws_iam_instance_profile.milvus.name

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = templatefile("${path.root}/user_data_milvus.sh", {
    milvus_version = var.milvus_version
    s3_bucket      = aws_s3_bucket.backups.id
    aws_region     = var.aws_region
  })

  tags = { Name = "${var.project_name}-milvus" }

  lifecycle { ignore_changes = [user_data] }
}

resource "aws_volume_attachment" "milvus_data" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.milvus_data.id
  instance_id = aws_instance.milvus.id
  depends_on  = [aws_instance.milvus]
}

resource "aws_eip" "milvus" {
  count    = var.use_elastic_ip ? 1 : 0
  instance = aws_instance.milvus.id
  domain   = "vpc"
  tags     = { Name = "${var.project_name}-milvus-eip" }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

output "milvus_private_ip" { value = aws_instance.milvus.private_ip }
output "milvus_public_ip" { value = var.use_elastic_ip ? aws_eip.milvus[0].public_ip : "" }
output "milvus_endpoint" { value = "${var.use_elastic_ip ? aws_eip.milvus[0].public_ip : aws_instance.milvus.private_ip}:19530" }
output "milvus_sg_id" { value = aws_security_group.milvus.id }
output "milvus_instance_id" { value = aws_instance.milvus.id }
output "milvus_ebs_volume_id" { value = aws_ebs_volume.milvus_data.id }
output "s3_backup_bucket" { value = aws_s3_bucket.backups.id }
output "s3_backup_bucket_arn" { value = aws_s3_bucket.backups.arn }
output "data_bucket_ids" { value = { for k, v in aws_s3_bucket.data : k => v.id } }
output "data_bucket_arns" { value = { for k, v in aws_s3_bucket.data : k => v.arn } }
