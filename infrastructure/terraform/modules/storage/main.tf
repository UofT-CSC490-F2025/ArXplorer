# ArXplorer Storage Module - S3 Buckets for Academic Search Data
# This module creates the data storage foundation for ArXplorer

# Random suffix to ensure globally unique bucket names
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
  numeric = true
}

# S3 Bucket #1: Raw ArXiv Data (Kaggle Dataset)
resource "aws_s3_bucket" "raw_data" {
  bucket = "${var.bucket_prefix}-${var.environment}-raw-data-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-raw-data"
    Purpose     = "Store original ArXiv dataset"
    Environment = var.environment
    DataType    = "Raw Academic Papers"
  }
}

# S3 Bucket #2: Processed Papers  
resource "aws_s3_bucket" "processed" {
  bucket = "${var.bucket_prefix}-${var.environment}-processed-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-processed"
    Purpose     = "Store cleaned and processed academic papers"
    Environment = var.environment
    DataType    = "Processed Text Data"
  }
}

# S3 Bucket #3: Embeddings and Vector Data
resource "aws_s3_bucket" "embeddings" {
  bucket = "${var.bucket_prefix}-${var.environment}-embeddings-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-embeddings"
    Purpose     = "Store SciBERT embeddings and FAISS search indices"
    Environment = var.environment
    DataType    = "Vector Embeddings"
  }
}

# S3 Bucket #4: Automated Backups
resource "aws_s3_bucket" "backups" {
  bucket = "${var.bucket_prefix}-${var.environment}-backups-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-backups"
    Purpose     = "Store automated backups and disaster recovery data"
    Environment = var.environment
    DataType    = "Backup Data"
  }
}

# Configure Bucket Versioning (Data Protection)
resource "aws_s3_bucket_versioning" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id
  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_versioning" "processed" {
  bucket = aws_s3_bucket.processed.id
  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_versioning" "embeddings" {
  bucket = aws_s3_bucket.embeddings.id
  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  versioning_configuration {
    status = "Enabled"  # Always enable versioning for backups
  }
}

# Server-Side Encryption (Security)
resource "aws_s3_bucket_server_side_encryption_configuration" "raw_data" {
  count  = var.enable_encryption ? 1 : 0
  bucket = aws_s3_bucket.raw_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "processed" {
  count  = var.enable_encryption ? 1 : 0
  bucket = aws_s3_bucket.processed.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "embeddings" {
  count  = var.enable_encryption ? 1 : 0
  bucket = aws_s3_bucket.embeddings.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  count  = var.enable_encryption ? 1 : 0
  bucket = aws_s3_bucket.backups.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# Block Public Access (Security Best Practice)
resource "aws_s3_bucket_public_access_block" "raw_data" {
  count  = var.block_public_access ? 1 : 0
  bucket = aws_s3_bucket.raw_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "processed" {
  count  = var.block_public_access ? 1 : 0
  bucket = aws_s3_bucket.processed.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "embeddings" {
  count  = var.block_public_access ? 1 : 0
  bucket = aws_s3_bucket.embeddings.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "backups" {
  count  = var.block_public_access ? 1 : 0
  bucket = aws_s3_bucket.backups.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle Management for Cost Optimization
resource "aws_s3_bucket_lifecycle_configuration" "processed" {
  count  = var.lifecycle_enabled ? 1 : 0
  bucket = aws_s3_bucket.processed.id

  rule {
    id     = "processed_data_lifecycle"
    status = "Enabled"

    # Move to cheaper storage after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    # Archive to Glacier after 90 days
    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    # Delete old versions after 90 days
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "backup_lifecycle"
    status = "Enabled"

    # Move to cheaper storage after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    # Archive after 90 days
    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    # Delete backups after retention period
    expiration {
      days = var.backup_retention_days
    }
  }
}

# S3 Bucket Notifications (for future pipeline automation)
resource "aws_s3_bucket_notification" "raw_data_notification" {
  bucket = aws_s3_bucket.raw_data.id

  # Placeholder for future Lambda triggers or SQS notifications
  # when new ArXiv data is uploaded
}

# CloudWatch Log Group for S3 access logging (monitoring)
resource "aws_cloudwatch_log_group" "s3_access_logs" {
  name              = "/aws/s3/${var.project_name}-${var.environment}-access-logs"
  retention_in_days = 30

  tags = {
    Name        = "${var.project_name}-${var.environment}-s3-logs"
    Environment = var.environment
  }
}
