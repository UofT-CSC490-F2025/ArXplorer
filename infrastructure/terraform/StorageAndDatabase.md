# Your Storage & Database Module Guide

## Your Mission: Data Layer for ArXplorer

You are responsible for building the **data foundation** that powers the ArXplorer academic search assistant. This includes all data storage, database management, and data security components.

---

## Phase 1: Storage Module (S3 Buckets) - Days 1-2

### Step 1: Navigate to Your Storage Module
```powershell
cd modules\storage
```

### Step 2: Create variables.tf for Storage
Add this content to `modules\storage\variables.tf`:

```hcl
variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "bucket_prefix" {
  description = "Prefix for S3 bucket names"
  type        = string
}

variable "enable_versioning" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

variable "lifecycle_enabled" {
  description = "Enable lifecycle management"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}
```

### Step 3: Create main.tf for Storage
Add this content to `modules\storage\main.tf`:

```hcl
# Random suffix for globally unique bucket names
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# S3 Bucket for Raw ArXiv Data (Kaggle dataset)
resource "aws_s3_bucket" "raw_data" {
  bucket = "${var.bucket_prefix}-${var.environment}-raw-data-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-raw-data"
    Purpose     = "Store original Kaggle ArXiv dataset"
    Environment = var.environment
  }
}

# S3 Bucket for Processed Papers
resource "aws_s3_bucket" "processed" {
  bucket = "${var.bucket_prefix}-${var.environment}-processed-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-processed"
    Purpose     = "Store cleaned and processed papers"
    Environment = var.environment
  }
}

# S3 Bucket for Embeddings and Vector Data
resource "aws_s3_bucket" "embeddings" {
  bucket = "${var.bucket_prefix}-${var.environment}-embeddings-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-embeddings"
    Purpose     = "Store SciBERT embeddings and FAISS indices"
    Environment = var.environment
  }
}

# S3 Bucket for Backups
resource "aws_s3_bucket" "backups" {
  bucket = "${var.bucket_prefix}-${var.environment}-backups-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-backups"
    Purpose     = "Store automated backups and disaster recovery data"
    Environment = var.environment
  }
}

# Bucket Versioning Configuration
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

# Server-Side Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "processed" {
  bucket = aws_s3_bucket.processed.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "embeddings" {
  bucket = aws_s3_bucket.embeddings.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
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

# Block Public Access (Security Best Practice)
resource "aws_s3_bucket_public_access_block" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "processed" {
  bucket = aws_s3_bucket.processed.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "embeddings" {
  bucket = aws_s3_bucket.embeddings.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "backups" {
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

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

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

    transition {
      days          = 7
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    expiration {
      days = var.backup_retention_days
    }
  }
}

# S3 Bucket Notifications (for pipeline automation)
resource "aws_s3_bucket_notification" "raw_data_notification" {
  bucket = aws_s3_bucket.raw_data.id

  # This can trigger Lambda functions or SQS when new data arrives
  # For now, we'll leave it empty but structure is ready
}
```

### Step 4: Create outputs.tf for Storage
Add this content to `modules\storage\outputs.tf`:

```hcl
output "bucket_names" {
  description = "Names of all S3 buckets"
  value = {
    raw_data    = aws_s3_bucket.raw_data.id
    processed   = aws_s3_bucket.processed.id
    embeddings  = aws_s3_bucket.embeddings.id
    backups     = aws_s3_bucket.backups.id
  }
}

output "bucket_arns" {
  description = "ARNs of all S3 buckets for IAM policies"
  value = {
    raw_data    = aws_s3_bucket.raw_data.arn
    processed   = aws_s3_bucket.processed.arn
    embeddings  = aws_s3_bucket.embeddings.arn
    backups     = aws_s3_bucket.backups.arn
  }
}

output "bucket_domain_names" {
  description = "Domain names of S3 buckets"
  value = {
    raw_data    = aws_s3_bucket.raw_data.bucket_domain_name
    processed   = aws_s3_bucket.processed.bucket_domain_name
    embeddings  = aws_s3_bucket.embeddings.bucket_domain_name
    backups     = aws_s3_bucket.backups.bucket_domain_name
  }
}

output "bucket_regional_domain_names" {
  description = "Regional domain names of S3 buckets"
  value = {
    raw_data    = aws_s3_bucket.raw_data.bucket_regional_domain_name
    processed   = aws_s3_bucket.processed.bucket_regional_domain_name
    embeddings  = aws_s3_bucket.embeddings.bucket_regional_domain_name
    backups     = aws_s3_bucket.backups.bucket_regional_domain_name
  }
}
```

### Step 5: Test Storage Module
```powershell
# Go to dev environment
cd ..\..\environments\dev

# Initialize Terraform (if not done already)
terraform init

# Plan only storage module
terraform plan -target=module.storage

# Apply only storage module (test in isolation)
terraform apply -target=module.storage
```

**Expected Result**: 4 S3 buckets created with encryption, versioning, and security settings.

---

## Phase 2: Database Module (RDS) - Days 2-3

### Step 6: Navigate to Database Module
```powershell
cd ..\..\modules\database
```

### Step 7: Create variables.tf for Database
Add this content to `modules\database\variables.tf`:

```hcl
variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID from networking module"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for database placement"
  type        = list(string)
}

variable "database_security_group_id" {
  description = "Security group ID for database access"
  type        = string
}

variable "instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "database_name" {
  description = "Name of the database"
  type        = string
  default     = "arxplorer"
}

variable "username" {
  description = "Database master username"
  type        = string
  default     = "admin"
}

variable "allocated_storage" {
  description = "Allocated storage in GB"
  type        = number
  default     = 20
}

variable "max_allocated_storage" {
  description = "Maximum allocated storage for autoscaling"
  type        = number
  default     = 100
}

variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

variable "multi_az" {
  description = "Enable Multi-AZ deployment"
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = false
}
```

### Step 8: Create main.tf for Database
Add this content to `modules\database\main.tf`:

```hcl
# Random password for database
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# AWS Secrets Manager secret for database credentials
resource "aws_secretsmanager_secret" "db_credentials" {
  name        = "${var.project_name}-${var.environment}-db-credentials"
  description = "Database credentials for ArXplorer ${var.environment}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-secret"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = var.username
    password = random_password.db_password.result
    engine   = "mysql"
    host     = aws_db_instance.main.endpoint
    port     = aws_db_instance.main.port
    dbname   = var.database_name
  })
}

# DB Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}-db-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-subnet-group"
    Environment = var.environment
  }
}

# Parameter Group for MySQL optimization
resource "aws_db_parameter_group" "main" {
  family = "mysql8.0"
  name   = "${var.project_name}-${var.environment}-db-params"

  parameter {
    name  = "innodb_buffer_pool_size"
    value = "{DBInstanceClassMemory*3/4}"
  }

  parameter {
    name  = "max_connections"
    value = "200"
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-params"
    Environment = var.environment
  }
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-${var.environment}-db"

  # Database Configuration
  engine         = "mysql"
  engine_version = "8.0"
  instance_class = var.instance_class
  db_name        = var.database_name
  username       = var.username
  password       = random_password.db_password.result

  # Storage Configuration
  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = "gp2"
  storage_encrypted     = true

  # Network Configuration
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [var.database_security_group_id]
  publicly_accessible    = false

  # High Availability
  multi_az = var.multi_az

  # Backup Configuration
  backup_retention_period = var.backup_retention_period
  backup_window          = "03:00-04:00"  # UTC
  maintenance_window     = "Sun:04:00-Sun:05:00"  # UTC

  # Parameter and Option Groups
  parameter_group_name = aws_db_parameter_group.main.name

  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn

  # Security
  deletion_protection = var.deletion_protection

  # Performance Insights (for production)
  performance_insights_enabled = var.environment == "prod" ? true : false

  tags = {
    Name        = "${var.project_name}-${var.environment}-database"
    Environment = var.environment
    Purpose     = "ArXplorer paper metadata and search results"
  }

  # Prevent accidental deletion
  lifecycle {
    prevent_destroy = false  # Set to true for production
  }
}

# IAM Role for Enhanced Monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "${var.project_name}-${var.environment}-rds-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# CloudWatch Log Groups for database logs
resource "aws_cloudwatch_log_group" "database_logs" {
  name              = "/aws/rds/instance/${aws_db_instance.main.identifier}/error"
  retention_in_days = 30

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-logs"
    Environment = var.environment
  }
}

# Database Schema Setup (via null resource)
resource "null_resource" "database_schema" {
  depends_on = [aws_db_instance.main]

  provisioner "local-exec" {
    command = <<-EOT
      echo "Database created: ${aws_db_instance.main.endpoint}"
      echo "Database schema will be initialized by application deployment"
      echo "Connection string stored in AWS Secrets Manager: ${aws_secretsmanager_secret.db_credentials.name}"
    EOT
  }

  triggers = {
    database_endpoint = aws_db_instance.main.endpoint
  }
}
```

### Step 9: Create outputs.tf for Database
Add this content to `modules\database\outputs.tf`:

```hcl
output "endpoint" {
  description = "Database endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "port" {
  description = "Database port"
  value       = aws_db_instance.main.port
}

output "database_name" {
  description = "Database name"
  value       = aws_db_instance.main.db_name
}

output "username" {
  description = "Database username"
  value       = aws_db_instance.main.username
  sensitive   = true
}

output "secret_arn" {
  description = "ARN of the secret containing database credentials"
  value       = aws_secretsmanager_secret.db_credentials.arn
}

output "secret_name" {
  description = "Name of the secret containing database credentials"
  value       = aws_secretsmanager_secret.db_credentials.name
}

output "db_instance_id" {
  description = "RDS instance ID"
  value       = aws_db_instance.main.id
}

output "db_subnet_group_name" {
  description = "DB subnet group name"
  value       = aws_db_subnet_group.main.name
}

output "parameter_group_name" {
  description = "DB parameter group name"
  value       = aws_db_parameter_group.main.name
}
```

### Step 10: Create Database Schema Script
Create `modules\database\schema.sql` for your ArXplorer database structure:

```sql
-- ArXplorer Database Schema
-- This file defines the database structure for the academic search assistant

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS arxplorer CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE arxplorer;

-- Papers table - stores original paper metadata
CREATE TABLE IF NOT EXISTS papers (
    id VARCHAR(50) PRIMARY KEY,  -- ArXiv ID
    title TEXT NOT NULL,
    authors TEXT,
    abstract TEXT,
    categories VARCHAR(255),
    doi VARCHAR(100),
    journal_ref TEXT,
    arxiv_url VARCHAR(255),
    pdf_url VARCHAR(255),
    submitted_date DATE,
    updated_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_categories (categories),
    INDEX idx_submitted_date (submitted_date),
    FULLTEXT idx_title_abstract (title, abstract)
) ENGINE=InnoDB;

-- Processed papers table - stores cleaned and processed text
CREATE TABLE IF NOT EXISTS processed_papers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    paper_id VARCHAR(50) NOT NULL,
    cleaned_title TEXT,
    cleaned_abstract TEXT,
    keywords TEXT,
    summary TEXT,
    word_count INT,
    processing_version VARCHAR(20),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    INDEX idx_paper_id (paper_id),
    INDEX idx_processing_version (processing_version),
    FULLTEXT idx_cleaned_content (cleaned_title, cleaned_abstract, keywords)
) ENGINE=InnoDB;

-- Embeddings table - stores vector embeddings metadata
CREATE TABLE IF NOT EXISTS embeddings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    paper_id VARCHAR(50) NOT NULL,
    embedding_type VARCHAR(50) NOT NULL,  -- 'scibert', 'title', 'abstract'
    vector_dimension INT NOT NULL,
    s3_path VARCHAR(500),  -- Path to actual embedding file in S3
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    INDEX idx_paper_id (paper_id),
    INDEX idx_embedding_type (embedding_type),
    INDEX idx_model_version (model_version),
    UNIQUE KEY unique_paper_embedding (paper_id, embedding_type, model_version)
) ENGINE=InnoDB;

-- Search results cache - stores frequently accessed search results
CREATE TABLE IF NOT EXISTS search_cache (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_hash VARCHAR(64) NOT NULL,  -- MD5 hash of search query
    query_text TEXT NOT NULL,
    results JSON,  -- JSON array of paper IDs and scores
    result_count INT,
    search_type VARCHAR(50),  -- 'semantic', 'keyword', 'hybrid'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    
    INDEX idx_query_hash (query_hash),
    INDEX idx_search_type (search_type),
    INDEX idx_created_at (created_at),
    INDEX idx_expires_at (expires_at)
) ENGINE=InnoDB;

-- User queries log - for analytics and improvement
CREATE TABLE IF NOT EXISTS user_queries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100),
    query_text TEXT NOT NULL,
    search_type VARCHAR(50),
    result_count INT,
    response_time_ms INT,
    user_ip VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_session_id (session_id),
    INDEX idx_search_type (search_type),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

-- Processing jobs table - track pipeline processing status
CREATE TABLE IF NOT EXISTS processing_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL,  -- 'ingestion', 'embedding', 'indexing'
    status VARCHAR(20) NOT NULL,    -- 'pending', 'running', 'completed', 'failed'
    input_path VARCHAR(500),
    output_path VARCHAR(500),
    parameters JSON,
    progress_percent INT DEFAULT 0,
    records_processed INT DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_job_type (job_type),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

-- System configuration table
CREATE TABLE IF NOT EXISTS system_config (
    config_key VARCHAR(100) PRIMARY KEY,
    config_value TEXT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Insert default configuration values
INSERT INTO system_config (config_key, config_value, description) VALUES
('embedding_model', 'allenai/scibert_scivocab_uncased', 'Current embedding model being used'),
('vector_dimension', '768', 'Dimension of embedding vectors'),
('index_version', '1.0', 'Current FAISS index version'),
('last_full_reindex', NULL, 'Timestamp of last complete reindexing'),
('max_search_results', '100', 'Maximum number of search results to return')
ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP;

-- Create views for common queries
CREATE OR REPLACE VIEW recent_papers AS
SELECT p.*, pp.word_count, pp.processed_at
FROM papers p
LEFT JOIN processed_papers pp ON p.id = pp.paper_id
WHERE p.submitted_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
ORDER BY p.submitted_date DESC;

CREATE OR REPLACE VIEW embedding_stats AS
SELECT 
    embedding_type,
    model_version,
    COUNT(*) as count,
    MIN(created_at) as first_created,
    MAX(created_at) as last_created
FROM embeddings
GROUP BY embedding_type, model_version;

-- Stored procedures for common operations
DELIMITER //

CREATE PROCEDURE GetPaperWithEmbeddings(IN paper_id VARCHAR(50))
BEGIN
    SELECT p.*, pp.cleaned_title, pp.cleaned_abstract, pp.keywords,
           e.embedding_type, e.s3_path, e.model_version
    FROM papers p
    LEFT JOIN processed_papers pp ON p.id = pp.paper_id
    LEFT JOIN embeddings e ON p.id = e.paper_id
    WHERE p.id = paper_id;
END //

CREATE PROCEDURE UpdateProcessingJob(
    IN job_id INT,
    IN new_status VARCHAR(20),
    IN progress INT,
    IN records INT,
    IN error_msg TEXT
)
BEGIN
    UPDATE processing_jobs 
    SET status = new_status,
        progress_percent = progress,
        records_processed = records,
        error_message = error_msg,
        completed_at = CASE WHEN new_status IN ('completed', 'failed') THEN NOW() ELSE completed_at END
    WHERE id = job_id;
END //

DELIMITER ;

-- Create indexes for better performance
ALTER TABLE papers ADD INDEX idx_title_length ((CHAR_LENGTH(title)));
ALTER TABLE processed_papers ADD INDEX idx_word_count (word_count);
ALTER TABLE user_queries ADD INDEX idx_response_time (response_time_ms);

-- Grant permissions (will be handled by application)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON arxplorer.* TO 'app_user'@'%';

COMMIT;
```

### Step 11: Test Database Module
```powershell
# Go to dev environment
cd ..\..\environments\dev

# Plan database module (will show dependency on networking)
terraform plan -target=module.database

# Apply database module
terraform apply -target=module.database
```

**Expected Result**: RDS MySQL instance created in private subnets with encryption and monitoring.

---

## Phase 3: Integration & Testing - Day 4

### Step 12: Update Root Outputs
Go back to the root `outputs.tf` and update it with real values:

```powershell
cd ..\..\
```

Replace the placeholder outputs in `outputs.tf` with actual module references:

```hcl
# Storage Module Outputs (Your responsibility)
output "storage_bucket_names" {
  description = "Names of S3 buckets created by storage module"
  value       = module.storage.bucket_names
}

output "storage_bucket_arns" {
  description = "ARNs of S3 buckets for IAM policies"  
  value       = module.storage.bucket_arns
}

# Database Module Outputs (Your responsibility)
output "database_endpoint" {
  description = "RDS database endpoint for application connections"
  value       = module.database.endpoint
  sensitive   = true
}

output "database_port" {
  description = "Database port number"
  value       = module.database.port
}

output "database_secret_name" {
  description = "Name of the secret containing database credentials"
  value       = module.database.secret_name
}
```

### Step 13: Test Complete Data Layer
```powershell
# Test your modules together
terraform plan -target=module.storage -target=module.database

# Apply your data layer
terraform apply -target=module.storage -target=module.database
```

### Step 14: Verify Your Infrastructure
```powershell
# Check what was created
terraform show | grep -E "(bucket|db_instance)"

# Get output values
terraform output storage_bucket_names
terraform output database_secret_name
```

### Step 15: Create Data Upload Script
Create `scripts\upload_kaggle_data.py` to test your S3 buckets:

```python
#!/usr/bin/env python3
"""
Script to upload Kaggle ArXiv data to S3 buckets
This tests that your storage infrastructure works correctly
"""

import boto3
import json
import os
from pathlib import Path

def upload_sample_data():
    s3 = boto3.client('s3')
    
    # Get bucket names from Terraform output
    # You'll need to replace these with actual bucket names
    raw_bucket = "arxplorer-dev-raw-data-xxxxxxxx"
    processed_bucket = "arxplorer-dev-processed-xxxxxxxx"
    
    # Create sample data
    sample_paper = {
        "id": "2301.00001",
        "title": "Sample ArXiv Paper for Testing",
        "authors": "Test Author",
        "abstract": "This is a sample abstract for testing the ArXplorer infrastructure.",
        "categories": "cs.AI",
        "submitted": "2023-01-01"
    }
    
    # Upload to raw data bucket
    try:
        s3.put_object(
            Bucket=raw_bucket,
            Key="sample_data/test_paper.json",
            Body=json.dumps(sample_paper),
            ContentType="application/json"
        )
        print(f"✅ Successfully uploaded to {raw_bucket}")
        
        # Upload processed version
        processed_paper = {
            **sample_paper,
            "cleaned_abstract": "This is a cleaned abstract for testing ArXplorer infrastructure",
            "keywords": ["testing", "infrastructure", "arxiv"],
            "processed_at": "2023-01-01T12:00:00Z"
        }
        
        s3.put_object(
            Bucket=processed_bucket,
            Key="processed_data/test_paper_processed.json",
            Body=json.dumps(processed_paper),
            ContentType="application/json"
        )
        print(f"✅ Successfully uploaded to {processed_bucket}")
        
    except Exception as e:
        print(f"❌ Error uploading data: {e}")

if __name__ == "__main__":
    upload_sample_data()
```

---

## Phase 4: Connect with Teammate - Day 5

### Step 16: Coordinate Integration Points

**What you provide to your teammate:**

1. **S3 Bucket Names**: From `terraform output storage_bucket_names`
2. **Database Endpoint**: From `terraform output database_secret_name`
3. **Database Security Group**: Your teammate's networking module creates this

**What you need from your teammate:**

1. **VPC ID**: For placing your database
2. **Private Subnet IDs**: For database subnet group
3. **Database Security Group ID**: For database access rules

### Step 17: Update Module Dependencies

In `environments\dev\main.tf`, ensure proper dependencies:

```hcl
module "storage" {
  source = "../../modules/storage"
  
  project_name  = var.project_name
  environment   = var.environment
  bucket_prefix = var.storage_bucket_prefix
}

module "database" {
  source = "../../modules/database"
  
  project_name = var.project_name
  environment  = var.environment
  
  # Dependencies from teammate's networking module
  vpc_id                    = module.networking.vpc_id
  private_subnet_ids        = module.networking.private_subnet_ids
  database_security_group_id = module.networking.database_security_group_id
  
  # Database configuration
  instance_class = var.db_instance_class
  database_name  = var.db_name
}
```

---

## Success Criteria ✅

You'll know you're successful when:

- [ ] **Storage Module**: 4 S3 buckets created with encryption and versioning
- [ ] **Database Module**: RDS MySQL instance running in private subnets
- [ ] **Security**: All resources encrypted, no public access
- [ ] **Integration**: Your outputs available for teammate's compute module
- [ ] **Testing**: Can upload files to S3 and connect to database
- [ ] **Coordination**: Teammate can reference your resources in their modules

## Troubleshooting Guide

### Common Issues:

1. **Bucket naming conflicts**: S3 bucket names must be globally unique
   - **Solution**: Use random suffix in bucket names

2. **Database subnet group errors**: Not enough subnets across AZs
   - **Solution**: Ensure private_subnet_ids has subnets in different AZs

3. **Security group dependencies**: Database security group doesn't exist
   - **Solution**: Coordinate with teammate on networking module first

4. **Permission errors**: Can't create RDS or S3 resources
   - **Solution**: Check IAM permissions for your AWS user

### Debug Commands:
```powershell
# Validate all syntax
terraform validate

# Plan specific modules
terraform plan -target=module.storage
terraform plan -target=module.database

# Show current state
terraform show

# Check AWS resources directly
aws s3 ls
aws rds describe-db-instances
```

## Next Steps After Completion

Once your storage and database modules are working:

1. **Performance Testing**: Test with larger datasets
2. **Backup Verification**: Ensure automated backups work
3. **Monitoring Setup**: Add CloudWatch alarms
4. **Documentation**: Document connection strings and procedures
5. **Security Review**: Audit IAM policies and access controls

**You're building the data foundation that makes ArXplorer possible! Your storage and database infrastructure will power academic search for students and researchers.**