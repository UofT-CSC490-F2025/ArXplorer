# Storage Module Outputs
# These values will be available to other modules and your teammate

# S3 Bucket Names - What your teammate needs to access the buckets
output "bucket_names" {
  description = "Names of all S3 buckets created for ArXplorer data storage"
  value = {
    raw_data    = aws_s3_bucket.raw_data.id
    processed   = aws_s3_bucket.processed.id
    embeddings  = aws_s3_bucket.embeddings.id
    backups     = aws_s3_bucket.backups.id
  }
}

# S3 Bucket ARNs - For IAM policies and permissions
output "bucket_arns" {
  description = "ARNs of all S3 buckets for IAM policy creation"
  value = {
    raw_data    = aws_s3_bucket.raw_data.arn
    processed   = aws_s3_bucket.processed.arn
    embeddings  = aws_s3_bucket.embeddings.arn
    backups     = aws_s3_bucket.backups.arn
  }
}

# S3 Bucket Domain Names - For direct access URLs
output "bucket_domain_names" {
  description = "Domain names of S3 buckets for direct access"
  value = {
    raw_data    = aws_s3_bucket.raw_data.bucket_domain_name
    processed   = aws_s3_bucket.processed.bucket_domain_name
    embeddings  = aws_s3_bucket.embeddings.bucket_domain_name
    backups     = aws_s3_bucket.backups.bucket_domain_name
  }
}

# Regional Domain Names - For regional access
output "bucket_regional_domain_names" {
  description = "Regional domain names of S3 buckets"
  value = {
    raw_data    = aws_s3_bucket.raw_data.bucket_regional_domain_name
    processed   = aws_s3_bucket.processed.bucket_regional_domain_name
    embeddings  = aws_s3_bucket.embeddings.bucket_regional_domain_name
    backups     = aws_s3_bucket.backups.bucket_regional_domain_name
  }
}

# Bucket Configuration Summary - For documentation
output "storage_summary" {
  description = "Summary of storage configuration"
  value = {
    project_name          = var.project_name
    environment          = var.environment
    total_buckets        = 4
    versioning_enabled   = var.enable_versioning
    encryption_enabled   = var.enable_encryption
    lifecycle_enabled    = var.lifecycle_enabled
    backup_retention     = var.backup_retention_days
  }
}
