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