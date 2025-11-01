# Storage Module Outputs (Your responsibility)
output "storage_bucket_names" {
  description = "Names of S3 buckets created by storage module"
  value = {
    raw_data    = "arxplorer-${var.environment}-raw-data"
    processed   = "arxplorer-${var.environment}-processed"
    embeddings  = "arxplorer-${var.environment}-embeddings"
    backups     = "arxplorer-${var.environment}-backups"
  }
}

output "storage_bucket_arns" {
  description = "ARNs of S3 buckets for IAM policies"
  value = {
    raw_data    = "arn:aws:s3:::arxplorer-${var.environment}-raw-data"
    processed   = "arn:aws:s3:::arxplorer-${var.environment}-processed"
    embeddings  = "arn:aws:s3:::arxplorer-${var.environment}-embeddings"
    backups     = "arn:aws:s3:::arxplorer-${var.environment}-backups"
  }
}

# Database Module Outputs (Your responsibility)
output "database_endpoint" {
  description = "RDS database endpoint for application connections"
  value       = "arxplorer-${var.environment}-db.cluster-xyz.ca-central-1.rds.amazonaws.com"
  sensitive   = true
}

output "database_port" {
  description = "Database port number"
  value       = 3306
}

output "database_name" {
  description = "Database name"
  value       = var.db_name
}

output "database_security_group_id" {
  description = "Security group ID for database access"
  value       = "sg-database-placeholder"
}

# Networking Module Outputs (Teammate's responsibility)
output "vpc_id" {
  description = "VPC ID for resource placement"
  value       = "vpc-placeholder-id"
}

output "public_subnet_ids" {
  description = "Public subnet IDs for load balancers"
  value       = ["subnet-pub-1", "subnet-pub-2"]
}

output "private_subnet_ids" {
  description = "Private subnet IDs for application servers and databases"
  value       = ["subnet-priv-1", "subnet-priv-2"]
}

output "app_security_group_id" {
  description = "Security group ID for application servers"
  value       = "sg-app-placeholder"
}

# Compute Module Outputs (Teammate's responsibility)
output "load_balancer_dns" {
  description = "Load balancer DNS name for external access"
  value       = "arxplorer-${var.environment}-lb-123456789.ca-central-1.elb.amazonaws.com"
}

output "pipeline_server_ips" {
  description = "Private IP addresses of pipeline processing servers"
  value       = ["10.0.10.100", "10.0.20.100"]
}
