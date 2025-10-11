# ArXplorer Development Environment
# CSC490 - Infrastructure as Code Implementation

terraform {
  required_version = ">= 1.0"
}

# ===== ACTIVE MODULES =====

# Storage Module - S3 buckets for ArXplorer data pipeline
module "storage" {
  source = "../../modules/storage"
  
  project_name  = var.project_name
  environment   = var.environment
  bucket_prefix = var.storage_bucket_prefix
}

# ===== FUTURE MODULES =====

# TODO: Uncomment when networking module is ready
# module "networking" {
#   source = "../../modules/networking"
#   project_name = var.project_name
#   environment  = var.environment
# }

# TODO: Uncomment when database module is ready
# module "database" {
#   source = "../../modules/database"
#   
#   project_name = var.project_name
#   environment  = var.environment
# }

# TODO: Uncomment when compute module is ready  
# module "compute" {
#   source = "../../modules/compute"
#   project_name = var.project_name
#   environment  = var.environment
# }

# ===== OUTPUTS =====
output "storage_buckets" {
  description = "S3 bucket information for ArXplorer data storage"
  value       = module.storage.bucket_names
}

output "storage_arns" {
  description = "S3 bucket ARNs for IAM policies"
  value       = module.storage.bucket_arns
}

output "storage_summary" {
  description = "Summary of storage configuration"
  value       = module.storage.storage_summary
}
