# Development Environment Main Configuration
terraform {
  required_version = ">= 1.0"
}

# Use the root module configurations
module "networking" {
  source = "../../modules/networking"
  
  project_name = var.project_name
  environment  = var.environment
}

module "storage" {
  source = "../../modules/storage"
  
  project_name = var.project_name
  environment  = var.environment
  bucket_prefix = var.storage_bucket_prefix
}

module "database" {
  source = "../../modules/database"
  
  project_name = var.project_name
  environment  = var.environment
  
  # Dependencies from networking module
  vpc_id              = module.networking.vpc_id
  private_subnet_ids  = module.networking.private_subnet_ids
  
  # Database configuration
  instance_class = var.db_instance_class
  database_name  = var.db_name
}

module "compute" {
  source = "../../modules/compute"
  
  project_name = var.project_name
  environment  = var.environment
  
  # Dependencies from other modules
  vpc_id              = module.networking.vpc_id
  public_subnet_ids   = module.networking.public_subnet_ids
  private_subnet_ids  = module.networking.private_subnet_ids
  database_endpoint   = module.database.endpoint
  storage_buckets     = module.storage.bucket_names
  
  # Compute configuration
  instance_type = var.instance_type
  min_size     = var.min_size
  max_size     = var.max_size
}
