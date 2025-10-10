# Development Environment Main Configuration
terraform {
  required_version = ">= 1.0"
}

module "networking" {
  source = "../../modules/networking"

  project_name       = var.project_name
  environment        = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
}

# module "storage" {
#   source = "../../modules/storage"
#
#   project_name  = var.project_name
#   environment   = var.environment
#   bucket_prefix = var.storage_bucket_prefix
# }

# module "database" {
#   source = "../../modules/database"
#
#   project_name = var.project_name
#   environment  = var.environment
#
#   vpc_id             = module.networking.vpc_id
#   private_subnet_ids = module.networking.private_subnet_ids
#
#   instance_class = var.db_instance_class
#   database_name  = var.db_name
# }

module "compute" {
  source = "../../modules/compute"

  project_name = var.project_name
  environment  = var.environment

  vpc_id                = module.networking.vpc_id
  public_subnet_ids     = module.networking.public_subnet_ids
  private_subnet_ids    = module.networking.private_subnet_ids
  app_security_group_id = module.networking.app_security_group_id
  database_endpoint     = "placeholder"
  storage_buckets       = {
    raw_data  = "placeholder"
    processed = "placeholder"
  }

  instance_type = var.instance_type
  min_size      = var.min_size
  max_size      = var.max_size
}
