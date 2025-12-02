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

module "network" {
  source                   = "./modules/network"
  project_name             = var.project_name
  vpc_cidr                 = var.vpc_cidr
  public_subnet_cidr       = var.public_subnet_cidr
  public_subnet_cidr_b     = var.public_subnet_cidr_b
  private_app_subnet_cidr  = var.private_app_subnet_cidr
  private_ml_subnet_cidr   = var.private_ml_subnet_cidr
  private_data_subnet_cidr = var.private_data_subnet_cidr
}

module "app" {
  source                     = "./modules/app"
  project_name               = var.project_name
  vpc_id                     = module.network.vpc_id
  public_subnet_ids          = module.network.public_subnet_ids
  app_subnet_ids             = module.network.app_subnet_ids
  allowed_ssh_cidr           = var.allowed_ssh_cidr
  search_api_port            = var.search_api_port
  search_api_health_path     = var.search_api_health_path
  search_api_instance_type   = var.search_api_instance_type
  search_api_min_size        = var.search_api_min_size
  search_api_max_size        = var.search_api_max_size
  search_api_desired_capacity= var.search_api_desired_capacity
  search_api_image           = var.search_api_image
  key_name                   = var.key_name
}

module "data" {
  source                = "./modules/data"
  project_name          = var.project_name
  environment           = var.environment
  aws_region            = var.aws_region
  vpc_id                = module.network.vpc_id
  data_subnet_ids       = module.network.data_subnet_ids
  allowed_ssh_cidr      = var.allowed_ssh_cidr
  use_elastic_ip        = var.use_elastic_ip
  key_name              = var.key_name
  milvus_instance_type  = var.milvus_instance_type
  milvus_version        = var.milvus_version
  milvus_ebs_size       = var.milvus_ebs_size
  backup_retention_days = var.backup_retention_days
  app_sg_id             = module.app.search_api_sg_id
  offline_sg_id         = module.app.offline_sg_id
}

module "offline" {
  source               = "./modules/offline"
  project_name         = var.project_name
  environment          = var.environment
  aws_region           = var.aws_region
  app_subnet_ids       = module.network.app_subnet_ids
  offline_sg_id        = module.app.offline_sg_id
  data_bucket_arns     = module.data.data_bucket_arns
  offline_task_image   = var.offline_task_image
  offline_task_cpu     = var.offline_task_cpu
  offline_task_memory  = var.offline_task_memory
  offline_desired_count= var.offline_desired_count
}
