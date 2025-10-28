# Project Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "arxplorer"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ca-central-1"
}

# Storage Team Variables (Your focus)
variable "storage_bucket_prefix" {
  description = "Prefix for S3 bucket names"
  type        = string
  default     = "arxplorer"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "arxplorer"
}

# Compute Team Variables (Teammate's focus)
variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "min_size" {
  description = "Minimum number of instances"
  type        = number
  default     = 1
}

variable "max_size" {
  description = "Maximum number of instances"
  type        = number
  default     = 3
}