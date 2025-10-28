# Development Environment Variables
# Only variables currently being used by active modules

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "arxplorer"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ca-central-1"
}

# Storage variables (currently active)
variable "storage_bucket_prefix" {
  description = "Prefix for S3 bucket names"
  type        = string
  default     = "arxplorer-dev"
}

# TODO: Add database variables when database module is implemented
# TODO: Add compute variables when compute module is implemented