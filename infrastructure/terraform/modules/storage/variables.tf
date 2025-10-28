# Storage Module Input Variables
# These define what information your storage module needs to work

# Basic Project Information
variable "project_name" {
  description = "Name of the project (e.g., 'arxplorer')"
  type        = string
  
  validation {
    condition     = length(var.project_name) > 0
    error_message = "Project name cannot be empty."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# S3 Bucket Configuration
variable "bucket_prefix" {
  description = "Prefix for S3 bucket names (e.g., 'arxplorer-dev')"
  type        = string
  default     = "arxplorer"
}

# Data Management Settings
variable "enable_versioning" {
  description = "Enable S3 bucket versioning for data protection"
  type        = bool
  default     = true
}

variable "lifecycle_enabled" {
  description = "Enable automatic lifecycle management to save costs"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backup files"
  type        = number
  default     = 120
  
  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 365
    error_message = "Backup retention must be between 1 and 365 days."
  }
}

# Security Settings
variable "enable_encryption" {
  description = "Enable server-side encryption for S3 buckets"
  type        = bool
  default     = true
}

variable "block_public_access" {
  description = "Block all public access to S3 buckets (security best practice)"
  type        = bool
  default     = true
}
