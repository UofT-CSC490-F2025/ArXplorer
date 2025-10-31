variable "project_name" {
  description = "Short name of the project used for tagging"
  type        = string
}

variable "environment" {
  description = "Environment identifier (dev, staging, prod)"
  type        = string
}

variable "storage_bucket_prefix" {
  description = "Prefix used by the storage module when naming S3 buckets"
  type        = string
}

variable "db_instance_class" {
  description = "Instance class for the database module (e.g., db.t3.micro)"
  type        = string
}

variable "db_name" {
  description = "Logical database name to create in the database module"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type used by the compute module"
  type        = string
}

variable "min_size" {
  description = "Minimum number of compute instances in the autoscaling group"
  type        = number
}

variable "max_size" {
  description = "Maximum number of compute instances in the autoscaling group"
  type        = number
}

variable "vpc_cidr" {
  description = "CIDR block assigned to the networking module VPC"
  type        = string
}

variable "availability_zones" {
  description = "List of availability zones used for networking and storage"
  type        = list(string)
}
