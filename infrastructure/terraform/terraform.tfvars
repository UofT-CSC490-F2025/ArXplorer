# Development Environment Configuration
environment = "dev"
aws_region  = "ca-central-1"

# Storage & Database Settings (Your focus)
storage_bucket_prefix = "arxplorer-dev"
db_instance_class     = "db.t3.micro"  # Free tier
db_name              = "arxplorer_dev"

# Compute Settings (Teammate's focus)
instance_type = "t3.micro"  # Free tier
min_size     = 1
max_size     = 2

# Development-specific settings
enable_deletion_protection = false
backup_retention_days     = 1
multi_az                 = false
