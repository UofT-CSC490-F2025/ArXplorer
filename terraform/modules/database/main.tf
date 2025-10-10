# Random password for database
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# AWS Secrets Manager secret for database credentials
resource "aws_secretsmanager_secret" "db_credentials" {
  name        = "${var.project_name}-${var.environment}-db-credentials"
  description = "Database credentials for ArXplorer ${var.environment}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-secret"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = var.username
    password = random_password.db_password.result
    engine   = "mysql"
    host     = aws_db_instance.main.endpoint
    port     = aws_db_instance.main.port
    dbname   = var.database_name
  })
}

# DB Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}-db-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-subnet-group"
    Environment = var.environment
  }
}

# Parameter Group for MySQL optimization
resource "aws_db_parameter_group" "main" {
  family = "mysql8.0"
  name   = "${var.project_name}-${var.environment}-db-params"

  parameter {
    name         = "innodb_buffer_pool_size"
    value        = "{DBInstanceClassMemory*3/4}"
    apply_method = "pending-reboot"
  }

  parameter {
    name  = "max_connections"
    value = "200"
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-params"
    Environment = var.environment
  }
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-${var.environment}-db"

  # Database Configuration
  engine         = "mysql"
  engine_version = "8.0"
  instance_class = var.instance_class
  db_name        = var.database_name
  username       = var.username
  password       = random_password.db_password.result

  # Storage Configuration
  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = "gp2"
  storage_encrypted     = true

  # Network Configuration
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [var.database_security_group_id]
  publicly_accessible    = false

  # High Availability
  multi_az = var.multi_az

  # Backup Configuration
  backup_retention_period = var.backup_retention_period
  backup_window           = "03:00-04:00"         # UTC
  maintenance_window      = "Sun:04:00-Sun:05:00" # UTC

  # Parameter and Option Groups
  parameter_group_name = aws_db_parameter_group.main.name

  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn

  # Security
  deletion_protection = var.deletion_protection

  # Performance Insights (for production)
  performance_insights_enabled = var.environment == "prod" ? true : false

  tags = {
    Name        = "${var.project_name}-${var.environment}-database"
    Environment = var.environment
    Purpose     = "ArXplorer paper metadata and search results"
  }

  # Prevent accidental deletion
  lifecycle {
    prevent_destroy = false # Set to true for production
  }
}

# IAM Role for Enhanced Monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "${var.project_name}-${var.environment}-rds-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# CloudWatch Log Groups for database logs
resource "aws_cloudwatch_log_group" "database_logs" {
  name              = "/aws/rds/instance/${aws_db_instance.main.identifier}/error"
  retention_in_days = 30

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-logs"
    Environment = var.environment
  }
}

# Database Schema Setup (via null resource)
resource "null_resource" "database_schema" {
  depends_on = [aws_db_instance.main]

  provisioner "local-exec" {
    command = <<-EOT
      echo "Database created: ${aws_db_instance.main.endpoint}"
      echo "Database schema will be initialized by application deployment"
      echo "Connection string stored in AWS Secrets Manager: ${aws_secretsmanager_secret.db_credentials.name}"
    EOT
  }

  triggers = {
    database_endpoint = aws_db_instance.main.endpoint
  }
}
