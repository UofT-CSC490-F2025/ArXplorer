# Teammate Setup Guide - Networking & Compute Modules

## Quick Start Checklist

### Prerequisites (Must Complete First)
- [v] AWS account created and configured
- [v] AWS CLI installed and configured (`aws configure`)
- [v] Terraform installed and working (`terraform --version`)
- [v] Copy the entire `terraform/` folder from your teammate
- [v] Verify you can run `aws sts get-caller-identity` successfully

### Your Responsibility Overview
You are building the **networking foundation** and **compute infrastructure** that will run the ArXplorer pipeline. Your teammate handles storage and database.

---

## Phase 1: Networking Module (Days 1-2)

### Step 1: Build VPC and Networking
Navigate to your networking module:
```powershell
cd terraform\modules\networking
```

### Step 2: Create variables.tf
Add this content to `modules\networking\variables.tf`:
```hcl
variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["ca-central-1a", "ca-central-1b"]
}
```

### Step 3: Create main.tf for Networking
Add this content to `modules\networking\main.tf`:
```hcl
# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-${var.environment}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-${var.environment}-igw"
  }
}

# Public Subnets (for load balancers)
resource "aws_subnet" "public" {
  count = length(var.availability_zones)
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-${var.environment}-public-${count.index + 1}"
    Type = "Public"
  }
}

# Private Subnets (for app servers and databases)
resource "aws_subnet" "private" {
  count = length(var.availability_zones)
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "${var.project_name}-${var.environment}-private-${count.index + 1}"
    Type = "Private"
  }
}

# NAT Gateway (for private subnet internet access)
resource "aws_eip" "nat" {
  domain     = "vpc"
  depends_on = [aws_internet_gateway.main]

  tags = {
    Name = "${var.project_name}-${var.environment}-nat-eip"
  }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name = "${var.project_name}-${var.environment}-nat"
  }
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-public-rt"
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-private-rt"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)
  
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# Security Groups
resource "aws_security_group" "web" {
  name_prefix = "${var.project_name}-${var.environment}-web-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-web-sg"
  }
}

resource "aws_security_group" "app" {
  name_prefix = "${var.project_name}-${var.environment}-app-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.web.id]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-app-sg"
  }
}

resource "aws_security_group" "database" {
  name_prefix = "${var.project_name}-${var.environment}-db-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 3306
    to_port         = 3306
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-db-sg"
  }
}
```

### Step 4: Create outputs.tf for Networking
Add this content to `modules\networking\outputs.tf`:
```hcl
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "web_security_group_id" {
  description = "ID of the web security group"
  value       = aws_security_group.web.id
}

output "app_security_group_id" {
  description = "ID of the app security group"
  value       = aws_security_group.app.id
}

output "database_security_group_id" {
  description = "ID of the database security group"
  value       = aws_security_group.database.id
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.main.id
}
```

### Step 5: Test Networking Module
```powershell
# Go to dev environment
cd ..\..\environments\dev

# Initialize Terraform
terraform init

# Plan the networking changes
terraform plan -target=module.networking

# Apply only networking (test in isolation)
terraform apply -target=module.networking
```

---

## Phase 2: Compute Module (Days 3-4)

### Step 6: Build Compute Infrastructure
Navigate to compute module:
```powershell
cd ..\..\modules\compute
```

### Step 7: Create variables.tf for Compute
Add this content to `modules\compute\variables.tf`:
```hcl
variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID from networking module"
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnet IDs for load balancer"
  type        = list(string)
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for app servers"
  type        = list(string)
}

variable "app_security_group_id" {
  description = "Security group ID for app servers"
  type        = string
}

variable "database_endpoint" {
  description = "Database endpoint from database module"
  type        = string
}

variable "storage_buckets" {
  description = "S3 bucket names from storage module"
  type        = map(string)
}

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
```

### Step 8: Create main.tf for Compute
Add this content to `modules\compute\main.tf`:
```hcl
# Data source for latest Amazon Linux AMI
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# IAM Role for EC2 instances
resource "aws_iam_role" "pipeline_role" {
  name = "${var.project_name}-${var.environment}-pipeline-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Policy for S3 and RDS access
resource "aws_iam_role_policy" "pipeline_policy" {
  name = "${var.project_name}-${var.environment}-pipeline-policy"
  role = aws_iam_role.pipeline_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.project_name}-${var.environment}-*",
          "arn:aws:s3:::${var.project_name}-${var.environment}-*/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "rds:DescribeDBInstances",
          "rds:DescribeDBClusters"
        ]
        Resource = "*"
      }
    ]
  })
}

# Instance Profile
resource "aws_iam_instance_profile" "pipeline_profile" {
  name = "${var.project_name}-${var.environment}-pipeline-profile"
  role = aws_iam_role.pipeline_role.name
}

# Launch Template
resource "aws_launch_template" "pipeline" {
  name_prefix   = "${var.project_name}-${var.environment}-"
  image_id      = data.aws_ami.amazon_linux.id
  instance_type = var.instance_type

  vpc_security_group_ids = [var.app_security_group_id]

  iam_instance_profile {
    name = aws_iam_instance_profile.pipeline_profile.name
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    project_name      = var.project_name
    environment       = var.environment
    database_endpoint = var.database_endpoint
    s3_bucket_raw     = lookup(var.storage_buckets, "raw_data", "")
    s3_bucket_processed = lookup(var.storage_buckets, "processed", "")
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.project_name}-${var.environment}-pipeline"
    }
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "pipeline" {
  name                = "${var.project_name}-${var.environment}-asg"
  vpc_zone_identifier = var.private_subnet_ids
  target_group_arns   = [aws_lb_target_group.pipeline.arn]
  health_check_type   = "ELB"
  min_size            = var.min_size
  max_size            = var.max_size
  desired_capacity    = var.min_size

  launch_template {
    id      = aws_launch_template.pipeline.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-${var.environment}-asg"
    propagate_at_launch = false
  }
}

# Application Load Balancer
resource "aws_lb" "pipeline" {
  name               = "${var.project_name}-${var.environment}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids

  tags = {
    Name = "${var.project_name}-${var.environment}-alb"
  }
}

# ALB Security Group
resource "aws_security_group" "alb" {
  name_prefix = "${var.project_name}-${var.environment}-alb-"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-alb-sg"
  }
}

# Target Group
resource "aws_lb_target_group" "pipeline" {
  name     = "${var.project_name}-${var.environment}-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = var.vpc_id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-tg"
  }
}

# ALB Listener
resource "aws_lb_listener" "pipeline" {
  load_balancer_arn = aws_lb.pipeline.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.pipeline.arn
  }
}
```

### Step 9: Create user_data.sh for Pipeline Setup
Create `modules\compute\user_data.sh`:
```bash
#!/bin/bash
yum update -y
yum install -y python3 python3-pip git

# Install your teammate's ArXplorer pipeline
cd /home/ec2-user
git clone https://github.com/yourusername/arxplorer.git
cd arxplorer

# Install Python dependencies
pip3 install -r requirements.txt

# Set environment variables
echo "export DB_ENDPOINT=${database_endpoint}" >> /home/ec2-user/.bashrc
echo "export S3_RAW_BUCKET=${s3_bucket_raw}" >> /home/ec2-user/.bashrc
echo "export S3_PROCESSED_BUCKET=${s3_bucket_processed}" >> /home/ec2-user/.bashrc

# Start the pipeline service (placeholder)
# You'll replace this with actual pipeline startup commands
python3 -m http.server 8000 &
```

### Step 10: Create outputs.tf for Compute
Add this content to `modules\compute\outputs.tf`:
```hcl
output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.pipeline.dns_name
}

output "load_balancer_arn" {
  description = "ARN of the load balancer"
  value       = aws_lb.pipeline.arn
}

output "autoscaling_group_name" {
  description = "Name of the auto scaling group"
  value       = aws_autoscaling_group.pipeline.name
}

output "launch_template_id" {
  description = "ID of the launch template"
  value       = aws_launch_template.pipeline.id
}
```

---

## Phase 3: Testing & Integration (Day 5)

### Step 11: Test Complete Infrastructure
```powershell
# Go to dev environment
cd ..\..\environments\dev

# Plan everything
terraform plan

# Apply in stages (safer)
terraform apply -target=module.networking
terraform apply -target=module.storage    # Your teammate's module
terraform apply -target=module.database   # Your teammate's module
terraform apply -target=module.compute
```

### Step 12: Verify Everything Works
```powershell
# Check what was created
terraform show

# Get load balancer DNS
terraform output load_balancer_dns

# Test the endpoint (replace with actual DNS)
curl http://your-load-balancer-dns.ca-central-1.elb.amazonaws.com/health
```

---

## Coordination with Your Teammate

### Daily Sync Points
1. **Morning**: "What did you complete yesterday? Any blockers?"
2. **Evening**: "What outputs do you need from my modules?"

### Critical Integration Points
- **Your storage module** must output S3 bucket names
- **Your database module** must output endpoint and security group
- **My networking module** provides VPC and subnet IDs
- **My compute module** uses all the above

### Shared Variables to Track
Create a shared document with:
```
Storage Module Outputs (teammate provides):
- storage_bucket_names.raw_data
- storage_bucket_names.processed
- database_endpoint
- database_security_group_id

Networking Module Outputs (I provide):
- vpc_id
- public_subnet_ids
- private_subnet_ids
- app_security_group_id
```

---

## Troubleshooting

### Common Issues
1. **Permission errors**: Check IAM policies
2. **Resource naming conflicts**: Use unique prefixes
3. **Security group rules**: Ensure proper ingress/egress
4. **Subnet routing**: Verify NAT gateway setup

### Debug Commands
```powershell
# Validate syntax
terraform validate

# Show plan without applying
terraform plan

# Show current state
terraform show

# Debug specific resources
terraform show aws_vpc.main
```

---

## Success Criteria

You'll know you're done when:
- [v] `terraform plan` shows no errors
- [v] VPC and subnets are created successfully
- [v] Security groups allow proper traffic flow
- [ ] Load balancer is accessible from internet
- [ ] EC2 instances can connect to your teammate's database
- [ ] Pipeline can read/write to your teammate's S3 buckets

**Good luck! You're building the foundation that makes ArXplorer run!**