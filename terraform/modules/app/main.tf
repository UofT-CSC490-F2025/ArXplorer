variable "project_name" { type = string }
variable "vpc_id" { type = string }
variable "public_subnet_ids" { type = list(string) }
variable "app_subnet_ids" { type = list(string) }
variable "allowed_ssh_cidr" { type = string }
variable "search_api_port" { type = number }
variable "search_api_health_path" { type = string }
variable "search_api_instance_type" { type = string }
variable "search_api_min_size" { type = number }
variable "search_api_max_size" { type = number }
variable "search_api_desired_capacity" { type = number }
variable "search_api_image" { type = string }
variable "key_name" { type = string }

resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-sg"
  description = "ALB security group"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP from internet"
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS from internet"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = { Name = "${var.project_name}-alb-sg" }
}

resource "aws_security_group" "search_api" {
  name        = "${var.project_name}-search-api-sg"
  description = "Security group for Search API instances"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = var.search_api_port
    to_port         = var.search_api_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "Traffic from ALB"
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "SSH access"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = { Name = "${var.project_name}-search-api-sg" }
}

resource "aws_security_group" "offline_ingest" {
  name        = "${var.project_name}-offline-sg"
  description = "Security group for offline ingestion / batch workers"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "SSH access"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = { Name = "${var.project_name}-offline-sg" }
}

resource "aws_iam_role" "search_api_role" {
  name = "${var.project_name}-search-api-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = { Service = "ec2.amazonaws.com" }
      }
    ]
  })

  tags = { Name = "${var.project_name}-search-api-role" }
}

resource "aws_iam_role_policy" "search_api_bedrock" {
  name = "${var.project_name}-search-api-bedrock"
  role = aws_iam_role.search_api_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "search_api" {
  name = "${var.project_name}-search-api-profile"
  role = aws_iam_role.search_api_role.name
}

resource "aws_lb" "search_api" {
  name               = "${var.project_name}-alb"
  load_balancer_type = "application"
  internal           = false
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids
  tags               = { Name = "${var.project_name}-alb" }
}

resource "aws_lb_target_group" "search_api" {
  name        = "${var.project_name}-search-api-tg"
  port        = var.search_api_port
  protocol    = "HTTP"
  target_type = "instance"
  vpc_id      = var.vpc_id

  health_check {
    healthy_threshold   = 3
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 5
    path                = var.search_api_health_path
    matcher             = "200-399"
  }

  tags = { Name = "${var.project_name}-search-api-tg" }
}

resource "aws_lb_listener" "search_api_http" {
  load_balancer_arn = aws_lb.search_api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.search_api.arn
  }
}

resource "aws_launch_template" "search_api" {
  name_prefix   = "${var.project_name}-search-api-"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = var.search_api_instance_type
  key_name      = var.key_name

  iam_instance_profile {
    name = aws_iam_instance_profile.search_api.name
  }

  vpc_security_group_ids = [aws_security_group.search_api.id]

  user_data = base64encode(templatefile("${path.root}/user_data_search_api.sh", {
    container_image = var.search_api_image
    app_port        = var.search_api_port
  }))

  tag_specifications {
    resource_type = "instance"
    tags = { Name = "${var.project_name}-search-api" }
  }
}

resource "aws_autoscaling_group" "search_api" {
  name                      = "${var.project_name}-search-api"
  desired_capacity          = var.search_api_desired_capacity
  max_size                  = var.search_api_max_size
  min_size                  = var.search_api_min_size
  vpc_zone_identifier       = var.app_subnet_ids
  target_group_arns         = [aws_lb_target_group.search_api.arn]
  health_check_type         = "EC2"
  health_check_grace_period = 120

  launch_template {
    id      = aws_launch_template.search_api.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-search-api"
    propagate_at_launch = true
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

output "alb_dns" { value = aws_lb.search_api.dns_name }
output "asg_name" { value = aws_autoscaling_group.search_api.name }
output "alb_sg_id" { value = aws_security_group.alb.id }
output "search_api_sg_id" { value = aws_security_group.search_api.id }
output "offline_sg_id" { value = aws_security_group.offline_ingest.id }
output "iam_instance_profile_name" { value = aws_iam_instance_profile.search_api.name }
