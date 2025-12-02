variable "project_name" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }
variable "app_subnet_ids" { type = list(string) }
variable "offline_sg_id" { type = string }
variable "data_bucket_arns" { type = map(string) }
variable "offline_task_image" { type = string }
variable "offline_task_cpu" { type = number }
variable "offline_task_memory" { type = number }
variable "offline_desired_count" { type = number }

resource "aws_cloudwatch_log_group" "offline" {
  name              = "/ecs/${var.project_name}-offline"
  retention_in_days = 30
}

resource "aws_iam_role" "task_execution" {
  name = "${var.project_name}-offline-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = { Service = "ecs-tasks.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "task" {
  name = "${var.project_name}-offline-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = { Service = "ecs-tasks.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "task_s3" {
  name = "${var.project_name}-offline-s3-policy"
  role = aws_iam_role.task.id
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
        Resource = flatten([
          values(var.data_bucket_arns),
          [for arn in values(var.data_bucket_arns) : "${arn}/*"]
        ])
      }
    ]
  })
}

resource "aws_ecs_cluster" "offline" {
  name = "${var.project_name}-offline"
}

resource "aws_ecs_task_definition" "offline" {
  family                   = "${var.project_name}-offline"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.offline_task_cpu
  memory                   = var.offline_task_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "offline-worker"
      image     = var.offline_task_image
      essential = true
      command   = []
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.offline.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "offline"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "offline" {
  name            = "${var.project_name}-offline"
  cluster         = aws_ecs_cluster.offline.id
  task_definition = aws_ecs_task_definition.offline.arn
  desired_count   = var.offline_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.app_subnet_ids
    security_groups = [var.offline_sg_id]
    assign_public_ip = false
  }
}

output "cluster_name" { value = aws_ecs_cluster.offline.name }
output "task_definition_arn" { value = aws_ecs_task_definition.offline.arn }
output "service_name" { value = aws_ecs_service.offline.name }
output "log_group_name" { value = aws_cloudwatch_log_group.offline.name }
