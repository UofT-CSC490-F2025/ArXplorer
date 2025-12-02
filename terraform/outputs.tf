# Outputs for ArXplorer infrastructure

output "search_api_alb_dns" {
  description = "DNS name for the Search API ALB"
  value       = module.app.alb_dns
}

output "search_api_asg_name" {
  description = "Name of the Search API autoscaling group"
  value       = module.app.asg_name
}

output "s3_backup_bucket" {
  description = "S3 bucket name for Milvus backups"
  value       = module.data.s3_backup_bucket
}

output "s3_backup_bucket_arn" {
  description = "ARN of S3 backup bucket"
  value       = module.data.s3_backup_bucket_arn
}

output "milvus_ebs_volume_id" {
  description = "EBS volume ID for Milvus data"
  value       = module.data.milvus_ebs_volume_id
}

output "ssh_command_milvus" {
  description = "SSH command to connect to Milvus instance"
  value       = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${var.use_elastic_ip ? module.data.milvus_public_ip : module.data.milvus_private_ip}"
}

output "config_yaml_snippet" {
  description = "Configuration snippet for config.yaml"
  value       = join("", [
    "# Update your config.yaml with these values:\n\n",
    "query_rewriting:\n",
    "  enabled: true\n",
    "  provider: bedrock\n",
    "  model: <your-bedrock-model-id>\n",
    "  region: ", var.aws_region, "\n\n",
    "milvus:\n",
    "  host: ", var.use_elastic_ip ? module.data.milvus_public_ip : module.data.milvus_private_ip, "\n",
    "  port: 19530\n"
  ])
}

output "health_check_commands" {
  description = "Commands to verify services are running"
  value       = join("", [
    "# Check Search API via ALB:\n",
    "curl http://", module.app.alb_dns, "\n\n",
    "# Check Milvus (from a host that can reach it):\n",
    "python -c \"from pymilvus import connections; connections.connect(host='", var.use_elastic_ip ? module.data.milvus_public_ip : module.data.milvus_private_ip, "', port='19530'); print('Milvus OK')\"\n"
  ])
}

output "milvus_private_ip" {
  description = "Private IP address of Milvus instance"
  value       = module.data.milvus_private_ip
}

output "milvus_public_ip" {
  description = "Public IP address of Milvus instance (if EIP enabled)"
  value       = module.data.milvus_public_ip
}

output "milvus_endpoint" {
  description = "Milvus gRPC endpoint"
  value       = module.data.milvus_endpoint
}

output "offline_cluster_name" {
  description = "ECS cluster for offline pipeline workers"
  value       = module.offline.cluster_name
}

output "offline_task_definition" {
  description = "ECS task definition ARN for offline workers"
  value       = module.offline.task_definition_arn
}

output "offline_service_name" {
  description = "ECS service name for offline workers (desired_count can be 0)"
  value       = module.offline.service_name
}
