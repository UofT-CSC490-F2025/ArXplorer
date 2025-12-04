# Outputs for ArXplorer infrastructure

output "vllm_public_ip" {
  description = "Public IP address of vLLM instance"
  value       = var.enable_vllm ? (var.use_elastic_ip ? aws_eip.vllm[0].public_ip : aws_instance.vllm[0].public_ip) : "vLLM disabled - set enable_vllm=true after GPU quota approval"
}

output "vllm_private_ip" {
  description = "Private IP address of vLLM instance"
  value       = var.enable_vllm ? aws_instance.vllm[0].private_ip : "vLLM disabled"
}

output "vllm_endpoint" {
  description = "vLLM API endpoint URL"
  value       = var.enable_vllm ? "http://${var.use_elastic_ip ? aws_eip.vllm[0].public_ip : aws_instance.vllm[0].public_ip}:${var.vllm_port}/v1" : "vLLM disabled - set enable_vllm=true after GPU quota approval"
}

output "milvus_public_ip" {
  description = "Public IP address of Milvus instance"
  value       = var.use_elastic_ip ? aws_eip.milvus[0].public_ip : aws_instance.milvus.public_ip
}

output "milvus_private_ip" {
  description = "Private IP address of Milvus instance"
  value       = aws_instance.milvus.private_ip
}

output "milvus_endpoint" {
  description = "Milvus gRPC endpoint"
  value       = "${var.use_elastic_ip ? aws_eip.milvus[0].public_ip : aws_instance.milvus.public_ip}:19530"
}

output "s3_backup_bucket" {
  description = "S3 bucket name for Milvus backups"
  value       = aws_s3_bucket.backups.id
}

output "s3_backup_bucket_arn" {
  description = "ARN of S3 backup bucket"
  value       = aws_s3_bucket.backups.arn
}

output "milvus_ebs_volume_id" {
  description = "EBS volume ID for Milvus data"
  value       = aws_ebs_volume.milvus_data.id
}

output "ssh_command_vllm" {
  description = "SSH command to connect to vLLM instance"
  value       = var.enable_vllm ? "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${var.use_elastic_ip ? aws_eip.vllm[0].public_ip : aws_instance.vllm[0].public_ip}" : "vLLM disabled"
}

output "ssh_command_milvus" {
  description = "SSH command to connect to Milvus instance"
  value       = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${var.use_elastic_ip ? aws_eip.milvus[0].public_ip : aws_instance.milvus.public_ip}"
}

output "query_api_public_ip" {
  description = "Public IP address of Query API instance"
  value       = var.enable_query_api ? (var.use_elastic_ip ? aws_eip.query_api[0].public_ip : aws_instance.query_api[0].public_ip) : "Query API disabled"
}

output "query_api_private_ip" {
  description = "Private IP address of Query API instance"
  value       = var.enable_query_api ? aws_instance.query_api[0].private_ip : "Query API disabled"
}

output "query_api_endpoint" {
  description = "Query API endpoint URL"
  value       = var.enable_query_api ? "http://${var.use_elastic_ip ? aws_eip.query_api[0].public_ip : aws_instance.query_api[0].public_ip}:${var.query_api_port}" : "Query API disabled"
}

output "ssh_command_query_api" {
  description = "SSH command to connect to Query API instance"
  value       = var.enable_query_api ? "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${var.use_elastic_ip ? aws_eip.query_api[0].public_ip : aws_instance.query_api[0].public_ip}" : "Query API disabled"
}

output "config_yaml_snippet" {
  description = "Configuration snippet for config.yaml"
  value       = var.enable_vllm ? join("", [
    "# Update your config.yaml with these values:\n\n",
    "query_rewriting:\n",
    "  enabled: true\n",
    "  use_vllm: true\n",
    "  vllm_endpoint: http://", var.use_elastic_ip ? aws_eip.vllm[0].public_ip : aws_instance.vllm[0].public_ip, ":", var.vllm_port, "/v1\n\n",
    "milvus:\n",
    "  host: ", var.use_elastic_ip ? aws_eip.milvus[0].public_ip : aws_instance.milvus.public_ip, "\n",
    "  port: 19530\n"
  ]) : join("", [
    "# Update your config.yaml with these values:\n\n",
    "query_rewriting:\n",
    "  enabled: false  # vLLM disabled - enable after GPU quota approval\n\n",
    "milvus:\n",
    "  host: ", var.use_elastic_ip ? aws_eip.milvus[0].public_ip : aws_instance.milvus.public_ip, "\n",
    "  port: 19530\n"
  ])
}

output "health_check_commands" {
  description = "Commands to verify services are running"
  value = join("", [
    var.enable_vllm ? "# Check vLLM health:\ncurl http://${var.use_elastic_ip ? aws_eip.vllm[0].public_ip : aws_instance.vllm[0].public_ip}:${var.vllm_port}/health\n\n" : "# vLLM disabled\n\n",
    var.enable_query_api ? "# Check Query API health:\ncurl http://${var.use_elastic_ip ? aws_eip.query_api[0].public_ip : aws_instance.query_api[0].public_ip}:${var.query_api_port}/health\n\n" : "# Query API disabled\n\n",
    "# Check Milvus (from local machine with pymilvus):\n",
    "python -c \"from pymilvus import connections; connections.connect(host='${var.use_elastic_ip ? aws_eip.milvus[0].public_ip : aws_instance.milvus.public_ip}', port='19530'); print('Milvus OK')\"\n"
  ])
}

output "milvus_sg_id" {
  description = "Security group ID for Milvus instance"
  value       = aws_security_group.milvus.id
}

output "query_api_sg_id" {
  description = "Security group ID for Query API instance"
  value       = var.enable_query_api ? aws_security_group.query_api[0].id : "Query API disabled"
}
