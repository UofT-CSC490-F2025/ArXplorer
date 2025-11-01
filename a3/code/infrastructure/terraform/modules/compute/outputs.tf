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
