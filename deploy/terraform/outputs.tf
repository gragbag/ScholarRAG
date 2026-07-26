# Outputs — values Terraform prints after apply and stores in state for reference
# (`terraform output <name>`). Handy for "what did I just create / how do I reach it".

output "cluster_name" {
  description = "The kind cluster Terraform manages."
  value       = kind_cluster.default.name
}

output "kubectl_context" {
  description = "kubectl context for this cluster."
  value       = "kind-${kind_cluster.default.name}"
}

output "ui_url" {
  value = "http://localhost:8080"
}
