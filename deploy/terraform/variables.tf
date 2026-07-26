# Input variables — the knobs, like a Helm values.yaml but for infrastructure.
# Override with -var, a *.tfvars file, or TF_VAR_<name> env vars.

variable "cluster_name" {
  type    = string
  default = "scholarrag"
}

variable "namespace" {
  type    = string
  default = "scholarrag"
}

variable "chart_path" {
  type    = string
  default = "../helm/scholarrag" # relative to this terraform/ dir
}

variable "image" {
  type    = string
  default = "scholarrag-api:latest"
}

variable "gemini_api_key" {
  type      = string
  sensitive = true # Terraform masks it in plan/apply output
  # No default — you MUST supply it. Easiest:
  #   export TF_VAR_gemini_api_key="$(grep -E '^GEMINI_API_KEY=' ../../.env | cut -d= -f2-)"
  # (The `make tf-apply` target wires this for you.)
}

variable "pinecone_api_key" {
  type      = string
  sensitive = true
  # The persistent vector store's key (dense retrieval). `make tf-apply` wires it
  # from .env, same as the Gemini key.
}

variable "api_replicas" {
  type    = number
  default = 1
}
