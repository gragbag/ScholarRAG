# How each provider authenticates.
#
# google: run `gcloud auth application-default login` once — Terraform picks up
#   those Application Default Credentials automatically (no key file needed).
# cloudflare: an API token (scoped to DNS edit for your zone) via a variable.

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}
