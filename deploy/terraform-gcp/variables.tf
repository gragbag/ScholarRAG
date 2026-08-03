# Inputs to the stack. Non-secret ones can have defaults; secrets come from
# terraform.tfvars (gitignored) so they never land in the repo.

# ── GCP ──────────────────────────────────────────────────────────────────────
variable "gcp_project" {
  type        = string
  description = "GCP project id (create it in the console first)."
}

variable "gcp_region" {
  type    = string
  default = "us-central1"
}

variable "image" {
  type        = string
  description = "Full Cloud Run image ref, e.g. us-central1-docker.pkg.dev/<project>/scholarrag/api:latest"
}

# ── Public domain (Cloudflare) ───────────────────────────────────────────────
# You pick this up front, so INTERNAL_INGEST_URL can be set without a chicken-and-egg
# on the Cloud Run URL. e.g. api.yourname.dev
variable "api_domain" {
  type = string
}

variable "cloudflare_zone_id" {
  type = string
}

variable "cloudflare_api_token" {
  type      = string
  sensitive = true
}

# ── App secrets (-> Secret Manager -> Cloud Run env) ─────────────────────────
variable "postgres_dsn" {
  type        = string
  sensitive   = true
  description = "Neon POOLED connection string (create the Neon project in its console)."
}

variable "openai_api_key" {
  type      = string
  sensitive = true # your Groq key (OpenAI-compatible)
}

variable "modal_embed_url" {
  type = string
}

variable "modal_embed_token" {
  type      = string
  sensitive = true
}

variable "internal_secret" {
  type      = string
  sensitive = true # shared secret between Cloud Tasks -> /internal/ingest
}

variable "jwt_secret" {
  type      = string
  sensitive = true
}

variable "google_client_id" {
  type = string
}
