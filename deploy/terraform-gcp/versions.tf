# Provider + version pins. `terraform init` reads this and downloads the plugins.
# (Separate from deploy/terraform/, which targets the Phase 7 local kind cluster.)
terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }

  # State lives locally for now (terraform.tfstate, gitignored). A team would move
  # this to a remote backend (a GCS bucket) so state is shared + locked.
}
