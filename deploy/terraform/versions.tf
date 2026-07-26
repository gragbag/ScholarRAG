# Provider requirements + version pins. `terraform init` reads this and downloads
# the providers into .terraform/. Pinning (like uv.lock) keeps builds reproducible.
terraform {
  required_version = ">= 1.5"

  required_providers {
    # Creates/deletes the local kind cluster. Swap this provider for
    # hashicorp/aws (aws_eks_cluster) to target real AWS — the rest is unchanged.
    kind = {
      source  = "tehcyx/kind"
      version = ">= 0.4.0"
    }
    # Installs the Helm chart (helm v2 line — uses the `kubernetes { }` block form).
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.12, < 3.0"
    }
    # Manages the namespace + secret directly.
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.20"
    }
    # Lets us run local shell (kind load, kubectl) as a Terraform resource.
    null = {
      source  = "hashicorp/null"
      version = ">= 3.0"
    }
  }
}
