# Provider CONFIGURATION (versions.tf declares WHICH providers; this says HOW to
# reach them). Both the kubernetes and helm providers are pointed at the cluster
# that kind_cluster.default creates — Terraform reads its generated credentials.
#
# Because these values are "known only after apply" (the cluster doesn't exist at
# plan time), Terraform automatically creates the cluster first, then configures
# these providers. (If a first `apply` ever complains about an unknown provider
# config, run `terraform apply -target=kind_cluster.default` once, then apply.)

provider "kubernetes" {
  host                   = kind_cluster.default.endpoint
  client_certificate     = kind_cluster.default.client_certificate
  client_key             = kind_cluster.default.client_key
  cluster_ca_certificate = kind_cluster.default.cluster_ca_certificate
}

provider "helm" {
  kubernetes {
    host                   = kind_cluster.default.endpoint
    client_certificate     = kind_cluster.default.client_certificate
    client_key             = kind_cluster.default.client_key
    cluster_ca_certificate = kind_cluster.default.cluster_ca_certificate
  }
}
