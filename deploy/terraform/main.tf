# The resource graph. Terraform reads the dependencies (explicit `depends_on` and
# implicit references like kubernetes_namespace...name) and orders everything for
# you — cluster first, app last.

# ── 1. The cluster ────────────────────────────────────────────────────────────
# Same shape as deploy/kind/cluster.yaml, expressed as HCL. This is the one
# resource you'd swap for aws_eks_cluster to go to real cloud.
resource "kind_cluster" "default" {
  name = var.cluster_name
  # Pin the node image to match the LOCAL `kind` CLI (v0.24.0 → v1.31.0). The
  # tehcyx/kind provider bundles a newer kind that would otherwise create a
  # v1.35.0 node whose containerd the older CLI can't `kind load` into
  # ("failed to detect containerd snapshotter"). Keep create + load on one version.
  node_image     = "kindest/node:v1.31.0"
  wait_for_ready = true

  kind_config {
    kind        = "Cluster"
    api_version = "kind.x-k8s.io/v1alpha4"

    node {
      role = "control-plane"
      kubeadm_config_patches = [
        "kind: InitConfiguration\nnodeRegistration:\n  kubeletExtraArgs:\n    node-labels: \"ingress-ready=true\"\n"
      ]
      extra_port_mappings {
        container_port = 80
        host_port      = 8080
      }
      extra_port_mappings {
        container_port = 443
        host_port      = 8443
      }
    }
  }
}

# ── 2. Load the local image into the cluster ──────────────────────────────────
# Terraform can't `kind load` natively, so we shell out. `triggers` re-runs this
# if the image ref changes. This is the local-dev bridge; on EKS you'd push to a
# registry (ECR) instead and drop this resource.
resource "null_resource" "load_image" {
  depends_on = [kind_cluster.default]
  triggers = {
    image      = var.image
    cluster_id = kind_cluster.default.id # re-load if the cluster is recreated
  }
  provisioner "local-exec" {
    command = "kind load docker-image ${var.image} --name ${var.cluster_name}"
  }
}

# ── 3. NGINX ingress controller ───────────────────────────────────────────────
resource "null_resource" "ingress" {
  depends_on = [kind_cluster.default]
  triggers = {
    cluster_id = kind_cluster.default.id # reinstall if the cluster is recreated
  }
  provisioner "local-exec" {
    command = <<-EOT
      kubectl --context kind-${var.cluster_name} apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
      kubectl --context kind-${var.cluster_name} -n ingress-nginx wait --for=condition=Ready pod -l app.kubernetes.io/component=controller --timeout=180s
    EOT
  }
}

# ── 4. Namespace (worked example of a kubernetes_* resource) ───────────────────
resource "kubernetes_namespace_v1" "scholarrag" {
  metadata {
    name = var.namespace
  }
}

# ── 5. The Gemini secret, from the sensitive variable (worked example) ─────────
resource "kubernetes_secret_v1" "gemini" {
  metadata {
    name      = "scholarrag-secrets"
    namespace = kubernetes_namespace_v1.scholarrag.metadata[0].name
  }
  type = "Opaque"
  data = {
    GEMINI_API_KEY   = var.gemini_api_key
    PINECONE_API_KEY = var.pinecone_api_key
  }
}

resource "helm_release" "scholarrag" {
  name       = "scholarrag"
  namespace  = kubernetes_namespace_v1.scholarrag.metadata[0].name
  chart      = var.chart_path
  wait       = true
  timeout    = 300
  depends_on = [null_resource.load_image, null_resource.ingress, kubernetes_secret_v1.gemini]

  set {
    name  = "api.replicas"
    value = var.api_replicas
  }
}
