# Terraform — Study Cheat Sheet

Grounded in the ScholarRAG IaC (`deploy/terraform/`).

## The one mental model

Terraform is **infrastructure as code**: you *declare* the resources you want in
`.tf` files, and Terraform makes the API calls to create/change/destroy them to
match. It is a *client* that drives other systems' APIs — it doesn't run anything itself.

```
.tf files  ──►  terraform plan  ──►  terraform apply  ──►  real resources
(desired)       (diff vs state)      (make it so)          + updated STATE file
```

- **Declarative + stateful:** you describe the end state; Terraform diffs it against
  a recorded **state file** (what it made last time) to compute create/change/destroy.
- **Providers = plugins** that teach Terraform an API (kind, kubernetes, helm, aws…).
- **The dependency graph is inferred**: reference `a.b.c` from another resource and
  Terraform orders them for you — no manual sequencing.

## Core building blocks

```hcl
terraform {                                   # settings + which providers
  required_providers { helm = { source = "hashicorp/helm", version = ">= 2.12" } }
}

provider "kubernetes" { host = kind_cluster.default.endpoint }   # HOW to reach a provider

resource "kind_cluster" "default" { name = "scholarrag" }        # a thing to create
#         └type────────┘ └name┘   → referenced elsewhere as kind_cluster.default.<attr>

variable "gemini_api_key" { type = string, sensitive = true }    # an input knob
output   "ui_url"         { value = "http://localhost:8080" }     # a value to surface
```

- **Reference / interpolate:** `kind_cluster.default.endpoint`, `"${var.name}-suffix"`.
- **`depends_on = [x]`** forces order when there's no direct reference.
- **`sensitive = true`** masks a value in output.
- **`data "…" "…"`** reads existing infra (vs `resource`, which manages it).

## Workflow commands

```bash
terraform init                 # download providers into .terraform/ (run first / after adding one)
terraform fmt                  # canonical formatting
terraform validate             # syntax + internal consistency (no cluster needed)
terraform plan                 # preview create/change/destroy — the safety net
terraform apply                # do it (add -auto-approve to skip the prompt)
terraform destroy              # remove everything in state (the time-box safety net)
terraform output [name]        # show outputs
terraform state list           # what Terraform currently tracks
terraform apply -target=res    # apply just one resource (+ its deps) — see gotcha below
```

`-chdir=deploy/terraform` runs from elsewhere (what our Makefile targets use).

## Providers & state

- `required_providers` (in `terraform {}`) declares **which**; `provider "x" {}` blocks say **how to connect**.
- **State** (`terraform.tfstate`) is Terraform's memory of what it made — it may
  contain **secrets** → NEVER commit it (`.gitignore` it). Real teams use *remote
  state* (S3 + locking) so a team shares one state.
- **`.terraform.lock.hcl`** pins provider versions → DO commit it (like uv.lock).

## Gotchas we hit (and the fixes)

- **Provider version skew:** the `tehcyx/kind` provider bundles its own (newer) kind
  and made a `v1.35.0` node the local CLI `kind load` (v0.24.0) couldn't read
  ("failed to detect containerd snapshotter"). **Fix:** pin `node_image =
  "kindest/node:v1.31.0"` so create + load agree. *Lesson: pin versions.*
- **Provider config that depends on a just-created resource:** the kubernetes/helm
  providers get their `host`/certs FROM `kind_cluster.default.*`, which is unknown
  until the cluster exists → a single apply fails (`dial tcp 127.0.0.1:80`, the
  provider defaulting to localhost). **Fix:** two-step —
  `apply -target=kind_cluster.default` first (make the endpoint concrete in state),
  then a full `apply`. Baked into `make tf-apply`.
- **`null_resource` + `local-exec`** = an escape hatch to run a shell command
  (`kind load`, `kubectl apply`) as a Terraform resource. Add
  `triggers = { k = ref }` so it re-runs when something it depends on changes
  (without a trigger it runs once and never again).
- **Recovering tangled local state:** it's just a local kind cluster — safe to
  `kind delete cluster` + `rm terraform.tfstate*` + re-apply from scratch.

## File layout (a convention, not a rule — Terraform reads ALL *.tf together)

| File | Holds |
|---|---|
| `versions.tf` | `terraform {}` block: `required_version`, `required_providers` |
| `providers.tf` | `provider "…" {}` connection config |
| `main.tf` | the `resource` graph (the actual infrastructure) |
| `variables.tf` | `variable` inputs (the knobs) |
| `outputs.tf` | `output` values to surface after apply |
| `terraform.tfvars` | real variable values — **gitignored** (may hold secrets) |

## Why Terraform (vs clicking a console)

Reproducible (code, reviewable, version-controlled), one command up **and** down
(`destroy` is the cost/safety lever), portable (swap the `kind_cluster` resource
for `aws_eks_cluster` → same config targets AWS), and it drives Helm/k8s too so the
whole stack — cluster + app — comes up from a single `apply`.
