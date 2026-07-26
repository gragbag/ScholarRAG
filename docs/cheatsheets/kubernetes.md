# Kubernetes — Study Cheat Sheet

Grounded in the ScholarRAG deploy (`deploy/k8s/*.yaml`, kind cluster).

## The one mental model

- You **declare desired state** (`spec`); controllers **reconcile** reality to match it.
- `spec` = what you want (you write it). `status` = what is (k8s writes it). The loop
  closing the gap is *why* k8s self-heals — kill a pod, the Deployment remakes it.
- **Cluster** → **nodes** (machines) → **pods** (1+ containers, the smallest unit).

## Every manifest = 4 fields

```yaml
apiVersion: <group/version>   # which schema (apps/v1, v1, networking.k8s.io/v1)
kind:       <ResourceType>    # Deployment, Service, ConfigMap, Ingress…
metadata:   {name, namespace, labels}
spec:       { ... }           # object-specific desired state (ConfigMap/Secret use `data`)
```

`kubectl explain <kind>.<field>...` is the built-in, field-by-field manual.

## Labels & selectors — the wiring

Nothing is connected by name-reference; everything is connected by **label-match**.

```yaml
Service.spec.selector        →  finds pods by label   →  writes their IPs into Endpoints
Deployment.spec.selector     →  which pods it manages
Deployment...template.labels →  the labels stamped on each pod
```

Selector must be a **stable subset** of the pod labels (name + component — never a
version label). **Mismatch = applies clean, silently connects nothing.**
Debug with `kubectl get endpoints <svc>` — **empty = selector matches no pods.**

## Objects you'll use

| kind | For | Key spec |
|---|---|---|
| **Deployment** | Stateless, interchangeable pods | `replicas`, `selector`, `template` |
| **StatefulSet** | Stateful (stable id + own disk) | `serviceName`, `volumeClaimTemplates` |
| **Service** | Stable in-cluster DNS + LB to pods | `selector`, `ports` (`port`→`targetPort`) |
| **ConfigMap / Secret** | Config / credentials → env via `envFrom` | `data` (Secret: base64, *not* encrypted) |
| **Ingress** | External HTTP routing → a Service | `rules`→`paths`→`backend` |
| **Namespace** | Grouping boundary | — |
| **PVC** | A persistent disk claim | via `volumeClaimTemplates` |

**Service types:** `ClusterIP` (in-cluster, default) · `clusterIP: None` (headless,
per-pod DNS for StatefulSets) · `NodePort` · `LoadBalancer` (cloud LB + public IP).

## Probes (how k8s judges health)

- **liveness** fails → **restart** the container (unwedge a hung process).
- **readiness** fails → **remove from Service** (don't send traffic; no restart).
- **startup** → hold the others off until a slow app boots.
- Check via `httpGet` (a URL) · `exec` (a command) · `tcpSocket` (port open?).
- Knobs: `initialDelaySeconds`, `periodSeconds`, `failureThreshold`.

## kubectl commands

```bash
kubectl get pods,svc,ingress -n NS          # what's here (add -o wide / --show-labels)
kubectl describe pod <name> -n NS           # DETAILS + Events (read Events first!)
kubectl logs <name> -n NS [-f] [--previous] # app stdout; --previous = last crash
kubectl apply -f dir/       /  delete -f dir/
kubectl exec -it <pod> -n NS -- sh          # shell inside a container
kubectl get endpoints <svc> -n NS           # the resolved Service→pod wiring
kubectl rollout status/undo deploy/<x> -n NS
kubectl port-forward svc/<x> 8080:80 -n NS  # reach a Service without an ingress
kubectl wait --for=condition=Ready pod --all -n NS --timeout=180s
kubectl explain deployment.spec.template.spec.containers   # field docs
```

## Debugging: STATUS → cause

| STATUS / READY | Meaning | Usual cause |
|---|---|---|
| `Pending` | Can't schedule | No resources / unbound PVC |
| `ContainerCreating` | Pulling / mounting | Normal briefly; stuck → `describe` |
| `ImagePullBackOff` | Can't get image | Wrong name / not `kind load`ed / registry `:latest` pull |
| `CreateContainerConfigError` | Missing ConfigMap/**Secret** | Referenced object not created yet |
| `CrashLoopBackOff` | Starts then dies | App error → `logs --previous` |
| `Running` but `0/1` | Up, not ready | readiness probe failing → `describe` |

**Reflex:** `get pods` → `describe` (read Events) → `logs`. That solves ~90%.

## kind (local cluster) + gotchas

```bash
kind create cluster --config deploy/kind/cluster.yaml   # a real k8s in Docker
kind load docker-image name:tag --name scholarrag       # make a local image usable
kind delete cluster --name scholarrag
```

- **`imagePullPolicy: IfNotPresent`** for `kind load`ed images (else k8s pulls `:latest`
  from a registry → `ImagePullBackOff`).
- **Secret must exist** before the pod, or `CreateContainerConfigError` (self-heals once created).
- **Service DNS**: reach `postgres`/`redis`/`api` by service name + container port, namespace-local.
