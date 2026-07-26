# Helm — Study Cheat Sheet

Grounded in the ScholarRAG chart (`deploy/helm/scholarrag/`).

## The one mental model

Helm is the **package manager for Kubernetes**. It does NOT run anything — it
**renders** templates + values into plain k8s YAML, then hands that to Kubernetes.

```
templates/*.yaml ─┐
                  ├─►  Go-template engine  ─►  plain k8s YAML  ─►  kubectl apply
values.yaml      ─┘        (fills holes)         (the objects)     (as a named RELEASE)
```

- **Chart** = the package. **Release** = one install of a chart (named, versioned).
- Templates = *structure* (stable). Values = *configuration* (varies per env).
- `helm template` runs just the render half (no cluster) — the best learning tool.

## Chart layout

```
Chart.yaml     metadata: name, version (chart), appVersion (your app)
values.yaml    ALL the configurable knobs (defaults)
templates/
  _helpers.tpl  reusable named templates (the "_" = renders NO objects)
  *.yaml        manifests with {{ .Values.* }} holes
  NOTES.txt     printed after install (creates no object)
```

## The three context objects (what `{{ . }}` reaches)

| `{{ .Values.x }}` | from `values.yaml` (+ `--set` / `-f`) |
| `{{ .Release.Name }}` `.Namespace` `.Service` | from the `helm install` call |
| `{{ .Chart.Name }}` `.Version` | from `Chart.yaml` |

## Template language

```gotemplate
{{ .Values.api.replicas }}                       # substitute a value
{{ .Values.logJson | quote }}                     # pipe through a function
{{- toYaml .Values.api.resources | nindent 12 }}  # render a sub-tree, re-indent
{{ .Values.x | default "fallback" }}              # default if unset
{{- include "scholarrag.labels" . | nindent 4 }}  # pull in a named template
{{- if .Values.ingress.enabled }} ... {{- end }}  # conditional
{{- range .Values.items }} {{ . }} {{- end }}     # loop
{{/* a comment that is NOT rendered/executed */}}
```

- **Whitespace:** `{{-` / `-}}` trim adjacent whitespace/newlines. Essential — YAML
  is indentation-sensitive, and a stray blank line breaks it.
- **`toYaml x | nindent N`**: render a values map as YAML, indented to fit under a key.
- **`_helpers.tpl`**: `{{- define "name" -}}…{{- end -}}`, used via `include "name" .`
  (pass `.` so the helper can read `.Chart`/`.Release`). DRY: edit once, applies everywhere.

## Commands

```bash
helm lint CHART                      # static checks
helm template REL CHART [--set k=v]  # render to stdout (no cluster) — VERIFY here
helm install REL CHART -n NS --create-namespace
helm upgrade --install REL CHART -n NS --wait   # idempotent install-or-update
helm list -n NS                      # releases
helm rollback REL [REV] -n NS        # revert to a previous release revision
helm uninstall REL -n NS             # remove everything the release created
helm get manifest REL -n NS          # what's actually deployed
```

Override values: `--set api.replicas=3` (inline) or `-f prod-values.yaml` (a file).

## Gotchas we hit

- **Helm templates the ENTIRE file, including `#` comments.** Any `{{ }}` in a comment
  gets *executed* → parse errors. Put template syntax only in real YAML, or use
  `{{/* ... */}}` (Helm's own comment, never executed).
- **`{{- if }}` / `{{- end }}` don't change indentation** — the content between them
  stays at its original column; the directive vanishes at render.
- **Image string = two lookups**: `"{{ .Values.image.repository }}:{{ .Values.image.tag }}"`
  (not one path; the `:` is literal; quote it because the value contains a colon).
- **Selector labels stay minimal + stable** (name + component) — never version labels.
- **Secrets never go in `values.yaml`** (it's committed) — create them out-of-band.
- **Label counts double on workloads**: the labels helper is included in both the
  object `metadata` AND the pod `template.metadata`, so N objects with M pod-templated
  ones → N+M label blocks.

## Why Helm over raw manifests

One `values.yaml` reconfigures everything (dev vs prod), one command installs/upgrades,
`rollback` undoes a bad release, and shared bits (`_helpers.tpl`) stay DRY.
