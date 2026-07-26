{{/*
_helpers.tpl — named template "functions" shared across the chart.

Files starting with `_` render NO Kubernetes objects; they only DEFINE reusable
snippets. Other templates pull them in with `include "name" <context>`. This is
Helm's answer to copy-paste: define the common labels once, use them everywhere.

Two Go-template mechanics to notice:
  - `{{- define "x" -}}` … `{{- end -}}` declares a named template.
  - The `{{-` / `-}}` dashes TRIM whitespace/newlines on that side — essential
    for keeping rendered YAML correctly indented (YAML is whitespace-sensitive).
*/}}

{{/*
scholarrag.labels — the common labels stamped on EVERY object in the chart.
Callers do:  {{- include "scholarrag.labels" . | nindent 4 }}
(`.` passes the whole context so we can read .Chart / .Release; `nindent 4`
re-indents the multi-line output to sit under a `labels:` key.)
────────────────────────────────────────────────────────────────────────────────
*/}}
{{- define "scholarrag.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
