{{- define "tradingagents.fullname" -}}
{{- printf "%s" .Release.Name | trunc 52 | trimSuffix "-" -}}
{{- end -}}

{{- define "tradingagents.claimName" -}}
{{- default (printf "%s-data" (include "tradingagents.fullname" .)) .Values.persistence.existingClaim -}}
{{- end -}}
