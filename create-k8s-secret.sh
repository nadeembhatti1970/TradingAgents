#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_file="$project_dir/.env"
namespace="tradingagents"
secret_name="tradingagents-secrets"
context=""

usage() {
  cat <<'EOF'
Create or update the Kubernetes Secret from selected credentials in .env.

Only API keys and broker credentials are copied; non-secret TRADINGAGENTS_*
settings belong in the Helm values. At least one entry must have a value.
A value of the form ${OTHER_KEY} is replaced by OTHER_KEY's value from the file.
Usage: ./create-k8s-secret.sh [--context NAME] [--env-file PATH] [--namespace NAME] [--secret-name NAME]
Defaults: .env, namespace "tradingagents", secret "tradingagents-secrets".
EOF
}

while (($#)); do
  case "$1" in
    --context)
      (($# >= 2)) || { echo "Missing value for --context" >&2; exit 2; }
      context="$2"
      shift 2
      ;;
    --env-file)
      (($# >= 2)) || { echo "Missing value for --env-file" >&2; exit 2; }
      env_file="$2"
      shift 2
      ;;
    --namespace)
      (($# >= 2)) || { echo "Missing value for --namespace" >&2; exit 2; }
      namespace="$2"
      shift 2
      ;;
    --secret-name)
      (($# >= 2)) || { echo "Missing value for --secret-name" >&2; exit 2; }
      secret_name="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$env_file" ]]; then
  echo "Environment file not found: $env_file" >&2
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required but was not found in PATH." >&2
  exit 1
fi

umask 077
secret_env="$(mktemp)"
trap 'rm -f "$secret_env"' EXIT

awk '
  BEGIN {
    count = split("OPENAI_API_KEY GOOGLE_API_KEY ANTHROPIC_API_KEY XAI_API_KEY DEEPSEEK_API_KEY DASHSCOPE_API_KEY DASHSCOPE_CN_API_KEY ZHIPU_API_KEY ZHIPU_CN_API_KEY MINIMAX_API_KEY MINIMAX_CN_API_KEY OPENROUTER_API_KEY MISTRAL_API_KEY MOONSHOT_API_KEY GROQ_API_KEY NVIDIA_API_KEY OPENAI_COMPATIBLE_API_KEY TYPESAFE_API_KEY AWS_BEARER_TOKEN_BEDROCK FRED_API_KEY MARKETAUX_API_KEY T212_API_KEY T212_API_SECRET", keys, " ")
    for (i = 1; i <= count; i++) allowed[keys[i]] = 1
  }
  {
    line = $0
    sub(/\r$/, "", line)
    if (line ~ /^[[:space:]]*($|#)/) next

    separator = index(line, "=")
    if (!separator) next
    key = substr(line, 1, separator - 1)
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
    if (!(key in allowed)) next

    if (key in seen) {
      printf "Duplicate %s entry in environment file.\n", key > "/dev/stderr"
      failed = 1
      next
    }

    value = substr(line, separator + 1)
    raw[key] = value
    if (match(value, /^\$\{[A-Z_0-9]+\}$/)) {
      ref = substr(value, 3, length(value) - 3)
      if (!(ref in raw) || raw[ref] == "") {
        printf "%s references %s, which has no earlier value.\n", key, ref > "/dev/stderr"
        failed = 1
        next
      }
      value = raw[ref]
    }
    if (value == "") next
    if (value ~ /^["\047]/) {
      printf "%s must use an unquoted KEY=value entry for Kubernetes env-file format.\n", key > "/dev/stderr"
      failed = 1
      next
    }

    print key "=" value
    seen[key] = 1
    written++
  }
  END {
    if (!failed && !written) {
      print "No supported credentials with values found in the environment file." > "/dev/stderr"
      failed = 1
    }
    if (failed) exit 1
  }
' "$env_file" > "$secret_env"

kubectl_args=()
[[ -n "$context" ]] && kubectl_args+=(--context "$context")

kubectl "${kubectl_args[@]}" create namespace "$namespace" --dry-run=client -o yaml |
  kubectl "${kubectl_args[@]}" apply -f -

kubectl "${kubectl_args[@]}" -n "$namespace" create secret generic "$secret_name" \
  --from-env-file="$secret_env" \
  --dry-run=client -o yaml |
  kubectl "${kubectl_args[@]}" apply -f -
