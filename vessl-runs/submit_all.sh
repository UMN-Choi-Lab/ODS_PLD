#!/usr/bin/env bash
# Render each template for every (network, seed) pair and submit to VESSL.
# Use --dry-run to print the rendered YAMLs without submitting.
#
# Usage:
#   bash vessl-runs/submit_all.sh                 # submit full sweep (4 methods x 3 networks x 3 seeds = 36 runs)
#   bash vessl-runs/submit_all.sh --dry-run       # preview only (no submission)
#   bash vessl-runs/submit_all.sh nnls            # single method: nnls | pld | pldturbo | sobolturbo

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDERED="$HERE/rendered"
mkdir -p "$RENDERED"

NETWORKS=("1ramp" "2corridor" "3junction")
SEEDS=(11 12 13)

WANDB_API_KEY="${WANDB_API_KEY:-}"
if [[ -z "$WANDB_API_KEY" && -f "$HOME/.netrc" ]]; then
  WANDB_API_KEY="$(awk '/machine api.wandb.ai/{f=1} f && $1=="password"{print $2; exit}' "$HOME/.netrc")"
fi
if [[ -z "$WANDB_API_KEY" ]]; then
  echo "ERROR: WANDB_API_KEY not found in env or ~/.netrc. Run 'wandb login' first." >&2
  exit 1
fi

DRY_RUN=0
METHOD_FILTER=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    nnls|pld|pldturbo|sobolturbo) METHOD_FILTER="$arg" ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

METHODS=("nnls" "pld" "pldturbo" "sobolturbo")

for method in "${METHODS[@]}"; do
  if [[ -n "$METHOD_FILTER" && "$method" != "$METHOD_FILTER" ]]; then
    continue
  fi
  tmpl="$HERE/${method}.yaml.template"
  for net in "${NETWORKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out="$RENDERED/${method}_${net}_seed${seed}.yaml"
      sed -e "s/{NETWORK}/${net}/g" \
          -e "s/{SEED}/${seed}/g" \
          -e "s|{WANDB_API_KEY}|${WANDB_API_KEY}|g" \
          "$tmpl" > "$out"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "=== $out ==="
        cat "$out"
      else
        echo ">>> submitting $out"
        vessl run create -f "$out"
      fi
    done
  done
done
