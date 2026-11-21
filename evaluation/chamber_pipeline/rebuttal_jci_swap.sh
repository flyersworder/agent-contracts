#!/usr/bin/env bash
# Runs the pre-registered strength swap (rebuttal_jci_swap.py). Mac, the
# backend of the paper's 5000-row table; both arms are scored here, so nothing
# is compared with a stored number. No LLM call. Resumable.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${W:-8}
uv run python -m evaluation.chamber_pipeline.rebuttal_jci_swap build
job() {  # job <out-stem> <rescore args...>
  local out=$1; shift
  if [ -f "runs/$out.parquet" ]; then echo "skip $out"; return; fi
  echo "start $out $(date +%H:%M:%S)"
  uv run python -m evaluation.chamber_pipeline.rescore runs/rebuttal-jci-swap-src.parquet \
    --max-workers "$W" --pc-max-rows 5000 --out "runs/$out.parquet" "$@" > "runs/$out.log" 2>&1
  echo "done  $out $(date +%H:%M:%S)"
}
job rebuttal-jci-swap-jci5000 --estimator jci_pc --context regime
job rebuttal-jci-swap-pc5000 --estimator pc
uv run python -m evaluation.chamber_pipeline.rebuttal_jci_swap analyze
echo ALLDONE
