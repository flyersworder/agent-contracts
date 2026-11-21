#!/usr/bin/env bash
# Runs the pre-registered swap intervention (rebuttal_swap.py). VPS, no LLM
# call. Every arm is scored here under both samplings, so nothing is compared
# against a stored number. Resumable: a job whose output exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${W:-3}
uv run python -m evaluation.chamber_pipeline.rebuttal_swap build
for R in 1500 300; do
  for S in uniform equal; do
    out="rebuttal-swap-$S-rows$R"
    if [ -f "runs/$out.parquet" ]; then echo "skip $out"; continue; fi
    echo "start $out $(date -u +%H:%M:%S)"
    uv run python -m evaluation.chamber_pipeline.rescore runs/rebuttal-swap-src.parquet \
      --row-sampling "$S" --pc-max-rows "$R" --max-workers "$W" \
      --out "runs/$out.parquet" > "runs/$out.log" 2>&1
    echo "done  $out $(date -u +%H:%M:%S)"
  done
done
uv run python -m evaluation.chamber_pipeline.rebuttal_swap analyze
echo ALLDONE
