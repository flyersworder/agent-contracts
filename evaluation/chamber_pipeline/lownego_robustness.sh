#!/usr/bin/env bash
# Robustness columns of the paper, re-scored on the all-low purchase lists
# (runs/lownego-{lt,wt}.parquet, docs/superpowers/specs/2026-09-29-low-negotiation-prereg.md).
# Settings match the corpus columns they replace:
#   GES        1500 rows, 3 seeds, LT k=30 + WT k=21   (rescored-vps-ges-rows1500-subset)
#   JCI-PC     1500 rows, 9 seeds, per-variable context (rescored-vps-jci-rows1500)
#   UT-IGSP    1000 rows per sample, alpha 1e-4, 3 seeds, LT only (rescored-vps-utigsp-lt)
#   PC + exo   300 and 1500 rows, 9 seeds, every budget (paper PR #3, vps-exo-pass)
# Run on the VPS (OpenBLAS, the backend of the all-low PC scores). Resumable:
# a job whose output exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${W:-4}

uv run python - <<'EOF'
import pandas as pd
lt = pd.read_parquet("runs/lownego-lt.parquet")
wt = pd.read_parquet("runs/lownego-wt.parquet")
lt[lt.budget_k == 30].to_parquet("runs/lownego-lt30.parquet")
wt[wt.budget_k == 21].to_parquet("runs/lownego-wt21.parquet")
EOF

job() {  # job <out-stem> <source> <rescore args...>
  local out=$1 src=$2; shift 2
  if [ -f "runs/$out.parquet" ]; then echo "skip $out"; return; fi
  echo "start $out $(date -u +%H:%M:%S)"
  uv run python -m evaluation.chamber_pipeline.rescore "runs/$src.parquet" \
    --max-workers "$W" --out "runs/$out.parquet" "$@" > "runs/$out.log" 2>&1
  echo "done  $out $(date -u +%H:%M:%S)"
}

job lownego-lt30-utigsp lownego-lt30 --estimator utigsp --pc-max-rows 1000 --pc-alpha 0.0001 --n-pc-seeds 3
job lownego-lt30-ges1500 lownego-lt30 --estimator ges --pc-max-rows 1500 --n-pc-seeds 3
job lownego-wt21-ges1500 lownego-wt21 --estimator ges --pc-max-rows 1500 --n-pc-seeds 3
job lownego-lt30-jci1500 lownego-lt30 --estimator jci_pc --context variable --pc-max-rows 1500
job lownego-wt21-jci1500 lownego-wt21 --estimator jci_pc --context variable --pc-max-rows 1500
for R in 300 1500; do
  job "lownego-lt-exo$R" lownego-lt --estimator pc_exo --pc-max-rows "$R"
  job "lownego-wt-exo$R" lownego-wt --estimator pc_exo --pc-max-rows "$R"
done
echo ALLDONE
