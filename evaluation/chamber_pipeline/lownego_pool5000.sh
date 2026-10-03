#!/usr/bin/env bash
# The paper's 5000-row pooling table on the all-low purchase lists: loop, team
# and varsplit at LT k=6/30/45 under plain PC and JCI-PC (one indicator per
# regime), 9 seeds. Run on the Mac (Accelerate), the machine of the table's
# rule and random rows (runs/rescored-t4-{k30,ends}-{pc5000,jcireg5000}),
# because PC at 5000 rows takes ~190 s per design-seed on the 4-core VPS.
# Input: the VPS sweep's lownego-lt.parquet, copied to runs-vps/lownego/.
# Resumable: a job whose output exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${W:-8}
SRC=runs-vps/lownego/lownego-lt.parquet

job() {  # job <out-stem> <rescore args...>
  local out=$1; shift
  if [ -f "runs/$out.parquet" ]; then echo "skip $out"; return; fi
  echo "start $out $(date +%H:%M:%S)"
  uv run python -m evaluation.chamber_pipeline.rescore "$SRC" --max-workers "$W" \
    --pc-max-rows 5000 --out "runs/$out.parquet" "$@" > "runs/$out.log" 2>&1
  echo "done  $out $(date +%H:%M:%S)"
}

job lownego-lt-jcireg5000 --estimator jci_pc --context regime
job lownego-lt-pc5000 --estimator pc
echo ALLDONE
