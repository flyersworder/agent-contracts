#!/usr/bin/env bash
# The paper's 5000-row pooling table on the all-low purchase lists: loop, team
# and varsplit at LT k=6/30/45 under plain PC and JCI-PC (one indicator per
# regime), 9 seeds. Run on the Mac (Accelerate), the machine of the table's
# rule and random rows (runs/rescored-t4-{k30,ends}-{pc5000,jcireg5000}),
# because PC at 5000 rows takes ~190 s per design-seed on the 4-core VPS.
# Input: the VPS sweep's lownego-lt.parquet, copied to runs-vps/lownego/.
# --allow-backend-mismatch: the source cells were scored on the VPS; every
# number in the table is a Mac re-score, never the source's own F1.
# Resumable: a job whose output exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${W:-8}
SRC=runs-vps/lownego/lownego-lt.parquet

job() {  # job <out-stem> <rescore args...>; later --pc-max-rows wins
  local out=$1; shift
  if [ -f "runs/$out.parquet" ]; then echo "skip $out"; return; fi
  echo "start $out $(date +%H:%M:%S)"
  uv run python -m evaluation.chamber_pipeline.rescore "$SRC" --max-workers "$W" \
    --pc-max-rows 5000 --allow-backend-mismatch --out "runs/$out.parquet" "$@" > "runs/$out.log" 2>&1
  echo "done  $out $(date +%H:%M:%S)"
}

job lownego-lt-jcireg5000 --estimator jci_pc --context regime
job lownego-lt-pc5000 --estimator pc
# The text compares the k=45 loop at 1500 and 5000 rows; score both here.
uv run python - <<'EOF2'
import pandas as pd
c = pd.read_parquet("runs-vps/lownego/lownego-lt.parquet")
c[(c.budget_k == 45) & (c.agent_name == "llm_pc")].to_parquet("runs/lownego-lt45-loop.parquet")
EOF2
SRC=runs/lownego-lt45-loop.parquet job lownego-lt45-loop-pc1500mac --estimator pc --pc-max-rows 1500
# The supplement's ring-vs-rule table: the ring sweep was re-scored on the Mac,
# so the 30 LT k=30 rule lists it is compared with are re-scored here too.
uv run python - <<'EOF2'
import pandas as pd
c = pd.read_parquet("runs/m7-coverage-ms.parquet")
c = c[(c.agent_name == "coverage_max_ms") & (c.budget_k == 30) & (c.status == "ok")]
c.to_parquet("runs/rule-lt30.parquet")
EOF2
for R in 300 1500; do
  SRC=runs/rule-lt30.parquet job "rescored-rule-lt30-mac-rows$R" --pc-max-rows "$R"
done
echo ALLDONE
