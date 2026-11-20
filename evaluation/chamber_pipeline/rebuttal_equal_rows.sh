#!/usr/bin/env bash
# AAMAS 2027 rebuttal, reviewer xQHG: "the pooled data are sampled uniformly,
# so larger experiments contribute more rows ... would sampling an equal
# number of rows from each bought experiment change the main comparisons?"
#
# Re-scores the purchase lists behind Figure 2 and the team contrasts with
# `--row-sampling equal` (inference.pool_equal_rows), plain PC, 9 seeds, at
# both row caps. No LLM call. The uniform comparator is the existing
#   runs/lownego-{lt,wt}-rescored-rows{300,1500}   (loop, team, varsplit; all-low)
#   runs/rescored-vps-rows{300,1500}               (random, single call, the rule)
# all scored on the VPS (OpenBLAS), so run this on the VPS too. Resumable: a
# job whose output exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${W:-3}

uv run python - <<'PY'
import pandas as pd

ADDED = [
    "selection_key", "design_key", "f1_rescored", "n_pc_seeds", "f1_skeleton_rescored",
    "f1_core_rescored", "rescore_blas_backend", "rescore_platform_tag",
]
BUDGETS = {"lt": [6, 30, 45], "wt": [7, 14, 21]}
CORPUS_ARMS = ["random", "one_shot", "coverage_max_ms", "wt_coverage_max"]

low = pd.concat([pd.read_parquet(f"runs/lownego-{ch}.parquet") for ch in ("lt", "wt")])
low["orig_source_file"] = "lownego"
corpus = pd.read_parquet("runs/rescored-vps-rows300.parquet")
corpus = corpus[
    corpus["agent_name"].isin(CORPUS_ARMS)
    & corpus.apply(lambda r: r["budget_k"] in BUDGETS.get(r["chamber"], []), axis=1)
].rename(columns={"source_file": "orig_source_file"})
# One model and one selection effort per arm, or an "arm" is a mixture. (The
# rule and random make no LLM call and carry nulls.)
mixed = corpus.groupby(["chamber", "agent_name"])[["model_id", "reasoning_effort"]].nunique()
assert (mixed <= 1).all().all(), f"a corpus arm mixes models or efforts:\n{mixed}"
src = pd.concat([low, corpus], ignore_index=True)
src = src.drop(columns=[c for c in ADDED if c in src.columns])
src = src[(src["status"] == "ok") & src["chosen_experiments"].notna()]
src.to_parquet("runs/rebuttal-eqrows-src.parquet", index=False)
print(src.groupby(["chamber", "agent_name", "budget_k"]).size().unstack(fill_value=0).to_string())
PY

for R in 300 1500; do
  out="rebuttal-eqrows-rows$R"
  if [ -f "runs/$out.parquet" ]; then echo "skip $out"; continue; fi
  echo "start $out $(date -u +%H:%M:%S)"
  uv run python -m evaluation.chamber_pipeline.rescore runs/rebuttal-eqrows-src.parquet \
    --row-sampling equal --pc-max-rows "$R" --max-workers "$W" \
    --out "runs/$out.parquet" > "runs/$out.log" 2>&1
  echo "done  $out $(date -u +%H:%M:%S)"
done
echo ALLDONE
