"""Did the provider drift DURING a sweep, and could it have biased an arm?

`iter_sweep_cells` used to run every seed of one arm before starting the next
(fixed 2026-09-02), so each arm occupied its own window of wall-clock time.
That is harmless while provider behaviour holds still and fatal when it does
not: on 2026-09-02 a WT run's output tokens climbed 134k -> 190k inside two
hours under an unchanged model id, which would have made the later arm look
different for reasons that have nothing to do with the arm.

Every sweep recorded before the fix needs checking rather than assuming, and
this module is that check.

**The probe is tokens per LLM call.** Two rejected alternatives, recorded so
they are not tried again:

* raw `tokens_out` is arm-dependent by construction — a two-scout `team` cell
  emits several times a loop cell — so its trend over a sweep mostly measures
  which arm ran first (a WT run showed r=+0.71 that way, and +0.001 once
  residualised on arm x budget);
* generation throughput (`tokens_out / wall_time_seconds`) is worse, because
  it tracks OUR OWN concurrency: per-cell latency rises while many workers
  hammer one endpoint and falls as a block drains, so every block picks up a
  ramp at its edges. It flagged 10 of 11 archived files, which is the
  signature of an artefact rather than ten defects.

`tokens_out / n_llm_calls` is immune to both. The call count is fixed by the
arm and budget, so dividing it out leaves how much the MODEL chose to reason
per call — the quantity that actually shifted (2.4x) on 2026-09-02, and one
that our scheduling cannot touch.

Two statistics, because they answer different questions:

* **within-block trend** — r(launch order, tokens-per-call) inside one
  arm x budget block. Detects drift while a single arm was running.
* **window overlap** — what fraction of an arm's wall-clock window is shared
  with the other arms it is contrasted against. Where arms overlap, blocked
  ordering cannot have biased them relative to each other, whatever the
  provider did.

**Order by `started_at`, never `finished_at`.** Under `--max-workers` a slow
cell finishes later by definition, so ranking by completion time induces a
negative correlation with throughput out of nothing. Launch order is set by
the scheduler and carries no such artefact. A first version of this module
used completion order and reported |r| up to 0.80 on runs whose launch-order
trends are flat.

**Between-block spread is NOT reported, because it is not identifiable.** A
two-scout `team` cell emits several times what a single-agent loop cell does,
so comparing block-mean throughput across arms measures the arms. Under
blocked ordering each arm occupies exactly one window, which makes arm and
window perfectly confounded — that is precisely what makes the ordering a
defect, and no statistic computed after the fact can separate them. What can
be said is reported: whether each block was internally stable, and whether the
windows overlapped at all.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Flag a block when |r| exceeds this many null standard errors. Under no
#: drift, r has sd ~ 1/sqrt(n-3), so a FIXED cutoff cries wolf: a sweep has
#: 9-15 blocks and reports the largest |r| among them, whose expected value is
#: ~2 sd from noise alone. A flat 0.30 flagged 10 of 11 archived files for
#: that reason. Three sd keeps the per-sweep false-alarm rate low without
#: hiding the real ones -- `shared_blackboard` k=14 at r=-0.730 (n=50, cutoff
#: 0.438) still flags, while `one_shot` k=45 at +0.525 (n=30, cutoff 0.577)
#: correctly does not.
WITHIN_BLOCK_SIGMA_LIMIT = 3.0


def block_r_cutoff(n: int) -> float:
    """The |r| a block of `n` cells must beat to be called a trend."""
    if n <= 4:
        return float("nan")
    return WITHIN_BLOCK_SIGMA_LIMIT / math.sqrt(n - 3)


REQUIRED_COLUMNS = (
    "agent_name",
    "budget_k",
    "started_at",
    "finished_at",
    "tokens_out",
    "n_llm_calls",
)


def reasoning_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Ok LLM-cells with a `tokens_per_call` column, in launch order.

    Cells that issue no LLM call (`random`, `greedy_ig`, the coverage rules)
    carry no signal about the provider and are dropped rather than counted as
    zero — averaging them in would dilute exactly the drift being looked for.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"missing {missing}; a sweep recorded before these were captured "
            "cannot be checked for drift and must be reported as unchecked "
            "rather than as clean"
        )
    ok = frame[frame["status"] == "ok"].copy()
    ok = ok[ok["n_llm_calls"].notna() & (ok["n_llm_calls"] > 0) & ok["tokens_out"].notna()]
    ok["tokens_per_call"] = ok["tokens_out"] / ok["n_llm_calls"]
    ok["started"] = pd.to_datetime(ok["started_at"])
    ok["finished"] = pd.to_datetime(ok["finished_at"])
    return ok.sort_values("started").reset_index(drop=True)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(xs) < 3 or xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def within_block_trends(cells: pd.DataFrame) -> pd.DataFrame:
    """r(launch order, tokens-per-call) inside each arm x budget block."""
    rows = []
    for (arm, budget_k), block in cells.groupby(["agent_name", "budget_k"]):
        block = block.sort_values("started")
        rows.append(
            {
                "agent_name": arm,
                "budget_k": budget_k,
                "n": len(block),
                "tokens_per_call": block["tokens_per_call"].mean(),
                "r_within": _pearson(range(len(block)), block["tokens_per_call"]),
                "cutoff": block_r_cutoff(len(block)),
                "window_start": block["started"].min(),
                "window_end": block["finished"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("window_start").reset_index(drop=True)


def window_overlap(blocks: pd.DataFrame) -> float:
    """Fraction of the sweep's span during which more than one block was live.

    1.0 means every arm ran concurrently with the others, so blocked ordering
    is moot. 0.0 means the arms were strictly sequential and arm is perfectly
    confounded with time.

    A sweep line over start/end events, not a pairwise scan: comparing spans
    to each other by VALUE silently treats two blocks that ran in exactly the
    same window as the same block and reports zero overlap for the most
    overlapped case there is.
    """
    if len(blocks) < 2:
        return 1.0
    events: list[tuple[int, int]] = []
    for row in blocks.itertuples():
        events.append((row.window_start.value, 1))
        events.append((row.window_end.value, -1))
    events.sort()
    total = events[-1][0] - events[0][0]
    if total <= 0:
        return 1.0
    covered, active, previous = 0, 0, events[0][0]
    for moment, delta in events:
        if active >= 2:
            covered += moment - previous
        active += delta
        previous = moment
    return min(covered / total, 1.0)


def residual_trend(cells: pd.DataFrame) -> float:
    """r(launch order, tokens-per-call) after removing arm x budget means.

    Removes what arm composition explains. Because each block is a contiguous
    window under blocked ordering, this aggregates the WITHIN-block trends and
    cannot see a step change that happened between blocks.
    """
    centred = cells["tokens_per_call"] - cells.groupby(["agent_name", "budget_k"])[
        "tokens_per_call"
    ].transform("mean")
    return _pearson(range(len(cells)), centred)


def audit(frame: pd.DataFrame, *, label: str = "") -> dict[str, object]:
    """Descriptive verdict for one sweep file."""
    cells = reasoning_frame(frame)
    if cells.empty:
        return {
            "file": label,
            "n_ok": 0,
            "hours": 0.0,
            "blocks": 0,
            "r_residual": float("nan"),
            "max_abs_r_within": float("nan"),
            "window_overlap": float("nan"),
            "verdict": "N/A (no LLM cells)",
            "_blocks": pd.DataFrame(),
        }
    blocks = within_block_trends(cells)
    blocks["exceeds"] = blocks["r_within"].abs() > blocks["cutoff"]
    worst = blocks["r_within"].abs().max()
    n_flagged = int(blocks["exceeds"].sum())
    hours = (cells["finished"].max() - cells["started"].min()).total_seconds() / 3600
    overlap = window_overlap(blocks)
    if n_flagged:
        verdict = f"FLAG: {n_flagged}/{len(blocks)} blocks trend"
    elif overlap >= 0.5:
        verdict = "CLEAN (arms overlap in time)"
    else:
        verdict = "CLEAN (blocks internally stable; windows disjoint)"
    return {
        "file": label,
        "n_ok": len(cells),
        "hours": round(hours, 1),
        "blocks": len(blocks),
        "r_residual": round(residual_trend(cells), 3),
        "max_abs_r_within": round(float(worst), 3) if pd.notna(worst) else float("nan"),
        "blocks_trending": n_flagged,
        "window_overlap": round(overlap, 2),
        "verdict": verdict,
        "_blocks": blocks,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="sweep parquet files")
    parser.add_argument("--detail", action="store_true", help="print per-block rows")
    args = parser.parse_args(list(argv) if argv is not None else None)

    summaries = []
    for source in args.sources:
        result = audit(pd.read_parquet(source), label=Path(source).stem)
        blocks = result.pop("_blocks")
        summaries.append(result)
        if args.detail:
            print(f"\n=== {result['file']} ===")
            print(
                blocks.assign(
                    window_start=blocks["window_start"].dt.strftime("%m-%d %H:%M"),
                    window_end=blocks["window_end"].dt.strftime("%H:%M"),
                    tokens_per_call=blocks["tokens_per_call"].round(0),
                    r_within=blocks["r_within"].round(3),
                    cutoff=blocks["cutoff"].round(3),
                ).to_string(index=False)
            )
    print("\n=== drift audit ===")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(
        f"\nflag threshold: |r_within| > {WITHIN_BLOCK_SIGMA_LIMIT}/sqrt(n-3) "
        "(launch order, tokens per LLM call)"
    )
    print(
        "window_overlap 1.0 = arms ran concurrently, so ordering cannot bias "
        "them; 0.0 = strictly sequential, arm confounded with time."
    )


if __name__ == "__main__":
    main()
