"""Re-score recorded cells at many PC subsample seeds, with no LLM calls.

A cell's F1 carries two independent sources of spread (register §21, §26):

* **design variance** — WHICH experiments were bought. For an LLM arm the cell
  seed does not reach the model (`_llm_select_loop` uses it only for the
  off-menu fallback), so every across-cell difference in the buy is the
  model's own chaotic fork.
* **inference variance** — given an identical buy, which graph PC returns from
  its 300-row subsample and its accept/reject cascade.

Only the second averages away, and it does so for free: `chosen_experiments`
is recorded from M7 onward, so the purchased data can be rebuilt and re-scored
under `m` different subsample seeds without re-running a single LLM call.
Averaging shrinks the inference component by sqrt(m) and leaves the design
component untouched, which tightens every contrast the corpus reports.

**Cluster by selection before averaging.** A single-call arm re-picks the same
design across cells (register §24: `one_shot` produced 6 distinct designs in 30
LT k=30 cells). Averaging without clustering treats 30 re-scorings of 6 designs
as 30 independent observations and manufactures precision that is not there —
the exact error this module exists to avoid, and one this session committed
before catching it. `selection_key` is emitted on every row so the caller can
group correctly, and `SELECTION_KEY_COLUMN` names it.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from agent_contracts.integrations.causalchamber import (
    create_contracted_chamber_agent,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

from .inference import pool_experiment_data, run_pc
from .scoring import f1_edges, shd

SELECTION_KEY_COLUMN = "selection_key"

#: Columns a source frame must carry to be re-scorable.
REQUIRED_COLUMNS = ("chamber", "configuration", "chosen_experiments", "status")


def selection_key(chamber: str, configuration: str, names: Sequence[str]) -> str:
    """Stable id for one purchased design.

    Order-insensitive: the buy is a SET, and two cells that bought the same
    experiments in a different order pooled identical data and must share a
    key. Hashed rather than joined so the value stays a fixed width whatever
    the budget.
    """
    payload = "|".join([chamber, configuration, *sorted(names)])
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def parse_selection(raw: object) -> list[str]:
    """Split a recorded `chosen_experiments` value into names.

    Recorded as a comma-separated string. Returns [] for null or empty, which
    the caller must skip: an empty buy has no data to pool and no graph to
    score.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def rescore_selections(
    frame: pd.DataFrame,
    *,
    n_pc_seeds: int = 9,
    pc_alpha: float = 0.05,
    progress_every: int = 50,
) -> pd.DataFrame:
    """Score every DISTINCT recorded buy in `frame` under `n_pc_seeds` seeds.

    Pools once per distinct buy rather than once per cell — a single-call arm
    can repeat one design dozens of times, and rebuilding its data each time
    would dominate the runtime while changing nothing.

    Returns one row per (selection_key, pc_seed).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"frame is missing {missing}; only M7-era files record "
            "`chosen_experiments` and can be re-scored"
        )

    ok = frame[frame["status"] == "ok"]
    seen: dict[str, tuple[str, str, list[str]]] = {}
    for _, row in ok.iterrows():
        names = parse_selection(row["chosen_experiments"])
        if not names:
            continue
        key = selection_key(row["chamber"], row["configuration"], names)
        seen.setdefault(key, (row["chamber"], row["configuration"], names))

    records: list[dict[str, Any]] = []
    for index, (key, (chamber, configuration, names)) in enumerate(seen.items()):
        adapter = create_contracted_chamber_agent(
            chamber=chamber,  # type: ignore[arg-type]
            configuration=configuration,  # type: ignore[arg-type]
            intervention_budget=len(names),
        )
        nodes = list(adapter.ground_truth().index)
        truth = adapter.ground_truth()
        pooled = pool_experiment_data([adapter.query_intervention(name) for name in names], nodes)
        for pc_seed in range(n_pc_seeds):
            predicted = run_pc(pooled, nodes, alpha=pc_alpha, seed=pc_seed)
            records.append(
                {
                    SELECTION_KEY_COLUMN: key,
                    "chamber": chamber,
                    "configuration": configuration,
                    "n_experiments": len(names),
                    "pc_seed": pc_seed,
                    "f1": float(f1_edges(predicted, truth)),
                    "shd": float(shd(predicted, truth)),
                }
            )
        if progress_every and (index + 1) % progress_every == 0:
            print(f"  [{index + 1}/{len(seen)}] designs re-scored", flush=True)
    return pd.DataFrame(records)


def attach_rescored(frame: pd.DataFrame, rescored: pd.DataFrame) -> pd.DataFrame:
    """Join averaged scores back onto the original cells.

    Adds `selection_key`, `f1_rescored` (mean over PC seeds) and `n_pc_seeds`.
    The original `f1` is left untouched so the two can be compared.
    """
    per_key = (
        rescored.groupby(SELECTION_KEY_COLUMN)
        .agg(f1_rescored=("f1", "mean"), n_pc_seeds=("f1", "size"))
        .reset_index()
    )
    out = frame.copy()
    out[SELECTION_KEY_COLUMN] = [
        selection_key(r["chamber"], r["configuration"], parse_selection(r["chosen_experiments"]))
        if parse_selection(r["chosen_experiments"])
        else None
        for _, r in out.iterrows()
    ]
    return out.merge(per_key, on=SELECTION_KEY_COLUMN, how="left")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="Parquet files to re-score")
    parser.add_argument("--n-pc-seeds", type=int, default=9)
    parser.add_argument("--out", default="runs/rescored.parquet")
    args = parser.parse_args(list(argv) if argv is not None else None)

    frames = []
    for src in args.sources:
        d = pd.read_parquet(src)
        d["source_file"] = Path(src).stem
        frames.append(d)
    combined = pd.concat(frames, ignore_index=True)
    print(f"{len(combined)} rows from {len(args.sources)} files", flush=True)

    rescored = rescore_selections(combined, n_pc_seeds=args.n_pc_seeds)
    joined = attach_rescored(combined, rescored)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out, index=False)
    rescored.to_parquet(out.with_name(out.stem + "-bykey.parquet"), index=False)
    print(f"wrote {len(joined)} cells to {out}")
    print(f"wrote {len(rescored)} (design x pc_seed) rows to {out.stem}-bykey.parquet")


if __name__ == "__main__":
    main()
