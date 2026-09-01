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
from .scoring import f1_edges, f1_skeleton, shd

SELECTION_KEY_COLUMN = "selection_key"

#: The 20 light-tunnel variables used by the chambers' own causal-discovery
#: case study (`causal-chamber-paper/case_studies/causal_discovery_iid.ipynb`).
#: Our node set adds 18 more, and every one of them is a PURE SOURCE in the
#: ground truth -- out-degree 1, in-degree 0 -- because they are apparatus
#: settings (exposure time, oversampling rate, reference voltage, diode
#: select) that each drive exactly one sensor. They carry 18 of the 57 true
#: edges, so a third of the recoverable structure is "did you buy the
#: experiment that makes this setting vary" rather than "did you infer
#: non-obvious structure". Scoring the induced subgraph on these 20 is the
#: robustness check for that, NOT a redefinition of the metric.
#: No published equivalent exists for the wind tunnel, so this is LT-only.
LT_CASE_STUDY_NODES: tuple[str, ...] = (
    "red",
    "green",
    "blue",
    "current",
    "ir_1",
    "ir_2",
    "ir_3",
    "vis_1",
    "vis_2",
    "vis_3",
    "pol_1",
    "pol_2",
    "angle_1",
    "angle_2",
    "l_11",
    "l_12",
    "l_21",
    "l_22",
    "l_31",
    "l_32",
)

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
                    # Undirected companion, for the robustness check: the
                    # chambers' own case study scores the equivalence class
                    # rather than one orientation. NOT a replacement -- see
                    # `f1_skeleton`, the two are different metrics.
                    "f1_skeleton": float(f1_skeleton(predicted, truth)),
                    # Induced subgraph on the case study's 20 variables,
                    # excluding the pure-source settings. LT only.
                    "f1_core": (
                        float(f1_edges(predicted.loc[core, core], truth.loc[core, core]))
                        if (core := [n for n in LT_CASE_STUDY_NODES if n in nodes])
                        and len(core) == len(LT_CASE_STUDY_NODES)
                        else float("nan")
                    ),
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
    # Aggregate whatever metric columns are present. The optional ones
    # (`f1_skeleton`, `f1_core`) were added after the first re-scoring run,
    # and a frame written before that must still join rather than raise --
    # pandas' `agg` fails hard on a named column it cannot find.
    aggregations: dict[str, tuple[str, str]] = {
        "f1_rescored": ("f1", "mean"),
        "n_pc_seeds": ("f1", "size"),
    }
    for column, alias in (
        ("f1_skeleton", "f1_skeleton_rescored"),
        ("f1_core", "f1_core_rescored"),
    ):
        if column in rescored.columns:
            aggregations[alias] = (column, "mean")
    per_key = (
        rescored.groupby(SELECTION_KEY_COLUMN).agg(**aggregations).reset_index()  # type: ignore[call-overload]
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
