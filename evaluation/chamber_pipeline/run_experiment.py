"""CLI entry point for chamber-pillar sweeps.

Thin wrapper around `orchestrator.run_sweep`. Nothing here is
framework-specific or LLM-specific — all logic lives in
`orchestrator.py`. This file's only job is to translate command-line
flags into a `SweepSpec` and shepherd the output to disk.

Usage:

    # M4 pilot (the §9 milestone): LT, 3 budgets, all 5 variants, 30 seeds
    python -m evaluation.chamber_pipeline.run_experiment --pilot --out runs/m4-pilot.parquet

    # Full M5 sweep (after M4 pilot succeeds): both chambers, 5 budgets
    python -m evaluation.chamber_pipeline.run_experiment --m5 --out runs/m5-flash.parquet

    # Custom: just the random + llm_pc variants on LT, 3 seeds, fast mock LLM
    python -m evaluation.chamber_pipeline.run_experiment \\
        --chambers lt --budgets 0.5 --variants random,llm_pc --seeds 3 \\
        --mock-llm --out runs/quick.parquet

    # Dry-run: how many cells will I actually invoke?
    python -m evaluation.chamber_pipeline.run_experiment --pilot --dry-run

The `--mock-llm` flag injects a FakeLLM that picks the first menu
entry on every call. Useful for end-to-end CLI smoke-testing without
spending OpenRouter credits. Real production runs (M4b / M5) omit
this flag and let agents lazy-import `litellm.completion`.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from .orchestrator import (
    AGENT_REGISTRY,
    SweepSpec,
    count_cells,
    iter_sweep_cells,
    run_sweep,
)
from .results import RunRecord, write_records_csv, write_records_parquet

# Pre-baked sweep specs matching plan §9 milestones. CLI flags --pilot
# and --m5 select these; --custom (the default) lets the user override
# every axis individually.
PILOT_SPEC = SweepSpec(
    chambers=("lt",),
    budget_fractions=(0.10, 0.50, 1.00),
    agent_names=None,  # all 5 — LT runs everything per plan §5.1
    seeds=tuple(range(30)),
    configuration="standard",
)

M5_SPEC = SweepSpec(
    chambers=("lt", "wt"),
    budget_fractions=(0.10, 0.25, 0.50, 0.75, 1.00),
    agent_names=None,  # all — registry handles WT-skip for greedy_ig_lite
    seeds=tuple(range(30)),
    configuration="standard",
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argparse parser. Factored for testability."""
    parser = argparse.ArgumentParser(
        prog="run_experiment",
        description=(
            "Run a chamber-pillar sweep and write RunRecords to Parquet/CSV. "
            "Plan §9 milestones M4 (--pilot) and M5 (--m5) have pre-baked specs; "
            "use --chambers/--budgets/--variants/--seeds to define a custom sweep."
        ),
    )

    # Mutually-exclusive preset selectors. None = custom (use individual flags).
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument(
        "--pilot",
        action="store_true",
        help="Use the M4 pilot spec: LT only, 3 budgets, 5 variants, 30 seeds = 450 cells.",
    )
    preset.add_argument(
        "--m5",
        action="store_true",
        help="Use the M5 full-sweep spec: both chambers, 5 budgets, all variants, 30 seeds.",
    )

    # Custom-sweep flags (used when no preset is selected).
    parser.add_argument(
        "--chambers",
        type=str,
        default="lt",
        help=(
            "Comma-separated chamber IDs. Default: 'lt'. "
            "Custom-sweep only — ignored under --pilot/--m5."
        ),
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="0.10,0.50,1.00",
        help=(
            "Comma-separated budget fractions in [0, 1]. Default: '0.10,0.50,1.00'. "
            "Custom-sweep only."
        ),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help=(
            "Comma-separated variant names from the registry. Empty (default) = all. "
            f"Available: {','.join(s.name for s in AGENT_REGISTRY)}. Custom-sweep only."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of seeds (always range(N)). Default: 30. Custom-sweep only.",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="standard",
        help="Chamber configuration. Default: 'standard'.",
    )
    parser.add_argument(
        "--pc-alpha",
        type=float,
        default=0.05,
        help="PC independence-test significance level. Default: 0.05.",
    )

    # Output / control flags.
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output file path. Extension determines format: .parquet (default if "
            "no extension given), .csv. Required unless --dry-run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cell grid + count, do not invoke agents.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help=(
            "Inject a FakeLLM that picks the first menu entry. Useful for "
            "end-to-end CLI smoke-testing without OpenRouter spend. "
            "Production runs omit this flag."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-cell progress output. Final summary still prints.",
    )

    return parser


def _parse_csv_list(s: str) -> list[str]:
    """Split a comma-separated string into a list, dropping empty parts."""
    return [part.strip() for part in s.split(",") if part.strip()]


def _build_sweep_from_args(args: argparse.Namespace) -> SweepSpec:
    """Translate parsed argparse Namespace into a SweepSpec."""
    if args.pilot:
        return PILOT_SPEC
    if args.m5:
        return M5_SPEC

    # Custom sweep.
    chambers = tuple(_parse_csv_list(args.chambers))
    budgets = tuple(float(x) for x in _parse_csv_list(args.budgets))
    variants_raw = _parse_csv_list(args.variants)
    agent_names: tuple[str, ...] | None = tuple(variants_raw) if variants_raw else None
    return SweepSpec(
        chambers=chambers,  # type: ignore[arg-type]
        budget_fractions=budgets,
        agent_names=agent_names,
        seeds=tuple(range(args.seeds)),
        configuration=args.configuration,  # type: ignore[arg-type]
        pc_alpha=args.pc_alpha,
    )


def _build_mock_llm() -> Any:
    """Construct a tiny in-process LLM fixture that picks the first
    menu entry from each user prompt.

    Used for `--mock-llm` smoke testing. Mirrors the FakeLLM /
    `_indexed_menu_responder` pattern from the M3b/M3c test files,
    but inlined here so the CLI doesn't import test modules.
    """

    class _CliMockLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def __call__(self, *, model: str, messages: list[dict[str, str]], **_: Any) -> dict:
            idx = len(self.calls)
            self.calls.append({"model": model, "messages": messages, "idx": idx})

            # Pick the idx-th distinct menu entry from the user prompt.
            user_text = messages[-1]["content"]
            menu_entries = [
                line.strip()
                for line in user_text.splitlines()
                if line.strip().startswith(("uniform_", "exp_", "actuators_", "loads_", "regime_"))
            ]
            # llm_only's adjacency-emission prompt has no menu — emit empty graph.
            content = menu_entries[idx % len(menu_entries)] if menu_entries else "{}"
            return {"choices": [{"message": {"content": content}}]}

    return _CliMockLLM()


def _format_record_summary(records: list[RunRecord]) -> str:
    """Tally records by status into a one-line summary string."""
    n_total = len(records)
    n_ok = sum(1 for r in records if r.status == "ok")
    n_skipped = sum(1 for r in records if r.status == "skipped")
    n_error = sum(1 for r in records if r.status == "error")
    n_pc_degen = sum(r.n_pc_degeneracies or 0 for r in records if r.n_pc_degeneracies is not None)
    return (
        f"Sweep complete: {n_total} cells "
        f"({n_ok} ok, {n_skipped} skipped, {n_error} error). "
        f"PC degeneracies fired: {n_pc_degen}."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns process exit code (0 success / 1 error)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    sweep = _build_sweep_from_args(args)

    # Dry-run: print the cell grid and exit.
    if args.dry_run:
        cells = list(iter_sweep_cells(sweep))
        compatible = [
            (spec, chamber, k) for spec, chamber, k, *_ in cells if spec.is_compatible(chamber)
        ]
        print(
            f"Sweep would iterate {len(cells)} cells "
            f"({count_cells(sweep, exclude_skipped=True)} after compatibility filter)."
        )
        print(f"Chambers: {sweep.chambers}")
        print(f"Budget fractions: {sweep.budget_fractions}")
        print(f"Agents: {[s.name for s in sweep.selected_specs()]}")
        print(f"Seeds: {len(sweep.seeds)} (range 0..{max(sweep.seeds)})")
        print(f"Skipped cells (registry-incompatible): {len(cells) - len(compatible)}")
        return 0

    if not args.out:
        parser.error("--out is required unless --dry-run is set.")

    # Optional mocked LLM for offline smoke runs.
    llm = _build_mock_llm() if args.mock_llm else None

    # Per-cell progress callback (skipped under --quiet).
    def progress(record: RunRecord, idx: int, total: int) -> None:
        if args.quiet:
            return
        marker = {"ok": ".", "skipped": "s", "error": "E"}[record.status]
        # One char per cell, line-wrapped at 80.
        sys.stdout.write(marker)
        sys.stdout.flush()
        if (idx + 1) % 80 == 0 or (idx + 1) == total:
            sys.stdout.write(f" {idx + 1}/{total}\n")
            sys.stdout.flush()

    records = run_sweep(sweep, llm=llm, on_cell=progress)

    # Write output. Extension determines format; .parquet is the default.
    out = args.out
    if out.endswith(".csv"):
        write_records_csv(records, out)
    else:
        if not out.endswith(".parquet"):
            out = out + ".parquet"
        write_records_parquet(records, out)

    print(_format_record_summary(records))
    print(f"Wrote {len(records)} records to {out}")

    # Exit non-zero if every cell errored — but tolerate partial failures.
    if records and all(r.status == "error" for r in records):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
