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
import socket
import sys
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

# Hard upper bound on every socket operation in this process. Without
# this, an SSL socket read can block forever when the upstream provider
# (e.g., OpenRouter routing through a backend like Parasail or AtlasCloud)
# accepts a request and stops sending bytes mid-response. The OpenAI
# Python SDK uses httpx via a sync→async bridge; when the worker thread
# stalls on `_ssl__SSLSocket_read → PySSL_select → poll`, the main thread
# blocks indefinitely on the inter-thread queue. `litellm.completion`'s
# own `timeout` kwarg only governs higher-level retry budget, not the
# socket read itself. This default ensures any stuck call surfaces a
# socket-timeout exception within 30s, which `num_retries=3` in
# `_CountingLLM` will then exponentially-backoff retry. Discovered via
# systematic-debugging on a 12-min hung CLI run during M4b smoke.
_DEFAULT_SOCKET_TIMEOUT_SECONDS = 30
socket.setdefaulttimeout(_DEFAULT_SOCKET_TIMEOUT_SECONDS)

# Imports below this line intentionally come AFTER socket.setdefaulttimeout
# so litellm / httpx / openai SDK pick up the global default at import time.
# E402 noqa is local to this file and intentional.

if TYPE_CHECKING:
    from collections.abc import Sequence

from .checkpoint import (  # noqa: E402 (intentional: after socket.setdefaulttimeout)
    append_record_jsonl,
    done_cell_keys,
    read_records_jsonl,
)
from .orchestrator import (  # noqa: E402 (intentional: after socket.setdefaulttimeout)
    AGENT_REGISTRY,
    SweepSpec,
    count_cells,
    iter_sweep_cells,
    run_sweep,
)
from .results import (  # noqa: E402 (intentional: after socket.setdefaulttimeout)
    RunRecord,
    write_records_csv,
    write_records_parquet,
)

# Load .env so OPENROUTER_API_KEY (and any other auth) is available to
# litellm.completion when LLM-bearing variants run. Idempotent: safe to
# call at import time. Mirrors the pattern in
# `evaluation/research_pipeline/run_experiment.py:54`. Without this,
# `python -m evaluation.chamber_pipeline.run_experiment --pilot` would
# fail with auth errors unless the user manually sourced .env first.
load_dotenv()

# Pre-baked sweep specs matching plan §9 milestones. CLI flags --pilot
# and --m5 select these; --custom (the default) lets the user override
# every axis individually.
# The five M4b variants, named EXPLICITLY rather than left as `None`.
# `agent_names=None` means "every registered agent", so when the M6 ladder
# arms joined the registry it silently redefined what `--pilot` and `--m5`
# mean -- `--pilot` would no longer reproduce the pilot it is named after.
# A preset that reproduces a published run must not depend on the registry
# staying the size it was.
_M4B_VARIANTS = (
    "random",
    "greedy_ig_lite",
    "llm_only",
    "llm_pc",
    "planner_reasoner",
)

# The M6 coordination ladder: two M4b arms reused as rungs 0 and 3, plus the
# three new ones. Selected with --variants; see spec §3.
LADDER_VARIANTS = (
    "llm_pc",  # rung 0: loop
    "fan_in_homog",  # rung 1: ensemble
    "fan_in_spec",  # rung 2: parallel roles
    "planner_reasoner",  # rung 3: chain
    "team",  # rung 4: negotiation
)

PILOT_SPEC = SweepSpec(
    chambers=("lt",),
    budget_fractions=(0.10, 0.50, 1.00),
    agent_names=_M4B_VARIANTS,  # LT runs everything per plan §5.1
    seeds=tuple(range(30)),
    configuration="standard",
)

M6_SPEC = SweepSpec(
    chambers=("lt",),
    # {6, 30, 45}, NOT the pilot's {6, 30, 59}. At k=59 the budget equals the
    # whole 59-item menu, so two blind scouts drawing 30 and 29 cover ~44
    # distinct experiments against `llm_pc`'s 59 -- a 25% coverage deficit
    # baked into the top budget point, the confound spec §3 rules out.
    #
    # The middle fraction stays 0.50, matching PILOT_SPEC exactly. 0.51 gives
    # the same k=30 but a different `budget_fraction`, and the analyzer groups
    # and plots on that column -- so reused pilot rows at 0.50 and new rows at
    # 0.51 would render as two separate Pareto points at the same budget, and
    # the mid-budget rung-vs-rung comparison would compare nothing.
    budget_fractions=(0.10, 0.50, 0.76),
    agent_names=LADDER_VARIANTS,
    seeds=tuple(range(30)),
    configuration="standard",
)

M5_SPEC = SweepSpec(
    chambers=("lt", "wt"),
    budget_fractions=(0.10, 0.25, 0.50, 0.75, 1.00),
    agent_names=_M4B_VARIANTS,  # registry handles WT-skip for greedy_ig_lite
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
    parser.add_argument(
        "--m6",
        action="store_true",
        help=(
            "Use the M6 coordination-ladder spec: LT, 3 budgets, the five "
            "ladder rungs, 30 seeds. Typing the rungs by hand into --variants "
            "risks a typo producing a silently smaller sweep."
        ),
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
    parser.add_argument(
        "--cell-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Optional per-cell wall-clock timeout. None (default) = no timeout. "
            "Recommended for M5 sweeps to recover from rare LLM API hangs "
            "without losing the surrounding sweep. Cells that time out are "
            "recorded as status='error' with error_type='TimeoutError'. "
            "Note: the underlying Python thread is NOT killed (Python doesn't "
            "support thread cancellation); use M4c parallelism for stronger "
            "isolation if needed."
        ),
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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore any existing JSONL checkpoint sidecar and start fresh. "
            "The existing sidecar is rotated to <name>.jsonl.bak-<timestamp>. "
            "Default behavior: if the sidecar exists, resume by skipping "
            "cells already in it."
        ),
    )

    return parser


def _parse_csv_list(s: str) -> list[str]:
    """Split a comma-separated string into a list, dropping empty parts."""
    return [part.strip() for part in s.split(",") if part.strip()]


def _build_sweep_from_args(args: argparse.Namespace) -> SweepSpec:
    """Translate parsed argparse Namespace into a SweepSpec.

    Pre-baked specs (--pilot, --m5, --m6) ignore custom-sweep flags entirely.
    Custom sweeps thread every CLI flag into the SweepSpec, including
    the per-cell timeout.
    """
    if args.pilot:
        # Apply only the timeout override (the rest of PILOT_SPEC is
        # plan-§9 fixed); replace via dataclasses.replace so PILOT_SPEC
        # itself stays immutable.
        if args.cell_timeout_seconds is not None:
            from dataclasses import replace

            return replace(PILOT_SPEC, cell_timeout_seconds=args.cell_timeout_seconds)
        return PILOT_SPEC
    if args.m5:
        if args.cell_timeout_seconds is not None:
            from dataclasses import replace

            return replace(M5_SPEC, cell_timeout_seconds=args.cell_timeout_seconds)
        return M5_SPEC

    if getattr(args, "m6", False):
        if args.cell_timeout_seconds is not None:
            return replace(M6_SPEC, cell_timeout_seconds=args.cell_timeout_seconds)
        return M6_SPEC

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
        cell_timeout_seconds=args.cell_timeout_seconds,
    )


class CliMockLLM:
    """In-process LLM fixture that picks the idx-th menu entry per call.

    Used for `--mock-llm` smoke testing — exercises the full pipeline
    end-to-end without OpenRouter spend. Promoted to module scope (vs
    inline in `_build_mock_llm`) so users debugging the CLI can
    instantiate it directly:

        >>> from evaluation.chamber_pipeline.run_experiment import CliMockLLM
        >>> mock = CliMockLLM()
        >>> # pass `mock` as `llm=` to run_sweep / run_cell directly

    Picks the idx-th distinct menu entry from the user prompt's
    `Menu:\n...` block. For `llm_only`'s adjacency-emission prompt
    (which has no menu, just node names + a JSON-format ask), emits
    `{}` — i.e., the "no edges" baseline. This means `--mock-llm`
    runs against `llm_only` produce all-zeros adjacencies, which is
    fine for smoke-testing the orchestration but useless for actual
    quality measurement (real LLM-only needs DeepSeek for that).

    Mirrors the FakeLLM / `_indexed_menu_responder` pattern from the
    M3b/M3c test files but inlined here so the CLI doesn't depend on
    the test directory at runtime.
    """

    # Menu entries across LT and WT chambers start with one of these
    # prefixes (LT: uniform_*; WT: actuators_*, loads_*, regime_*).
    # Conservative match — adding new chambers may need new prefixes.
    _MENU_PREFIXES = ("uniform_", "exp_", "actuators_", "loads_", "regime_")

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, model: str, messages: list[dict[str, str]], **_: Any) -> dict:
        idx = len(self.calls)
        self.calls.append({"model": model, "messages": messages, "idx": idx})

        user_text = messages[-1]["content"]
        menu_entries = [
            line.strip()
            for line in user_text.splitlines()
            if line.strip().startswith(self._MENU_PREFIXES)
        ]
        # llm_only's adjacency-emission prompt has no menu — emit empty graph.
        content = menu_entries[idx % len(menu_entries)] if menu_entries else "{}"
        return {"choices": [{"message": {"content": content}}]}


def _build_mock_llm() -> CliMockLLM:
    """Factory wrapper kept for argparse-driven CLI use."""
    return CliMockLLM()


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
        seeds_max = max(sweep.seeds) if sweep.seeds else -1
        print(f"Seeds: {len(sweep.seeds)} (range 0..{seeds_max})")
        print(f"Skipped cells (registry-incompatible): {len(cells) - len(compatible)}")
        return 0

    if not args.out:
        parser.error("--out is required unless --dry-run is set.")

    # Validate output extension early — same check as the final write, but
    # surfaced before the (potentially hours-long) sweep so the user
    # doesn't lose work to a typo.
    out = args.out
    if not (out.endswith(".csv") or out.endswith(".parquet")):
        parser.error(
            f"--out must end in .parquet or .csv; got {out!r}. "
            f"Append the desired extension explicitly."
        )

    # JSONL sidecar lives next to --out. Per-cell appends are atomic at
    # line granularity (POSIX small-write guarantee); a kill loses at
    # most the current cell, not the entire sweep. See checkpoint.py.
    from pathlib import Path as _Path

    sidecar_path = _Path(out).with_suffix(".jsonl")

    # Resume-detection (Option A): if the parquet output already exists
    # but the sidecar doesn't, the user is at risk of clobbering a
    # completed run. Error out with a hint rather than silently
    # overwriting.
    if _Path(out).exists() and not sidecar_path.exists() and not args.no_resume:
        parser.error(
            f"Output file {out!r} exists but no sidecar {str(sidecar_path)!r} "
            f"was found — this looks like a completed prior run. Pass "
            f"--no-resume to start fresh (overwrites the existing file), "
            f"or use a different --out path."
        )

    # Honor --no-resume: rotate the sidecar so the existing partial
    # results are preserved for forensics but the new run starts clean.
    if args.no_resume and sidecar_path.exists():
        from datetime import datetime as _dt

        backup = sidecar_path.with_suffix(f".jsonl.bak-{_dt.now().strftime('%Y%m%dT%H%M%S')}")
        sidecar_path.rename(backup)
        print(f"[resume] --no-resume: rotated existing sidecar to {backup}")

    # Read sidecar (if any) to build the skip-keyset for run_sweep.
    prior_records = read_records_jsonl(sidecar_path)
    skip_keys = done_cell_keys(prior_records) if prior_records else None
    total_planned = count_cells(sweep)
    if skip_keys:
        print(
            f"[resume] found {len(prior_records)} prior records in "
            f"{sidecar_path}; skipping those cells "
            f"({total_planned - len(skip_keys)}/{total_planned} remaining)"
        )

    # Optional mocked LLM for offline smoke runs.
    llm = _build_mock_llm() if args.mock_llm else None

    # Per-cell progress callback. Three modes:
    #   - --quiet        : silent
    #   - default        : every `progress_interval` cells, print ETA + counts
    #   - last cell      : always print final summary line
    # The 30-60min M4b run benefits from periodic ETA updates (vs the
    # original "dot per cell" stream which gave no time information).
    import time as _time

    sweep_t0 = _time.perf_counter()
    progress_interval = 10  # Print every Nth cell (and on errors / on the final cell).
    counts = {"ok": 0, "skipped": 0, "error": 0}

    def progress(record: RunRecord, idx: int, total: int) -> None:
        # DURABILITY FIRST: append to JSONL sidecar BEFORE any printing
        # or stat-keeping. If the process is killed between the write
        # and the print, the data is still on disk and the next run
        # will see this cell as already-done. The print is decorative;
        # the file write is the source of truth.
        append_record_jsonl(record, sidecar_path)

        counts[record.status] += 1
        # Always log error details to stderr immediately, even under
        # --quiet — without this, debugging a failing sweep means
        # waiting for the parquet write at the very end. The 4-line
        # block (cell index, agent, error_type, message snippet) is
        # enough to diagnose most failures (rate limits, API errors,
        # parsing bugs) in real time.
        if record.status == "error":
            sys.stderr.write(
                f"[ERROR-DETAIL] cell {idx + 1}/{total} "
                f"({record.agent_name} chamber={record.chamber} "
                f"k={record.budget_k} seed={record.seed}): "
                f"{record.error_type}: "
                f"{(record.error_message or '')[:200]}\n"
            )
            sys.stderr.flush()

        if args.quiet:
            return
        # Print on intervals, on the final cell, or when an error fires
        # (errors are interesting enough to surface immediately).
        is_last = (idx + 1) == total
        is_interval = (idx + 1) % progress_interval == 0
        is_error = record.status == "error"
        if not (is_last or is_interval or is_error):
            return
        elapsed = _time.perf_counter() - sweep_t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (total - (idx + 1)) / rate if rate > 0 else 0
        eta_min, eta_sec = divmod(int(remaining), 60)
        elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
        prefix = "[ERROR]" if is_error else "[progress]"
        sys.stdout.write(
            f"{prefix} {idx + 1}/{total} cells "
            f"(ok={counts['ok']} skipped={counts['skipped']} error={counts['error']}) "
            f"elapsed={elapsed_min}m{elapsed_sec:02d}s "
            f"eta={eta_min}m{eta_sec:02d}s\n"
        )
        sys.stdout.flush()

    new_records = run_sweep(sweep, llm=llm, on_cell=progress, skip_keys=skip_keys)

    # Consolidate from sidecar (NOT the in-memory list) so the final
    # Parquet contains both the prior records and the new ones. Reading
    # back from disk is also a sanity check: if the sidecar wrote
    # successfully every cell, this should equal `prior_records + new_records`.
    all_records = read_records_jsonl(sidecar_path)

    # Write output. Extension was validated above.
    if out.endswith(".csv"):
        write_records_csv(all_records, out)
    else:
        write_records_parquet(all_records, out)

    print(_format_record_summary(all_records))
    print(
        f"Wrote {len(all_records)} records to {out} "
        f"(new this run: {len(new_records)}, resumed: {len(prior_records)})"
    )

    # Exit non-zero if every cell errored — but tolerate partial failures.
    if all_records and all(r.status == "error" for r in all_records):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
