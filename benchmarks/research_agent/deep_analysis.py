"""Deep analysis of all benchmark runs to identify fundamental issues."""

import json
from pathlib import Path
from typing import Any


def analyze_all_runs() -> None:
    """Analyze all recent benchmark runs."""
    result_dir = Path("benchmarks/research_agent/results")

    # Get the 3 latest runs with planning=1200
    files = [
        "benchmark_results_20251102_120355.json",  # Run 1
        "benchmark_results_20251102_123748.json",  # Run 2
        "benchmark_results_20251102_141623.json",  # Run 3
    ]

    print("=" * 80)
    print("DEEP ANALYSIS: Are We Using More Tokens For Better Quality?")
    print("=" * 80)
    print()

    # Collect data across all runs
    all_data: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for i, filename in enumerate(files, 1):
        with open(result_dir / filename) as f:
            data = json.load(f)

        print(f"Run {i}: {filename}")
        print("-" * 80)

        for result in data["results"]:
            q_id = result["question_id"]
            agent_type = result["agent_type"]

            if q_id not in all_data:
                all_data[q_id] = {"uncontracted": [], "contracted": []}

            all_data[q_id][agent_type].append(
                {
                    "run": i,
                    "quality": result["quality_score"]["total"],
                    "tokens": result["total_tokens"],
                    "reasoning_tokens": result["total_reasoning_tokens"],
                    "text_tokens": result["total_text_tokens"],
                    "cost": result["total_cost"],
                    "api_calls": result["api_calls"],
                }
            )

        # Print aggregate for this run
        unc_avg_quality = (
            sum(
                r["quality_score"]["total"]
                for r in data["results"]
                if r["agent_type"] == "uncontracted"
            )
            / 5
        )
        cont_avg_quality = (
            sum(
                r["quality_score"]["total"]
                for r in data["results"]
                if r["agent_type"] == "contracted"
            )
            / 5
        )
        unc_avg_tokens = (
            sum(r["total_tokens"] for r in data["results"] if r["agent_type"] == "uncontracted") / 5
        )
        cont_avg_tokens = (
            sum(r["total_tokens"] for r in data["results"] if r["agent_type"] == "contracted") / 5
        )

        print(f"  Uncontracted: Quality={unc_avg_quality:.1f}, Tokens={unc_avg_tokens:.0f}")
        print(f"  Contracted:   Quality={cont_avg_quality:.1f}, Tokens={cont_avg_tokens:.0f}")
        print(
            f"  Difference:   Quality={cont_avg_quality - unc_avg_quality:+.1f}, Tokens={cont_avg_tokens - unc_avg_tokens:+.0f}"
        )
        print()

    print("\n" + "=" * 80)
    print("PER-QUESTION ANALYSIS ACROSS ALL 3 RUNS")
    print("=" * 80)
    print()

    for q_id in sorted(all_data.keys()):
        q_data = all_data[q_id]

        print(f"\n{q_id}:")
        print("-" * 80)

        # Print table header
        print(
            f"{'Run':<6} {'Agent':<15} {'Quality':<10} {'Tokens':<10} {'Reasoning':<12} {'Text':<10} {'Cost':<10}"
        )
        print("-" * 80)

        for run_idx in range(1, 4):
            unc = q_data["uncontracted"][run_idx - 1]
            cont = q_data["contracted"][run_idx - 1]

            print(
                f"{run_idx:<6} {'Uncontracted':<15} {unc['quality']:<10.1f} {unc['tokens']:<10} {unc['reasoning_tokens']:<12} {unc['text_tokens']:<10} ${unc['cost']:<9.4f}"
            )
            print(
                f"{'':6} {'Contracted':<15} {cont['quality']:<10.1f} {cont['tokens']:<10} {cont['reasoning_tokens']:<12} {cont['text_tokens']:<10} ${cont['cost']:<9.4f}"
            )

            quality_diff = cont["quality"] - unc["quality"]
            token_diff = cont["tokens"] - unc["tokens"]

            symbol = (
                "✅"
                if quality_diff >= 0 and token_diff <= 0
                else "⚠️"
                if quality_diff >= 0
                else "❌"
            )
            print(
                f"{'':6} {'Δ (C-U)':<15} {quality_diff:+10.1f} {token_diff:+10} {'':12} {'':10} {'':10} {symbol}"
            )
            print()

        # Statistics across runs
        unc_qualities = [d["quality"] for d in q_data["uncontracted"]]
        cont_qualities = [d["quality"] for d in q_data["contracted"]]
        unc_tokens = [d["tokens"] for d in q_data["uncontracted"]]
        cont_tokens = [d["tokens"] for d in q_data["contracted"]]

        import statistics

        print(
            f"{'Mean':<6} {'Uncontracted':<15} {statistics.mean(unc_qualities):<10.1f} {statistics.mean(unc_tokens):<10.0f}"
        )
        print(
            f"{'':6} {'Contracted':<15} {statistics.mean(cont_qualities):<10.1f} {statistics.mean(cont_tokens):<10.0f}"
        )
        print(
            f"{'StdDev':<6} {'Uncontracted':<15} {statistics.stdev(unc_qualities) if len(unc_qualities) > 1 else 0:<10.1f} {statistics.stdev(unc_tokens) if len(unc_tokens) > 1 else 0:<10.0f}"
        )
        print(
            f"{'':6} {'Contracted':<15} {statistics.stdev(cont_qualities) if len(cont_qualities) > 1 else 0:<10.1f} {statistics.stdev(cont_tokens) if len(cont_tokens) > 1 else 0:<10.0f}"
        )

    print("\n" + "=" * 80)
    print("FUNDAMENTAL QUESTIONS")
    print("=" * 80)
    print()

    # Calculate overall averages
    all_unc_quality = []
    all_cont_quality = []
    all_unc_tokens = []
    all_cont_tokens = []

    for q_data in all_data.values():
        all_unc_quality.extend([d["quality"] for d in q_data["uncontracted"]])
        all_cont_quality.extend([d["quality"] for d in q_data["contracted"]])
        all_unc_tokens.extend([d["tokens"] for d in q_data["uncontracted"]])
        all_cont_tokens.extend([d["tokens"] for d in q_data["contracted"]])

    avg_unc_quality = statistics.mean(all_unc_quality)
    avg_cont_quality = statistics.mean(all_cont_quality)
    avg_unc_tokens = statistics.mean(all_unc_tokens)
    avg_cont_tokens = statistics.mean(all_cont_tokens)

    quality_wins = sum(
        1 for cq, uq in zip(all_cont_quality, all_unc_quality, strict=True) if cq > uq
    )
    quality_ties = sum(
        1 for cq, uq in zip(all_cont_quality, all_unc_quality, strict=True) if abs(cq - uq) < 0.1
    )
    quality_losses = sum(
        1 for cq, uq in zip(all_cont_quality, all_unc_quality, strict=True) if cq < uq
    )

    token_wins = sum(1 for ct, ut in zip(all_cont_tokens, all_unc_tokens, strict=True) if ct < ut)
    token_losses = sum(1 for ct, ut in zip(all_cont_tokens, all_unc_tokens, strict=True) if ct > ut)

    print("1. Do we consume MORE tokens with contracts?")
    print(f"   Avg Uncontracted: {avg_unc_tokens:.0f} tokens")
    print(f"   Avg Contracted:   {avg_cont_tokens:.0f} tokens")
    print(
        f"   Difference:       {avg_cont_tokens - avg_unc_tokens:+.0f} tokens ({(avg_cont_tokens - avg_unc_tokens) / avg_unc_tokens * 100:+.1f}%)"
    )
    print(f"   Result: {token_wins} times LESS, {token_losses} times MORE")
    print()

    print("2. Do we get BETTER quality with contracts?")
    print(f"   Avg Uncontracted: {avg_unc_quality:.1f}/100")
    print(f"   Avg Contracted:   {avg_cont_quality:.1f}/100")
    print(f"   Difference:       {avg_cont_quality - avg_unc_quality:+.1f} points")
    print(f"   Result: {quality_wins} wins, {quality_ties} ties, {quality_losses} losses")
    print()

    print("3. Is there a CONSISTENT pattern?")

    # Check how many times we use MORE tokens AND get WORSE quality
    bad_cases = sum(
        1
        for ct, ut, cq, uq in zip(
            all_cont_tokens, all_unc_tokens, all_cont_quality, all_unc_quality, strict=True
        )
        if ct > ut and cq < uq
    )
    good_cases = sum(
        1
        for ct, ut, cq, uq in zip(
            all_cont_tokens, all_unc_tokens, all_cont_quality, all_unc_quality, strict=True
        )
        if ct <= ut and cq >= uq
    )

    print(
        f"   BAD cases (MORE tokens, WORSE quality): {bad_cases}/15 ({bad_cases / 15 * 100:.1f}%)"
    )
    print(
        f"   GOOD cases (SAME/LESS tokens, SAME/BETTER quality): {good_cases}/15 ({good_cases / 15 * 100:.1f}%)"
    )
    print()

    print("4. What's the variance in baseline (uncontracted)?")
    print(f"   Quality StdDev: {statistics.stdev(all_unc_quality):.1f} points")
    print(f"   Tokens StdDev: {statistics.stdev(all_unc_tokens):.0f} tokens")
    print()

    print("5. What's the variance in contracted?")
    print(f"   Quality StdDev: {statistics.stdev(all_cont_quality):.1f} points")
    print(f"   Tokens StdDev: {statistics.stdev(all_cont_tokens):.0f} tokens")
    print()

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()

    if avg_cont_tokens > avg_unc_tokens:
        print("❌ PROBLEM: Contracted agent uses MORE tokens on average")
        print("   This suggests budgets are NOT constraining behavior")
        print()

    if statistics.stdev(all_unc_quality) > 10:
        print(
            f"❌ PROBLEM: Uncontracted baseline has HIGH variance (stddev={statistics.stdev(all_unc_quality):.1f})"
        )
        print("   This makes it hard to measure framework value reliably")
        print()

    if bad_cases > 3:
        print(f"❌ PROBLEM: {bad_cases} cases where we use MORE tokens for WORSE quality")
        print("   This suggests a fundamental design issue")
        print()

    print("POSSIBLE ROOT CAUSES:")
    print("1. Both agents use the SAME workflow (decompose-research-synthesize)")
    print("   → Budgets aren't forcing different strategies")
    print()
    print("2. Budgets are TOO GENEROUS (not binding)")
    print("   → Agents make similar decisions regardless of constraints")
    print()
    print("3. Baseline variance is TOO HIGH")
    print("   → Can't measure small improvements reliably")
    print()
    print("4. Benchmark doesn't test the RIGHT scenarios")
    print("   → Should test: budget violations, cost control, governance")
    print()


if __name__ == "__main__":
    analyze_all_runs()
