#!/usr/bin/env python3
"""Phase 1 Value Demonstration: Prioritized Multi-Document Q&A

This benchmark demonstrates the core value proposition of Agent Contracts:
Budget-aware agents achieve better quality-per-token efficiency through
strategic resource allocation and prioritization.

Key Hypothesis (H3): Budget-aware agents achieve ≥30% better quality-per-token
than unconstrained baseline.

Methodology:
- Task: Answer weighted questions about a document
- Questions have different importance levels (critical 3x, important 2x, nice-to-have 1x)
- Same output token budget (max_tokens) for baseline vs budget-aware
- Different strategies: baseline tries to answer all, budget-aware prioritizes
- Metric: Weighted accuracy and quality-per-token efficiency

Key Insight:
With the SAME token budget, strategic prioritization yields better weighted
accuracy by focusing on high-value questions first.
"""

import sys
import time
from dataclasses import dataclass
from typing import Any

import litellm
from dotenv import load_dotenv

from agent_contracts import Contract, ContractedLLM, ResourceConstraints

# Load environment variables
load_dotenv("../.env")

# Test document about Quantum Computing
DOCUMENT = """
Quantum Computing: A Revolutionary Paradigm

Introduction
Quantum computing represents a fundamental shift in computational capability, leveraging quantum mechanical phenomena to process information in ways impossible for classical computers. Unlike traditional computers that use bits (0 or 1), quantum computers use quantum bits or "qubits" that can exist in superposition - simultaneously representing both 0 and 1.

Core Principles
The power of quantum computing stems from three quantum mechanical principles: superposition, entanglement, and interference. Superposition allows qubits to process multiple possibilities simultaneously. Entanglement creates correlations between qubits that classical systems cannot replicate. Quantum interference amplifies correct answers while canceling out incorrect ones.

Current State (2025)
As of 2025, IBM has demonstrated quantum computers with over 1000 qubits, while Google achieved quantum supremacy in 2019 with their Sycamore processor. The technology remains largely experimental, with major challenges in error correction and maintaining quantum coherence. Current quantum computers require extreme cooling to near absolute zero (-273°C) to maintain qubit stability.

Key Applications
Quantum computing shows promise in several critical domains. In cryptography, quantum computers could break current encryption methods like RSA-2048 within hours, necessitating quantum-resistant cryptography. For drug discovery, quantum simulation could model molecular interactions at unprecedented scales, potentially reducing development time from 10 years to 1 year. In optimization, quantum algorithms could revolutionize logistics, financial modeling, and machine learning.

Major Players and Investment
Leading tech companies have invested heavily in quantum computing. IBM's Quantum Network includes over 170 organizations. Google's Quantum AI division focuses on error correction and scalability. Microsoft is developing topological qubits through Azure Quantum. China has invested $15 billion in quantum research since 2020. Total global investment in quantum computing exceeded $30 billion in 2024.

Technical Challenges
The path to practical quantum computing faces significant obstacles. Quantum decoherence causes qubits to lose their quantum state within milliseconds. Error rates remain high at approximately 0.1-1% per operation. Scaling to millions of qubits needed for practical applications requires breakthroughs in error correction. The dilution refrigerators needed cost $1-3 million each.

Timeline and Future
Experts predict "quantum advantage" for specific practical problems by 2027-2030. Fault-tolerant quantum computers with millions of qubits may not emerge until the 2040s. In the near term (2025-2027), hybrid quantum-classical algorithms show promise for optimization and machine learning. Cloud-based quantum computing services are making the technology accessible to researchers and companies without hardware investments.

Societal Impact
Quantum computing will likely transform cybersecurity, requiring complete infrastructure overhauls. The technology could accelerate climate modeling, enabling better predictions and mitigation strategies. Financial markets may leverage quantum algorithms for portfolio optimization and risk assessment. National security implications have made quantum computing a strategic priority for major nations.
"""


@dataclass
class Question:
    """A weighted question about the document."""

    text: str
    answer: str
    importance: int  # 1 (nice-to-have), 2 (important), 3 (critical)
    category: str


# Questions with different importance levels
QUESTIONS = [
    # Critical questions (3x weight) - Core facts
    Question(
        "What are the three core quantum mechanical principles that enable quantum computing?",
        "superposition, entanglement, and interference",
        importance=3,
        category="core_principles",
    ),
    Question(
        "What major achievement did Google accomplish in 2019?",
        "achieved quantum supremacy with their Sycamore processor",
        importance=3,
        category="milestones",
    ),
    Question(
        "What is the approximate error rate per operation in current quantum computers?",
        "0.1-1% per operation",
        importance=3,
        category="technical",
    ),
    Question(
        "When do experts predict quantum advantage for practical problems?",
        "2027-2030",
        importance=3,
        category="timeline",
    ),
    Question(
        "What temperature is required to maintain qubit stability?",
        "near absolute zero (-273°C)",
        importance=3,
        category="technical",
    ),
    # Important questions (2x weight) - Supporting details
    Question(
        "How many qubits has IBM demonstrated as of 2025?",
        "over 1000 qubits",
        importance=2,
        category="current_state",
    ),
    Question(
        "How much has China invested in quantum research since 2020?",
        "$15 billion",
        importance=2,
        category="investment",
    ),
    Question(
        "What could quantum computers reduce drug development time from and to?",
        "from 10 years to 1 year",
        importance=2,
        category="applications",
    ),
    Question(
        "How many organizations are in IBM's Quantum Network?",
        "over 170 organizations",
        importance=2,
        category="industry",
    ),
    Question(
        "What is the cost of dilution refrigerators needed for quantum computers?",
        "$1-3 million each",
        importance=2,
        category="technical",
    ),
    # Nice-to-have questions (1x weight) - Minor details
    Question(
        "What encryption method could quantum computers break within hours?",
        "RSA-2048",
        importance=1,
        category="applications",
    ),
    Question(
        "What was the total global investment in quantum computing in 2024?",
        "$30 billion",
        importance=1,
        category="investment",
    ),
    Question(
        "What type of qubits is Microsoft developing?",
        "topological qubits",
        importance=1,
        category="industry",
    ),
    Question(
        "Within what timeframe do qubits lose their quantum state?",
        "milliseconds",
        importance=1,
        category="technical",
    ),
    Question(
        "When might fault-tolerant quantum computers with millions of qubits emerge?",
        "2040s",
        importance=1,
        category="timeline",
    ),
]


def parse_numbered_answers(text: str) -> dict[int, str]:
    """Parse numbered answers from model response.

    Returns dict mapping question number (1-15) to answer text.
    """
    import re

    answers = {}
    lines = text.split("\n")

    current_num = None
    current_answer = []

    for line in lines:
        # Look for numbered answers (e.g., "1.", "1)", "1 -", etc.)
        match = re.match(r"^\s*(\d+)[\.\)\-\:]\s*(.+)", line)
        if match:
            # Save previous answer if exists
            if current_num is not None and current_answer:
                answers[current_num] = " ".join(current_answer).strip()

            # Start new answer
            current_num = int(match.group(1))
            current_answer = [match.group(2)]
        elif current_num is not None and line.strip():
            # Continue current answer
            current_answer.append(line.strip())

    # Save last answer
    if current_num is not None and current_answer:
        answers[current_num] = " ".join(current_answer).strip()

    return answers


def evaluate_answer(predicted: str | None, expected: str) -> float:
    """Evaluate answer correctness with fuzzy matching.

    Returns 1.0 for correct, 0.0 for incorrect.
    Uses simple keyword matching for this demo.
    """
    if predicted is None or not predicted.strip():
        return 0.0

    predicted = predicted.lower().strip()
    expected = expected.lower().strip()

    # Extract key terms from expected answer (words > 3 chars)
    expected_terms = [term for term in expected.split() if len(term) > 3]

    # Also look for exact numbers/years
    import re

    expected_numbers = set(re.findall(r"\d+", expected))
    predicted_numbers = set(re.findall(r"\d+", predicted))

    # Check if most key terms are present
    if expected_terms:
        term_matches = sum(1 for term in expected_terms if term in predicted)
        term_ratio = term_matches / len(expected_terms)
    else:
        term_ratio = 0.0

    # Check if important numbers are present
    if expected_numbers:
        number_matches = len(expected_numbers & predicted_numbers)
        number_ratio = number_matches / len(expected_numbers)
    else:
        number_ratio = 0.0

    # Consider correct if >60% of key terms present OR >60% of numbers present
    return 1.0 if (term_ratio >= 0.6 or number_ratio >= 0.6) else 0.0


def calculate_weighted_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate weighted accuracy and efficiency metrics."""
    total_weight = sum(q.importance for q in QUESTIONS)

    weighted_correct = sum(result["correct"] * result["importance"] for result in results)

    weighted_accuracy = weighted_correct / total_weight

    # Count by importance level
    by_importance: dict[int, list[float]] = {1: [], 2: [], 3: []}
    for result in results:
        by_importance[result["importance"]].append(result["correct"])

    importance_accuracy = {
        level: (sum(scores) / len(scores) if scores else 0.0)
        for level, scores in by_importance.items()
    }

    return {
        "weighted_accuracy": weighted_accuracy,
        "critical_accuracy": importance_accuracy[3],
        "important_accuracy": importance_accuracy[2],
        "nice_to_have_accuracy": importance_accuracy[1],
        "total_questions": len(results),
        "answered_questions": sum(1 for r in results if r["answered"]),
    }


def run_qa_baseline(
    model: str = "gemini/gemini-2.5-flash-preview-09-2025",
    max_tokens: int = 500,
) -> dict[str, Any]:
    """Run baseline QA - tries to answer all questions without prioritization.

    Strategy: Attempts to be comprehensive, answers questions in order.
    """
    print("\n" + "=" * 80)
    print("  BASELINE: Non-Strategic QA")
    print("=" * 80)
    print("\nStrategy: Answer all questions comprehensively")
    print(f"Output budget: {max_tokens} tokens\n")

    start_time = time.time()

    # Create prompt - simple, non-strategic
    questions_text = "\n".join([f"{i + 1}. {q.text}" for i, q in enumerate(QUESTIONS)])

    prompt = f"""Read this document and answer all the questions below.

DOCUMENT:
{DOCUMENT}

QUESTIONS:
{questions_text}

Please answer each question. Number your answers 1-15."""

    # Make LLM call
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    elapsed = time.time() - start_time

    # Extract usage
    usage = response.get("usage", {})
    tokens = usage.get("total_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    # Parse answers from response
    answer_text = ""
    if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
        choice = response.choices[0]
        if hasattr(choice, "message"):
            message = choice.message
            if hasattr(message, "content") and message.content:
                answer_text = message.content
            elif hasattr(message, "model_dump"):
                msg_dict = message.model_dump()
                answer_text = msg_dict.get("content") or ""

    if not answer_text:
        answer_text = ""
        print(
            f"  ⚠️  WARNING: No text output received (reasoning tokens: {output_tokens}, text tokens: 0)"
        )

    # Parse numbered answers from response
    parsed_answers = parse_numbered_answers(answer_text)

    # Evaluate each question
    results = []
    for i, question in enumerate(QUESTIONS):
        question_num = i + 1
        predicted = parsed_answers.get(question_num, None)
        answered = predicted is not None
        correct = evaluate_answer(predicted, question.answer) if answered else 0.0

        results.append(
            {
                "question": question.text,
                "importance": question.importance,
                "category": question.category,
                "correct": correct,
                "answered": answered,
            }
        )

    metrics = calculate_weighted_accuracy(results)

    print("📊 Results:")
    print(f"  Total Tokens:       {tokens:,}")
    print(f"  Output Tokens:      {output_tokens:,}")
    print(f"  Time:               {elapsed:.2f}s")
    print(f"  Weighted Accuracy:  {metrics['weighted_accuracy']:.1%}")
    print(f"    - Critical (3x):  {metrics['critical_accuracy']:.1%}")
    print(f"    - Important (2x): {metrics['important_accuracy']:.1%}")
    print(f"    - Nice-to-have:   {metrics['nice_to_have_accuracy']:.1%}")
    print(f"  Questions Answered: {metrics['answered_questions']}/{metrics['total_questions']}")

    quality_per_token = metrics["weighted_accuracy"] / output_tokens if output_tokens > 0 else 0
    print(f"  Quality/Token:      {quality_per_token:.6f}")

    return {
        "condition": "baseline",
        "tokens": tokens,
        "output_tokens": output_tokens,
        "time": elapsed,
        "metrics": metrics,
        "quality_per_token": quality_per_token,
        "results": results,
    }


def run_qa_budget_aware(
    max_tokens: int,
    model: str = "gemini/gemini-2.5-flash-preview-09-2025",
) -> dict[str, Any]:
    """Run budget-aware QA - strategically prioritizes high-value questions.

    Strategy: Explicitly prioritize critical questions first, be concise.
    Uses same max_tokens as baseline to show efficiency through better strategy.
    """
    print("\n" + "=" * 80)
    print("  BUDGET-AWARE: Strategic Prioritization")
    print("=" * 80)
    print("\nStrategy: Prioritize critical (3x) > important (2x) > nice-to-have (1x)")
    print(f"Output budget: {max_tokens} tokens (SAME as baseline)\n")

    # Create contract for monitoring (not strict enforcement)
    contract = Contract(
        id="qa-budget-aware",
        name="Prioritized QA - Budget Aware",
        resources=ResourceConstraints(tokens=10000),  # High limit, just for monitoring
    )

    # Use lenient mode - monitoring only, no hard stops
    llm = ContractedLLM(contract, strict_mode=False)

    start_time = time.time()

    # Create strategic prompt - emphasizes prioritization and conciseness
    # IMPORTANT: Use consistent 1-15 numbering matching QUESTIONS list order
    critical_qs = [(i, q) for i, q in enumerate(QUESTIONS) if q.importance == 3]
    important_qs = [(i, q) for i, q in enumerate(QUESTIONS) if q.importance == 2]
    nice_qs = [(i, q) for i, q in enumerate(QUESTIONS) if q.importance == 1]

    critical_text = "\n".join([f"{i + 1}. {q.text}" for i, q in critical_qs])
    important_text = "\n".join([f"{i + 1}. {q.text}" for i, q in important_qs])
    nice_text = "\n".join([f"{i + 1}. {q.text}" for i, q in nice_qs])

    prompt = f"""Read this document and answer the questions below. You have limited response space, so prioritize strategically.

DOCUMENT:
{DOCUMENT}

QUESTIONS BY PRIORITY:

CRITICAL (3x weight - answer these FIRST):
{critical_text}

IMPORTANT (2x weight - answer if space allows):
{important_text}

NICE-TO-HAVE (1x weight - answer if space allows):
{nice_text}

STRATEGY:
1. Answer ALL critical questions first (they're worth 3x)
2. Then answer important questions if you have space (2x)
3. Finally nice-to-have questions if space remains (1x)
4. Be concise - focus on accuracy over elaboration

Number your answers 1-15 based on which questions you answer."""

    try:
        # Make LLM call with contract monitoring
        response = llm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )

        elapsed = time.time() - start_time

        # Extract usage
        summary = llm.get_usage_summary()
        tokens = summary["usage"]["tokens"]

        # Get output tokens from response
        usage = response.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)

        # Parse answers from response
        answer_text = ""
        if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                message = choice.message
                if hasattr(message, "content") and message.content:
                    answer_text = message.content
                elif hasattr(message, "model_dump"):
                    msg_dict = message.model_dump()
                    answer_text = msg_dict.get("content") or ""

        if not answer_text:
            answer_text = ""
            print(
                f"  ⚠️  WARNING: No text output received (reasoning tokens: {output_tokens}, text tokens: 0)"
            )

        # Parse numbered answers from response
        parsed_answers = parse_numbered_answers(answer_text)

        # Map question numbers - critical (1-5), important (6-10), nice (11-15)
        # But model might number them differently based on prioritization
        # For now, try to match by order they appear in QUESTIONS
        results = []
        for i, question in enumerate(QUESTIONS):
            question_num = i + 1
            predicted = parsed_answers.get(question_num, None)
            answered = predicted is not None
            correct = evaluate_answer(predicted, question.answer) if answered else 0.0

            results.append(
                {
                    "question": question.text,
                    "importance": question.importance,
                    "category": question.category,
                    "correct": correct,
                    "answered": answered,
                }
            )

        metrics = calculate_weighted_accuracy(results)

        print("📊 Results:")
        print(f"  Total Tokens:       {tokens:,}")
        print(f"  Output Tokens:      {output_tokens:,}")
        print(f"  Time:               {elapsed:.2f}s")
        print(f"  Weighted Accuracy:  {metrics['weighted_accuracy']:.1%}")
        print(f"    - Critical (3x):  {metrics['critical_accuracy']:.1%}")
        print(f"    - Important (2x): {metrics['important_accuracy']:.1%}")
        print(f"    - Nice-to-have:   {metrics['nice_to_have_accuracy']:.1%}")
        print(f"  Questions Answered: {metrics['answered_questions']}/{metrics['total_questions']}")

        quality_per_token = metrics["weighted_accuracy"] / output_tokens if output_tokens > 0 else 0
        print(f"  Quality/Token:      {quality_per_token:.6f}")

        return {
            "condition": "budget-aware",
            "tokens": tokens,
            "output_tokens": output_tokens,
            "time": elapsed,
            "metrics": metrics,
            "quality_per_token": quality_per_token,
            "results": results,
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        elapsed = time.time() - start_time

        return {
            "condition": "budget-aware",
            "tokens": 0,
            "output_tokens": 0,
            "time": elapsed,
            "metrics": calculate_weighted_accuracy([]),
            "quality_per_token": 0,
            "error": str(e),
        }


def print_comparison(baseline: dict[str, Any], budget_aware: dict[str, Any]) -> None:
    """Print comprehensive comparison of results."""
    print("\n" + "=" * 80)
    print("  📊 EFFICIENCY COMPARISON")
    print("=" * 80)
    print("\nKey: SAME max_tokens budget, DIFFERENT strategies\n")

    print("🎯 Weighted Accuracy:")
    print(f"  Baseline (non-strategic):    {baseline['metrics']['weighted_accuracy']:.1%}")
    print(f"  Budget-Aware (strategic):    {budget_aware['metrics']['weighted_accuracy']:.1%}")

    acc_diff = (
        budget_aware["metrics"]["weighted_accuracy"] - baseline["metrics"]["weighted_accuracy"]
    )
    acc_symbol = "📈" if acc_diff > 0 else "📉" if acc_diff < 0 else "➡️"
    print(f"  Difference:                  {acc_diff:+.1%} {acc_symbol}")

    print("\n💰 Output Token Usage:")
    print(f"  Baseline:                    {baseline['output_tokens']:,} tokens")
    print(f"  Budget-Aware:                {budget_aware['output_tokens']:,} tokens")

    token_diff = budget_aware["output_tokens"] - baseline["output_tokens"]
    token_symbol = "📈" if token_diff > 0 else "📉" if token_diff < 0 else "➡️"
    print(f"  Difference:                  {token_diff:+,} tokens {token_symbol}")

    print("\n⚡ Efficiency (Weighted Accuracy per Output Token):")
    baseline_eff = baseline["quality_per_token"]
    budget_eff = budget_aware["quality_per_token"]
    print(f"  Baseline:                    {baseline_eff:.6f}")
    print(f"  Budget-Aware:                {budget_eff:.6f}")

    improvement = ((budget_eff - baseline_eff) / baseline_eff * 100) if baseline_eff > 0 else 0
    eff_symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
    print(f"  Improvement:                 {improvement:+.1f}% {eff_symbol}")

    print("\n📋 Question Priority Performance:")
    print(f"  {'Strategy':20} {'Critical':>10} {'Important':>10} {'Nice-to-have':>12}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 12}")

    print(
        f"  {'Baseline':20} "
        f"{baseline['metrics']['critical_accuracy']:>9.1%} "
        f"{baseline['metrics']['important_accuracy']:>9.1%} "
        f"{baseline['metrics']['nice_to_have_accuracy']:>11.1%}"
    )

    print(
        f"  {'Budget-Aware':20} "
        f"{budget_aware['metrics']['critical_accuracy']:>9.1%} "
        f"{budget_aware['metrics']['important_accuracy']:>9.1%} "
        f"{budget_aware['metrics']['nice_to_have_accuracy']:>11.1%}"
    )

    print("\n💡 Key Findings:")

    if improvement >= 30:
        print(f"  ✅ H3 VALIDATED: {improvement:.1f}% efficiency gain (target: ≥30%)")
        print("  • Budget-aware agent achieves significantly better quality-per-token")
        print("  • Strategic prioritization focuses on high-value questions")
        print("  • Same max_tokens budget, superior efficiency through better strategy")
    elif improvement > 0:
        print(f"  ✅ Efficiency gain: {improvement:.1f}%")
        print(
            f"  • Budget-aware agent used {baseline['output_tokens'] - budget_aware['output_tokens']} fewer output tokens"
        )
        print(
            f"  • Achieved same quality ({baseline['metrics']['weighted_accuracy']:.0%}) with less verbosity"
        )
        print("  • Demonstrates strategic conciseness and resource-aware behavior")
        if (
            baseline["metrics"]["weighted_accuracy"]
            == budget_aware["metrics"]["weighted_accuracy"]
            == 1.0
        ):
            print(
                "  • Note: Both strategies achieved 100% - task may benefit from tighter constraints"
            )
    else:
        print(f"  ⚠️  No efficiency gain observed: {improvement:.1f}%")
        print("  • May need to adjust max_tokens or prompting strategy")
        print("  • Model may be answering all questions regardless of strategy")


def main() -> None:
    """Run Phase 1 QA benchmark demonstrating contract value."""
    print("=" * 80)
    print("  Agent Contracts - Phase 1 Value Demonstration")
    print("  Prioritized Multi-Question Answering")
    print("=" * 80)

    print("""
This benchmark demonstrates the core value of Agent Contracts:

🎯 Hypothesis (H3): Budget-aware agents achieve ≥30% better quality-per-token

Approach:
- SAME output token budget (max_tokens) for both baseline and budget-aware
- DIFFERENT strategies: baseline tries to answer all, budget-aware prioritizes
- Task: Answer 15 weighted questions about quantum computing
- Questions: 5 critical (3x), 5 important (2x), 5 nice-to-have (1x)
- Metric: Weighted accuracy per output token

Expected Result:
- Budget-aware strategically focuses on high-value questions first
- Achieves HIGHER weighted accuracy with same token budget
- Proves efficiency through better strategy, not just hard limits
""")

    # Configuration
    # Note: Gemini 2.5 Flash uses internal reasoning tokens + text tokens
    # We need enough budget for BOTH reasoning and actual text output
    # With reasoning models, max_tokens must be generous enough for both phases
    max_output_tokens = 2000  # Allows reasoning + text generation

    # Run baseline (non-strategic)
    baseline = run_qa_baseline(max_tokens=max_output_tokens)

    # Run budget-aware (strategic prioritization)
    budget_aware = run_qa_budget_aware(max_tokens=max_output_tokens)

    # Print comparison
    print_comparison(baseline, budget_aware)

    print("\n" + "=" * 80)
    print("  ✅ Phase 1 Value Demonstration Complete")
    print("=" * 80)

    print("""
If efficiency gain ≥30%, we have validated H3:
Budget-aware agents achieve better quality-per-token through strategic
resource allocation - SAME budget, BETTER strategy, SUPERIOR results.

This proves the core value proposition of Agent Contracts! 🎉
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
