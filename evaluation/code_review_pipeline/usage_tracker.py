"""Token usage tracking utilities for code review pipeline.

This module provides utilities for extracting and tracking token usage
from Google ADK agent events, enabling dynamic status updates during
iterative agent loops.

Based on the approach from agent-budget-capstone project.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IterationUsage:
    """Token usage for a single iteration.

    Attributes:
        iteration: Iteration number (1-indexed)
        coder_thinking_tokens: Coder's thinking/reasoning tokens
        coder_output_tokens: Coder's output tokens
        coder_total_tokens: Coder's total tokens
        reviewer_thinking_tokens: Reviewer's thinking/reasoning tokens
        reviewer_output_tokens: Reviewer's output tokens
        reviewer_total_tokens: Reviewer's total tokens
    """

    iteration: int
    coder_thinking_tokens: int = 0
    coder_output_tokens: int = 0
    coder_total_tokens: int = 0
    reviewer_thinking_tokens: int = 0
    reviewer_output_tokens: int = 0
    reviewer_total_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this iteration."""
        return self.coder_total_tokens + self.reviewer_total_tokens


@dataclass
class CumulativeUsage:
    """Cumulative token usage across all iterations.

    Attributes:
        coder_tokens: Total tokens used by Coder
        reviewer_tokens: Total tokens used by Reviewer
        iterations: List of per-iteration usage
    """

    coder_tokens: int = 0
    reviewer_tokens: int = 0
    iterations: list[IterationUsage] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total tokens used by both agents."""
        return self.coder_tokens + self.reviewer_tokens

    def add_iteration(self, usage: IterationUsage) -> None:
        """Add an iteration's usage to cumulative totals.

        Args:
            usage: Usage data for the iteration
        """
        self.coder_tokens += usage.coder_total_tokens
        self.reviewer_tokens += usage.reviewer_total_tokens
        self.iterations.append(usage)


def extract_token_usage(event: Any) -> tuple[int, int]:
    """Extract thinking and output tokens from an ADK event.

    Args:
        event: Google ADK event object

    Returns:
        Tuple of (thinking_tokens, output_tokens)
    """
    thinking = 0
    output = 0

    if hasattr(event, "usage_metadata") and event.usage_metadata:
        thinking = getattr(event.usage_metadata, "thoughts_token_count", 0) or 0
        output = getattr(event.usage_metadata, "candidates_token_count", 0) or 0

        # Fallback to total_token_count if specific counts not available
        if thinking == 0 and output == 0:
            total = getattr(event.usage_metadata, "total_token_count", 0) or 0
            output = total  # Attribute all to output if breakdown unavailable

    return thinking, output


def generate_status_message(
    current_iteration: int,
    max_iterations: int,
    coder_tokens_used: int,
    reviewer_tokens_used: int,
    coder_budget: int | None = None,
    reviewer_budget: int | None = None,
) -> str:
    """Generate a status message for injection between iterations.

    This message is injected into the conversation to give agents
    visibility into their resource consumption.

    Design principles (from agent-budget-capstone):
    1. Include iteration context (temporal awareness)
    2. Show actual usage vs budget (if available)
    3. Frame remaining resources positively
    4. Keep it concise to minimize token overhead

    Args:
        current_iteration: Current iteration number (1-indexed)
        max_iterations: Maximum iterations allowed
        coder_tokens_used: Cumulative tokens used by Coder
        reviewer_tokens_used: Cumulative tokens used by Reviewer
        coder_budget: Coder's token budget (optional)
        reviewer_budget: Reviewer's token budget (optional)

    Returns:
        Formatted status message string
    """
    remaining_iterations = max_iterations - current_iteration
    total_tokens = coder_tokens_used + reviewer_tokens_used

    lines = [
        f"[STATUS: Iteration {current_iteration} of {max_iterations} complete]",
        f"Coder tokens used: {coder_tokens_used:,}",
        f"Reviewer tokens used: {reviewer_tokens_used:,}",
        f"Total tokens: {total_tokens:,}",
    ]

    # Add budget info if available
    if coder_budget is not None:
        coder_remaining = max(0, coder_budget - coder_tokens_used)
        lines.append(f"Coder budget remaining: {coder_remaining:,} tokens")

    if reviewer_budget is not None:
        reviewer_remaining = max(0, reviewer_budget - reviewer_tokens_used)
        lines.append(f"Reviewer budget remaining: {reviewer_remaining:,} tokens")

    # Iteration guidance
    if remaining_iterations == 0:
        lines.append("This is the FINAL iteration - make it count!")
    elif remaining_iterations == 1:
        lines.append("1 iteration remaining after this.")
    else:
        lines.append(f"{remaining_iterations} iterations remaining.")

    return "\n".join(lines)


def generate_coder_status_prefix(
    iteration: int,
    max_iterations: int,
    tokens_used: int,
    token_budget: int | None = None,
) -> str:
    """Generate status prefix for Coder agent message.

    Args:
        iteration: Current iteration number (1-indexed)
        max_iterations: Maximum iterations allowed
        tokens_used: Cumulative tokens used by Coder
        token_budget: Coder's token budget (optional)

    Returns:
        Status prefix to prepend to Coder's message
    """
    remaining = max_iterations - iteration + 1  # Including current

    lines = ["[ITERATION STATUS]"]
    lines.append(f"- Attempt {iteration} of {max_iterations}")
    lines.append(f"- Tokens used so far: {tokens_used:,}")

    if token_budget is not None:
        remaining_tokens = max(0, token_budget - tokens_used)
        pct_used = (tokens_used / token_budget * 100) if token_budget > 0 else 0
        lines.append(f"- Budget remaining: {remaining_tokens:,} ({100 - pct_used:.0f}%)")

    if remaining == 1:
        lines.append("- ⚠️ FINAL ATTEMPT - ensure correctness!")
    elif remaining == 2:
        lines.append("- Only 1 retry remaining after this.")

    lines.append("")  # Blank line before actual message
    return "\n".join(lines)


def generate_reviewer_status_prefix(
    iteration: int,
    max_iterations: int,
    tokens_used: int,
    token_budget: int | None = None,
) -> str:
    """Generate status prefix for Reviewer agent message.

    Args:
        iteration: Current iteration number (1-indexed)
        max_iterations: Maximum iterations allowed
        tokens_used: Cumulative tokens used by Reviewer
        token_budget: Reviewer's token budget (optional)

    Returns:
        Status prefix to prepend to Reviewer's message
    """
    remaining = max_iterations - iteration  # After this iteration

    lines = ["[ITERATION STATUS]"]
    lines.append(f"- Review {iteration} of {max_iterations}")
    lines.append(f"- Tokens used so far: {tokens_used:,}")

    if token_budget is not None:
        remaining_tokens = max(0, token_budget - tokens_used)
        lines.append(f"- Budget remaining: {remaining_tokens:,}")

    if remaining == 0:
        lines.append("- This is the FINAL review opportunity.")
    elif remaining == 1:
        lines.append("- 1 more review possible after this.")

    lines.append("")  # Blank line before actual message
    return "\n".join(lines)
