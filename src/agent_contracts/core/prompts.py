"""Budget-aware prompt generation for contract-governed agents.

This module implements contract-aware prompting strategies from whitepaper Section 5.4,
enabling agents to self-regulate resource usage through strategic system prompts.

Key features:
- Mode-specific prompts (URGENT/BALANCED/ECONOMICAL)
- Adaptive instructions based on budget utilization
- Resource and temporal constraint communication
- Strategic guidance for resource allocation
"""

from agent_contracts.core.contract import Contract, ContractMode
from agent_contracts.core.monitor import ResourceUsage


def generate_budget_prompt(
    contract: Contract,
    task_description: str,
    current_usage: ResourceUsage | None = None,
) -> str:
    """Generate contract-aware system prompt with budget constraints.

    Creates a comprehensive system prompt that informs the agent about its
    resource constraints, strategic mode, and provides adaptive guidance based
    on current budget utilization.

    Args:
        contract: The active contract with resource/temporal constraints
        task_description: Human-readable description of the task
        current_usage: Optional current resource usage (for adaptive prompts)

    Returns:
        System prompt string with budget awareness and strategic guidance

    Example:
        >>> contract = Contract(
        ...     id="research",
        ...     name="Research Task",
        ...     mode=ContractMode.ECONOMICAL,
        ...     resources=ResourceConstraints(tokens=5000, web_searches=3)
        ... )
        >>> prompt = generate_budget_prompt(contract, "Research electric vehicles")
        >>> "ECONOMICAL mode" in prompt
        True
    """
    # Generate mode-specific introduction
    mode_intro = _generate_mode_introduction(contract.mode)

    # Generate resource budget section
    budget_section = _generate_budget_section(contract, current_usage)

    # Generate temporal constraints section
    temporal_section = _generate_temporal_section(contract)

    # Generate strategic guidance based on mode and budget state
    strategy_section = _generate_strategic_guidance(contract, current_usage)

    # Calculate budget utilization for conditional meta-instructions
    utilization = 0.0
    if current_usage and contract.resources.tokens:
        utilization = current_usage.tokens / contract.resources.tokens

    # Assemble complete prompt
    prompt_parts = [
        "You are operating under a formal Agent Contract with explicit resource and time constraints.",
        "",
        mode_intro,
        "",
        "# Task",
        task_description,
        "",
        budget_section,
    ]

    if temporal_section:
        prompt_parts.extend(["", temporal_section])

    prompt_parts.extend(["", strategy_section])

    # Only add meta-instructions under real budget pressure (>70% utilization)
    # or for ECONOMICAL mode which requires active optimization
    if utilization > 0.7 or contract.mode == ContractMode.ECONOMICAL:
        prompt_parts.extend(
            [
                "",
                "# Important",
                "- Monitor your resource usage continuously",
                "- If you cannot complete the full task within constraints, provide partial results with confidence scores",
                "- Explain your resource allocation decisions when relevant",
            ]
        )

    return "\n".join(prompt_parts)


def _generate_mode_introduction(mode: ContractMode) -> str:
    """Generate mode-specific introduction section.

    Args:
        mode: The contract execution mode

    Returns:
        Mode-specific introduction text
    """
    intros = {
        ContractMode.URGENT: """
# Execution Mode: URGENT ⚡
**Optimize for speed.** Complete the task as quickly as possible.
- Priority: Fast response over deep analysis
- Approach: Use first-pass reasoning, trust your initial judgment
- Quality target: 85% accuracy is acceptable for faster completion
""",
        ContractMode.BALANCED: """
# Execution Mode: BALANCED ⚖️
**Optimize for quality.** Deliver the highest quality results.
- Priority: Accuracy and correctness
- Approach: Reason carefully, verify key facts before responding
- Quality target: Aim for the best possible output
""",
        ContractMode.ECONOMICAL: """
# Execution Mode: ECONOMICAL 💰
**Optimize for cost.** Minimize reasoning overhead while maintaining quality.
- Priority: Efficient thinking, avoid over-analysis
- Approach: Direct reasoning path, skip unnecessary deliberation
- Quality target: 90% quality with minimal cognitive overhead
""",
    }
    return intros[mode].strip()


def _generate_budget_section(contract: Contract, current_usage: ResourceUsage | None) -> str:
    """Generate resource budget section with usage information.

    Args:
        contract: The active contract
        current_usage: Optional current resource usage

    Returns:
        Budget section text
    """
    lines = ["# Resource Budget"]

    # Add each non-None resource constraint
    resources = contract.resources

    if resources.tokens is not None:
        used_tokens = current_usage.tokens if current_usage else 0
        remaining_tokens = resources.tokens - used_tokens
        pct = (used_tokens / resources.tokens * 100) if resources.tokens > 0 else 0
        lines.append(
            f"- Tokens: {resources.tokens:,} total ({remaining_tokens:,} remaining, {pct:.0f}% used)"
        )

    # Reasoning/thinking token budget (for models with extended thinking like Gemini 2.5)
    # Check both resources.reasoning_tokens and metadata (for delegation cases)
    reasoning_tokens = resources.reasoning_tokens
    if reasoning_tokens is None and contract.metadata:
        reasoning_tokens = contract.metadata.get("reasoning_tokens")

    if reasoning_tokens is not None:
        lines.append(
            f"- Reasoning Budget: {reasoning_tokens:,} tokens allocated for internal thinking"
        )
        lines.append("  (Use this budget wisely for planning, analysis, and careful reasoning)")

    if resources.api_calls is not None:
        used_calls = current_usage.api_calls if current_usage else 0
        remaining_calls = resources.api_calls - used_calls
        lines.append(f"- API Calls: {resources.api_calls} total ({remaining_calls} remaining)")

    # Iterations limit (maps to LLM calls in multi-turn execution)
    if resources.iterations is not None:
        lines.append(
            f"- LLM Calls: {resources.iterations} maximum (plan your reasoning steps accordingly)"
        )

    if resources.web_searches is not None:
        used_searches = current_usage.web_searches if current_usage else 0
        remaining_searches = resources.web_searches - used_searches
        lines.append(
            f"- Web Searches: {resources.web_searches} total ({remaining_searches} remaining)"
        )

    # Per-tool limits (e.g., google_search: 15)
    if resources.per_tool_limits:
        for tool_name, limit in resources.per_tool_limits.items():
            # Get usage if available
            used = 0
            if current_usage:
                used = current_usage.get_tool_usage(tool_name)
            remaining = limit - used
            # Format tool name for readability (google_search -> Google Search)
            display_name = tool_name.replace("_", " ").title()
            lines.append(f"- {display_name}: {limit} calls maximum ({remaining} remaining)")

    if resources.cost_usd is not None:
        used_cost = current_usage.cost_usd if current_usage else 0.0
        remaining_cost = resources.cost_usd - used_cost
        lines.append(f"- Cost: ${resources.cost_usd:.4f} total (${remaining_cost:.4f} remaining)")

    return "\n".join(lines)


def _generate_temporal_section(contract: Contract) -> str:
    """Generate temporal constraints section.

    Args:
        contract: The active contract

    Returns:
        Temporal section text
    """
    temporal = contract.temporal
    execution = contract.execution

    # Check if we have any temporal constraints to report
    has_deadline = temporal.deadline is not None
    has_duration = temporal.max_duration is not None
    has_timeout = execution is not None and execution.timeout_seconds is not None

    if not has_deadline and not has_duration and not has_timeout:
        return ""

    lines = ["# Time Constraints"]

    # API response timeout (hard enforcement)
    if has_timeout:
        timeout_secs = execution.timeout_seconds  # type: ignore[union-attr]
        # Add mode-aware guidance for timeout
        if contract.mode == ContractMode.URGENT:
            lines.append(
                f"- ⚡ Response Timeout: {timeout_secs:.0f} seconds "
                "(be concise to complete quickly)"
            )
        elif contract.mode == ContractMode.ECONOMICAL:
            lines.append(
                f"- 💰 Response Timeout: {timeout_secs:.0f} seconds "
                "(keep output brief to minimize tokens)"
            )
        else:  # BALANCED
            lines.append(
                f"- ⚖️ Response Timeout: {timeout_secs:.0f} seconds "
                "(take time for thorough, quality output)"
            )

    if has_deadline:
        lines.append(f"- Deadline: {temporal.deadline.strftime('%Y-%m-%d %H:%M:%S')}")  # type: ignore[union-attr]
        lines.append(f"- Deadline Type: {temporal.deadline_type.value.upper()}")

    if has_duration:
        hours = temporal.max_duration.total_seconds() / 3600  # type: ignore[union-attr]
        lines.append(f"- Maximum Duration: {hours:.1f} hours")

    return "\n".join(lines)


def _generate_strategic_guidance(contract: Contract, current_usage: ResourceUsage | None) -> str:
    """Generate strategic guidance based on mode and budget state.

    Args:
        contract: The active contract
        current_usage: Optional current resource usage

    Returns:
        Strategic guidance text
    """
    lines = ["# Strategic Guidance"]

    # Calculate budget utilization if usage is provided
    utilization = 0.0
    if current_usage and contract.resources.tokens:
        utilization = current_usage.tokens / contract.resources.tokens

    # Add mode-specific guidance (focused on reasoning approach, not output format)
    if contract.mode == ContractMode.URGENT:
        lines.append("- Think fast: Trust your first instinct")
        lines.append("- Skip deliberation: Don't second-guess yourself")
        lines.append("- Accept trade-offs: Speed matters more than perfection")
    elif contract.mode == ContractMode.ECONOMICAL:
        lines.append("- Think efficiently: Take the shortest reasoning path")
        lines.append("- Avoid over-analysis: Don't explore every angle")
        lines.append("- Be decisive: Commit to your answer quickly")
    else:  # BALANCED
        lines.append("- Think carefully: Verify key facts before answering")
        lines.append("- Consider alternatives: Explore multiple angles")
        lines.append("- Prioritize accuracy: Quality is the priority")

    # Add adaptive guidance based on utilization
    if utilization > 0:
        lines.append("\n## Budget State")
        if utilization < 0.3:
            lines.append("✅ Ample budget remaining (< 30% used). Prioritize thoroughness.")
        elif utilization < 0.7:
            lines.append("⚠️ Moderate utilization (30-70% used). Balance quality and efficiency.")
        else:
            lines.append("🚨 Low budget remaining (> 70% used). Enter conservation mode:")
            lines.append("  - Rely on parametric knowledge, avoid web searches")
            lines.append("  - Prepare to return partial results if needed")
            lines.append("  - Focus only on highest-priority aspects")

    return "\n".join(lines)


def generate_adaptive_instruction(budget_utilization: float, mode: ContractMode) -> str:
    """Generate adaptive instruction based on budget state and mode.

    Provides concise, actionable guidance that changes as budget is consumed.
    Used for dynamic prompting during multi-step agent execution.

    Args:
        budget_utilization: Fraction of budget used (0.0 to 1.0)
        mode: The contract execution mode

    Returns:
        Adaptive instruction text

    Example:
        >>> instruction = generate_adaptive_instruction(0.8, ContractMode.ECONOMICAL)
        >>> "conservation mode" in instruction.lower()
        True
    """
    if budget_utilization < 0.3:
        if mode == ContractMode.ECONOMICAL:
            return "Budget available. Still prioritize efficiency over exhaustive search."
        return "Ample budget. Prioritize quality and thoroughness."

    elif budget_utilization < 0.7:
        return "Moderate budget usage. Balance quality and efficiency carefully."

    else:
        if mode == ContractMode.URGENT:
            return "Budget low but speed is priority. Complete core task, skip refinements."
        return (
            "⚠️ Conservation mode: Minimize new resource usage. "
            "Use cached results, parametric knowledge. Prepare partial results."
        )


def estimate_prompt_tokens(prompt: str) -> int:
    """Estimate token count for a prompt string.

    Uses simple heuristic: ~4 characters per token for English text.

    Args:
        prompt: The prompt text to estimate

    Returns:
        Estimated token count

    Example:
        >>> estimate_prompt_tokens("Hello world!")
        3
    """
    # Simple heuristic: ~4 chars per token
    return len(prompt) // 4
