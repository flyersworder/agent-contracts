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

    # Assemble complete prompt
    prompt = f"""You are operating under a formal Agent Contract with explicit resource and time constraints.

{mode_intro}

# Task
{task_description}

{budget_section}

{temporal_section}

{strategy_section}

# Important
- Monitor your resource usage continuously
- If you cannot complete the full task within constraints, provide partial results with confidence scores
- Explain your resource allocation decisions when relevant
"""
    return prompt.strip()


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
**Optimize for speed.** You have relaxed resource constraints to enable rapid completion.
- Target: Complete in ≤50% of normal time while maintaining ≥85% quality
- Prioritize: Speed over thoroughness
- Strategy: Use parallel operations, leverage caching, provide quick approximations
""",
        ContractMode.BALANCED: """
# Execution Mode: BALANCED ⚖️
**Optimize for balance.** Maintain equilibrium between quality, cost, and time.
- Target: Optimal quality-cost-time tradeoff
- Prioritize: Balanced resource utilization
- Strategy: Standard execution with measured resource allocation
""",
        ContractMode.ECONOMICAL: """
# Execution Mode: ECONOMICAL 💰
**Optimize for cost.** Minimize resource consumption while maintaining quality.
- Target: ≥90% quality at ≤40% of normal token usage
- Prioritize: Resource efficiency over speed
- Strategy: Batch operations, use parametric knowledge, avoid expensive tools
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

    if resources.api_calls is not None:
        used_calls = current_usage.api_calls if current_usage else 0
        remaining_calls = resources.api_calls - used_calls
        lines.append(f"- API Calls: {resources.api_calls} total ({remaining_calls} remaining)")

    if resources.web_searches is not None:
        used_searches = current_usage.web_searches if current_usage else 0
        remaining_searches = resources.web_searches - used_searches
        lines.append(
            f"- Web Searches: {resources.web_searches} total ({remaining_searches} remaining)"
        )

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

    if temporal.deadline is None and temporal.max_duration is None:
        return ""

    lines = ["# Time Constraints"]

    if temporal.deadline:
        lines.append(f"- Deadline: {temporal.deadline.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- Deadline Type: {temporal.deadline_type.value.upper()}")

    if temporal.max_duration:
        hours = temporal.max_duration.total_seconds() / 3600
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

    # Add mode-specific guidance
    if contract.mode == ContractMode.URGENT:
        lines.append("- Prioritize speed: Use fastest available methods")
        lines.append("- Accept approximations: 85% accuracy is sufficient")
        lines.append("- Parallelize when possible: Don't wait for sequential operations")
    elif contract.mode == ContractMode.ECONOMICAL:
        lines.append("- Minimize API calls: Use parametric knowledge when possible")
        lines.append("- Batch operations: Consolidate multiple queries")
        lines.append("- Cache aggressively: Reuse results from previous calls")
    else:  # BALANCED
        lines.append("- Balance quality and efficiency")
        lines.append("- Use tools judiciously based on expected value")
        lines.append("- Monitor usage and adapt if approaching limits")

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
