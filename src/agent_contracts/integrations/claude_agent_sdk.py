"""Claude Agent SDK integration for Agent Contracts.

This module provides contract-aware wrappers for Claude Agent SDK agents,
enabling resource governance, per-tool enforcement, temporal constraints,
and audit trails via the SDK's hook system.

All SDK features (tools, MCP servers, subagents, skills, permissions)
remain fully available — the contract wraps on top, not replacing anything.

Example:
    >>> from agent_contracts import Contract, ResourceConstraints
    >>> from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent
    >>>
    >>> contract = Contract(
    ...     id="my-agent",
    ...     resources=ResourceConstraints(tokens=50000, cost_usd=2.0, iterations=10)
    ... )
    >>> contracted = ContractedClaudeAgent(
    ...     contract=contract,
    ...     prompt="Review auth.py",
    ... )
    >>> result = await contracted.aexecute()
"""

from typing import Any

try:
    from claude_agent_sdk import ClaudeAgentOptions

    CLAUDE_AGENT_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_AGENT_SDK_AVAILABLE = False
    ClaudeAgentOptions = Any  # type: ignore


class ContractedClaudeAgent:
    """Placeholder — implemented in subsequent tasks."""

    pass
