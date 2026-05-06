"""Integration modules for various LLM and agent frameworks.

This module contains adapters for popular frameworks and LLM providers.
"""

from agent_contracts.core.wrapper import ContractViolationError

# LiteLLM integration (optional, requires litellm package)
try:
    from agent_contracts.integrations.litellm_wrapper import ContractedLLM

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    ContractedLLM = None  # type: ignore

# LangChain integration (optional, requires langchain package)
try:
    from agent_contracts.integrations.langchain import (
        ContractedChain,
        ContractedChainLLM,
        create_contracted_chain,
    )

    # Backward-compat alias (pre-1.0)
    LangChainContractedLLM = ContractedChainLLM

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ContractedChain = None  # type: ignore
    ContractedChainLLM = None  # type: ignore
    LangChainContractedLLM = None  # type: ignore
    create_contracted_chain = None  # type: ignore

# LangGraph integration (optional, requires langgraph package)
try:
    from agent_contracts.integrations.langgraph import (
        ContractedGraph,
        create_contracted_graph,
    )

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    ContractedGraph = None  # type: ignore
    create_contracted_graph = None  # type: ignore

# Google ADK integration (optional, requires google-adk package)
try:
    from agent_contracts.integrations.google_adk import (
        GOOGLE_ADK_AVAILABLE,
        ContractedAdkAgent,
        ContractedAdkMultiAgent,
        DelegatingAdkAgent,
        create_contracted_adk_agent,
    )

    # Re-check: the module may load but google.adk may not be available
    if not GOOGLE_ADK_AVAILABLE:
        raise ImportError("google-adk not installed")
except ImportError:
    GOOGLE_ADK_AVAILABLE = False
    ContractedAdkAgent = None  # type: ignore
    ContractedAdkMultiAgent = None  # type: ignore
    DelegatingAdkAgent = None  # type: ignore
    create_contracted_adk_agent = None  # type: ignore

# Claude Agent SDK integration (optional, requires claude-agent-sdk package)
try:
    from agent_contracts.integrations.claude_agent_sdk import (
        CLAUDE_AGENT_SDK_AVAILABLE,
        ContractedClaudeAgent,
    )

    if not CLAUDE_AGENT_SDK_AVAILABLE:
        raise ImportError("claude-agent-sdk not installed")
except ImportError:
    CLAUDE_AGENT_SDK_AVAILABLE = False  # type: ignore
    ContractedClaudeAgent = None  # type: ignore

# Causal Chamber integration (optional, requires causalchamber package)
# M1 stub — see docs/causal_chamber_M1_decisions.md §2.1
try:
    from agent_contracts.integrations.causalchamber import (
        CAUSAL_CHAMBER_AVAILABLE,
        ContractedChamberAgent,
        create_contracted_chamber_agent,
    )

    if not CAUSAL_CHAMBER_AVAILABLE:
        raise ImportError("causalchamber not installed")
except ImportError:
    CAUSAL_CHAMBER_AVAILABLE = False  # type: ignore
    ContractedChamberAgent = None  # type: ignore
    create_contracted_chamber_agent = None  # type: ignore

__all__ = [
    "CAUSAL_CHAMBER_AVAILABLE",
    "CLAUDE_AGENT_SDK_AVAILABLE",
    "GOOGLE_ADK_AVAILABLE",
    "LANGCHAIN_AVAILABLE",
    "LANGGRAPH_AVAILABLE",
    "LITELLM_AVAILABLE",
    "ContractViolationError",
    "ContractedAdkAgent",
    "ContractedAdkMultiAgent",
    "ContractedChain",
    "ContractedChainLLM",
    "ContractedChamberAgent",
    "ContractedClaudeAgent",
    "ContractedGraph",
    "ContractedLLM",
    "DelegatingAdkAgent",
    "LangChainContractedLLM",
    "create_contracted_adk_agent",
    "create_contracted_chain",
    "create_contracted_chamber_agent",
    "create_contracted_graph",
]
