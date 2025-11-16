"""Integration modules for various LLM and agent frameworks.

This module contains adapters for popular frameworks and LLM providers.
"""

from agent_contracts.integrations.litellm_wrapper import (
    ContractedLLM,
    ContractViolationError,
)

# LangChain integration (optional, requires langchain package)
try:
    from agent_contracts.integrations.langchain import (
        ContractedChain,
        create_contracted_chain,
    )
    from agent_contracts.integrations.langchain import (
        ContractedLLM as LangChainContractedLLM,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ContractedChain = None  # type: ignore
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
        ContractedAdkAgent,
        ContractedAdkMultiAgent,
        create_contracted_adk_agent,
    )

    GOOGLE_ADK_AVAILABLE = True
except ImportError:
    GOOGLE_ADK_AVAILABLE = False
    ContractedAdkAgent = None  # type: ignore
    ContractedAdkMultiAgent = None  # type: ignore
    create_contracted_adk_agent = None  # type: ignore

__all__ = [
    "GOOGLE_ADK_AVAILABLE",
    "LANGCHAIN_AVAILABLE",
    "LANGGRAPH_AVAILABLE",
    "ContractViolationError",
    "ContractedAdkAgent",
    "ContractedAdkMultiAgent",
    "ContractedChain",
    "ContractedGraph",
    "ContractedLLM",
    "create_contracted_adk_agent",
    "create_contracted_chain",
    "create_contracted_graph",
]
