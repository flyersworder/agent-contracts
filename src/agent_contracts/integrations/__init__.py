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

__all__ = [
    "LANGCHAIN_AVAILABLE",
    "LANGGRAPH_AVAILABLE",
    "ContractViolationError",
    "ContractedChain",
    "ContractedGraph",
    "ContractedLLM",
    "create_contracted_chain",
    "create_contracted_graph",
]
