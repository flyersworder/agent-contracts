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
        ContractedLLM as LangChainContractedLLM,
        create_contracted_chain,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ContractedChain = None  # type: ignore
    LangChainContractedLLM = None  # type: ignore
    create_contracted_chain = None  # type: ignore

__all__ = [
    "ContractViolationError",
    "ContractedLLM",
    # LangChain (if available)
    "ContractedChain",
    "create_contracted_chain",
    "LANGCHAIN_AVAILABLE",
]
