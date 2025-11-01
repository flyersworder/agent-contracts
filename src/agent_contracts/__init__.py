"""Agent Contracts: A Resource-Bounded Optimization Framework for Autonomous AI Systems.

This package provides formal contracts for governing autonomous AI agents through
explicit resource constraints and temporal boundaries.
"""

__version__ = "0.1.0"
__author__ = "Qing Ye"
__license__ = "CC BY 4.0"

# Core contract components
from agent_contracts.core.contract import (
    Contract,
    ContractState,
    DeadlineType,
    InputSpecification,
    OutputSpecification,
    ResourceConstraints,
    SuccessCriterion,
    TemporalConstraints,
    TerminationCondition,
)

# Monitoring and enforcement
from agent_contracts.core.enforcement import (
    ContractEnforcer,
    EnforcementAction,
    EnforcementCallback,
    EnforcementEvent,
)
from agent_contracts.core.monitor import (
    ResourceMonitor,
    ResourceUsage,
    ViolationInfo,
)
from agent_contracts.core.tokens import (
    CostEstimate,
    TokenCount,
    TokenCounter,
)

# Integrations
from agent_contracts.integrations.litellm_wrapper import (
    ContractedLLM,
    ContractViolationError,
)

__all__ = [
    "Contract",
    "ContractEnforcer",
    "ContractState",
    "ContractViolationError",
    "ContractedLLM",
    "CostEstimate",
    "DeadlineType",
    "EnforcementAction",
    "EnforcementCallback",
    "EnforcementEvent",
    "InputSpecification",
    "OutputSpecification",
    "ResourceConstraints",
    "ResourceMonitor",
    "ResourceUsage",
    "SuccessCriterion",
    "TemporalConstraints",
    "TerminationCondition",
    "TokenCount",
    "TokenCounter",
    "ViolationInfo",
]
