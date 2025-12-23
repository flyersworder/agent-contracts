"""Core contract components.

This module contains the fundamental data structures and enforcement mechanisms
for Agent Contracts.
"""

from agent_contracts.core.contract import (
    AgentSpec,
    Capabilities,
    Contract,
    ContractMode,
    ContractState,
    CoordinationPattern,
    DeadlineType,
    ExecutionConfig,
    InputSpecification,
    OutputSpecification,
    ResourceConstraints,
    SuccessCriterion,
    TemporalConstraints,
    TerminationCondition,
)
from agent_contracts.core.delegation import (
    AllocationRecord,
    ConservationViolationError,
    ContractingCapability,
    DelegationSummary,
)
from agent_contracts.core.enforcement import (
    ContractEnforcer,
    EnforcementAction,
    EnforcementCallback,
    EnforcementEvent,
)
from agent_contracts.core.executor import (
    ContractExecutor,
    ExecutionResult,  # Primary ExecutionResult from executor
)
from agent_contracts.core.monitor import (
    ResourceMonitor,
    ResourceUsage,
    TemporalMonitor,
    ViolationInfo,
)
from agent_contracts.core.planning import (
    ResourceAllocation,
    StrategyRecommendation,
    Task,
    TaskPriority,
    estimate_quality_cost_time,
    plan_resource_allocation,
    prioritize_tasks,
    recommend_strategy,
)
from agent_contracts.core.prompts import (
    estimate_prompt_tokens,
    generate_adaptive_instruction,
    generate_budget_prompt,
)
from agent_contracts.core.tokens import (
    CostEstimate,
    TokenCount,
    TokenCounter,
    estimate_cost,
    estimate_tokens,
)
from agent_contracts.core.wrapper import (
    ContractAgent,
    ContractViolationError,
    ExecutionLog,
)
from agent_contracts.core.wrapper import (
    ExecutionResult as WrapperExecutionResult,  # Aliased to avoid conflict with executor
)

__all__ = [
    "AgentSpec",
    "AllocationRecord",
    "Capabilities",
    "ConservationViolationError",
    "Contract",
    "ContractAgent",
    "ContractEnforcer",
    "ContractExecutor",
    "ContractMode",
    "ContractState",
    "ContractViolationError",
    "ContractingCapability",
    "CoordinationPattern",
    "CostEstimate",
    "DeadlineType",
    "DelegationSummary",
    "EnforcementAction",
    "EnforcementCallback",
    "EnforcementEvent",
    "ExecutionConfig",
    "ExecutionLog",
    "ExecutionResult",
    "InputSpecification",
    "OutputSpecification",
    "ResourceAllocation",
    "ResourceConstraints",
    "ResourceMonitor",
    "ResourceUsage",
    "StrategyRecommendation",
    "SuccessCriterion",
    "Task",
    "TaskPriority",
    "TemporalConstraints",
    "TemporalMonitor",
    "TerminationCondition",
    "TokenCount",
    "TokenCounter",
    "ViolationInfo",
    "WrapperExecutionResult",
    "estimate_cost",
    "estimate_prompt_tokens",
    "estimate_quality_cost_time",
    "estimate_tokens",
    "generate_adaptive_instruction",
    "generate_budget_prompt",
    "plan_resource_allocation",
    "prioritize_tasks",
    "recommend_strategy",
]
