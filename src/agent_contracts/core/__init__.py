"""Core contract components.

This module contains the fundamental data structures and enforcement mechanisms
for Agent Contracts.
"""

from agent_contracts.core.capabilities import (
    AgentSpec,
    Capabilities,
    CoordinationPattern,
    ExecutionConfig,
)
from agent_contracts.core.contract import (
    Contract,
    ContractMode,
    ContractState,
    DeadlineType,
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
    CheckContext,
    CheckHook,
    ContractEnforcer,
    EnforcementAction,
    EnforcementCallback,
    EnforcementEvent,
    HookResult,
)
from agent_contracts.core.executor import (
    ContractExecutionResult,
    ContractExecutor,
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
from agent_contracts.core.skillspec import SkillSpec
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
    ExecutionResult,
)

__all__ = [
    "AgentSpec",
    "AllocationRecord",
    "Capabilities",
    "CheckContext",
    "CheckHook",
    "ConservationViolationError",
    "Contract",
    "ContractAgent",
    "ContractEnforcer",
    "ContractExecutionResult",
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
    "HookResult",
    "InputSpecification",
    "OutputSpecification",
    "ResourceAllocation",
    "ResourceConstraints",
    "ResourceMonitor",
    "ResourceUsage",
    "SkillSpec",
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
