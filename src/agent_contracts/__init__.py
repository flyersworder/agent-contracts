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
    TemporalMonitor,
    ViolationInfo,
)
from agent_contracts.core.wrapper import (
    ContractAgent,
    ExecutionLog,
    ExecutionResult,
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
)

# Contract templates
from agent_contracts.templates import (
    CodeReviewContract,
    CustomerSupportContract,
    DataAnalysisContract,
    ResearchContract,
    get_template,
)

# Integrations
from agent_contracts.integrations.litellm_wrapper import (
    ContractedLLM,
    ContractViolationError,
)

# LangChain integration (optional)
try:
    from agent_contracts.integrations.langchain import (
        ContractedChain,
        create_contracted_chain,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ContractedChain = None  # type: ignore
    create_contracted_chain = None  # type: ignore

__all__ = [
    # Core
    "Contract",
    "ContractAgent",
    "ContractEnforcer",
    "ContractMode",
    "ContractState",
    "ContractViolationError",
    "ContractedLLM",
    "CostEstimate",
    "DeadlineType",
    "EnforcementAction",
    "EnforcementCallback",
    "EnforcementEvent",
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
    # Templates
    "CodeReviewContract",
    "CustomerSupportContract",
    "DataAnalysisContract",
    "ResearchContract",
    "get_template",
    # LangChain (optional)
    "ContractedChain",
    "create_contracted_chain",
    "LANGCHAIN_AVAILABLE",
    # Planning & Prompts
    "estimate_prompt_tokens",
    "estimate_quality_cost_time",
    "generate_adaptive_instruction",
    "generate_budget_prompt",
    "plan_resource_allocation",
    "prioritize_tasks",
    "recommend_strategy",
]
