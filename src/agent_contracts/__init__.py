"""Agent Contracts: A Resource-Bounded Optimization Framework for Autonomous AI Systems.

This package provides formal contracts for governing autonomous AI agents through
explicit resource constraints and temporal boundaries.
"""

__version__ = "0.5.0"
__author__ = "Qing Ye"
__license__ = "Apache-2.0"

# Core contract components
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
    AllocationRecord,  # noqa: F401
    ConservationViolationError,  # noqa: F401
    ContractingCapability,
    DelegationSummary,  # noqa: F401
)

# Monitoring and enforcement
from agent_contracts.core.enforcement import (
    CheckContext,
    CheckHook,
    ContractEnforcer,
    EnforcementAction,  # noqa: F401
    EnforcementCallback,  # noqa: F401
    EnforcementEvent,
    HookResult,
)

# Execution engine
from agent_contracts.core.executor import ContractExecutionResult, ContractExecutor
from agent_contracts.core.monitor import (
    ResourceMonitor,
    ResourceUsage,
    TemporalMonitor,  # noqa: F401
    ViolationInfo,  # noqa: F401
)
from agent_contracts.core.planning import (
    ResourceAllocation,  # noqa: F401
    StrategyRecommendation,  # noqa: F401
    Task,  # noqa: F401
    TaskPriority,  # noqa: F401
    estimate_quality_cost_time,  # noqa: F401
    plan_resource_allocation,  # noqa: F401
    prioritize_tasks,  # noqa: F401
    recommend_strategy,  # noqa: F401
)
from agent_contracts.core.prompts import (
    estimate_prompt_tokens,  # noqa: F401
    generate_adaptive_instruction,  # noqa: F401
    generate_budget_prompt,  # noqa: F401
)
from agent_contracts.core.skillspec import SkillSpec
from agent_contracts.core.tokens import (
    CostEstimate,  # noqa: F401
    TokenCount,  # noqa: F401
    TokenCounter,  # noqa: F401
)
from agent_contracts.core.wrapper import (
    ContractAgent,
    ContractViolationError,
    ExecutionLog,
    ExecutionResult,
)

# Integrations
# LiteLLM integration (optional, requires litellm package)
try:
    from agent_contracts.integrations.litellm_wrapper import ContractedLLM

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    ContractedLLM = None  # type: ignore

# Contract templates
from agent_contracts.templates import (
    CodeReviewContract,  # noqa: F401
    CustomerSupportContract,  # noqa: F401
    DataAnalysisContract,  # noqa: F401
    ResearchContract,  # noqa: F401
    get_template,  # noqa: F401
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
    "LITELLM_AVAILABLE",
    "AgentSpec",
    "Capabilities",
    "CheckContext",
    "CheckHook",
    "Contract",
    "ContractAgent",
    "ContractEnforcer",
    "ContractExecutionResult",
    "ContractExecutor",
    "ContractMode",
    "ContractState",
    "ContractViolationError",
    "ContractedLLM",
    "ContractingCapability",
    "CoordinationPattern",
    "DeadlineType",
    "EnforcementEvent",
    "ExecutionConfig",
    "ExecutionLog",
    "ExecutionResult",
    "HookResult",
    "InputSpecification",
    "OutputSpecification",
    "ResourceConstraints",
    "ResourceMonitor",
    "ResourceUsage",
    "SkillSpec",
    "SuccessCriterion",
    "TemporalConstraints",
    "TerminationCondition",
]
