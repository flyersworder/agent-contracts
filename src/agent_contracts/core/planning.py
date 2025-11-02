"""Strategic planning utilities for contract-governed agents.

This module implements strategic planning capabilities that enable agents to make
mode-aware decisions about resource allocation, task prioritization, and quality-cost-time
tradeoffs.

Key features:
- Resource allocation across subtasks
- Task prioritization based on contract mode
- Quality-cost-time tradeoff analysis
- Strategic recommendations based on budget state
"""

from dataclasses import dataclass
from enum import Enum

from agent_contracts.core.contract import Contract, ContractMode
from agent_contracts.core.monitor import ResourceUsage


class TaskPriority(Enum):
    """Priority levels for tasks."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Task:
    """Represents a task with resource requirements and metadata.

    Attributes:
        id: Unique task identifier
        name: Human-readable task name
        estimated_tokens: Estimated token consumption
        estimated_time: Estimated execution time (seconds)
        estimated_quality: Expected output quality (0.0 to 1.0)
        priority: Task priority level
        required: Whether task is required for success
    """

    id: str
    name: str
    estimated_tokens: int = 0
    estimated_time: float = 0.0
    estimated_quality: float = 0.8
    priority: TaskPriority = TaskPriority.MEDIUM
    required: bool = False


@dataclass
class ResourceAllocation:
    """Represents resource allocation for a task.

    Attributes:
        task_id: Task identifier
        allocated_tokens: Tokens allocated to this task
        allocated_time: Time allocated to this task (seconds)
        expected_quality: Expected quality with this allocation
    """

    task_id: str
    allocated_tokens: int
    allocated_time: float
    expected_quality: float


@dataclass
class StrategyRecommendation:
    """Strategic recommendation for agent execution.

    Attributes:
        mode: The contract mode
        budget_utilization: Current budget utilization (0.0 to 1.0)
        recommended_approach: Description of recommended approach
        risk_level: Risk assessment (low/medium/high)
        should_continue: Whether execution should continue
        warnings: List of warnings or concerns
    """

    mode: ContractMode
    budget_utilization: float
    recommended_approach: str
    risk_level: str
    should_continue: bool
    warnings: list[str]


def plan_resource_allocation(
    tasks: list[Task],
    contract: Contract,
    current_usage: ResourceUsage | None = None,
) -> list[ResourceAllocation]:
    """Plan resource allocation across multiple tasks based on contract mode.

    Distributes available resources across tasks according to the contract's strategic
    mode and current budget state.

    Args:
        tasks: List of tasks to allocate resources for
        contract: The active contract with resource constraints
        current_usage: Optional current resource usage

    Returns:
        List of resource allocations for each task

    Example:
        >>> contract = Contract(
        ...     id="multi-task",
        ...     name="Multi-Task Agent",
        ...     mode=ContractMode.ECONOMICAL,
        ...     resources=ResourceConstraints(tokens=10000)
        ... )
        >>> tasks = [
        ...     Task(id="t1", name="Task 1", estimated_tokens=3000, required=True),
        ...     Task(id="t2", name="Task 2", estimated_tokens=2000, priority=TaskPriority.HIGH)
        ... ]
        >>> allocations = plan_resource_allocation(tasks, contract)
    """
    # Calculate remaining resources
    if contract.resources.tokens is None:
        # No token limit - allocate based on estimates
        return [
            ResourceAllocation(
                task_id=task.id,
                allocated_tokens=task.estimated_tokens,
                allocated_time=task.estimated_time,
                expected_quality=task.estimated_quality,
            )
            for task in tasks
        ]

    total_budget = contract.resources.tokens
    used_tokens = current_usage.tokens if current_usage else 0
    remaining_tokens = total_budget - used_tokens

    # Calculate total estimated tokens needed
    total_estimated = sum(task.estimated_tokens for task in tasks)

    # Mode-specific allocation strategy
    if contract.mode == ContractMode.URGENT:
        # URGENT: Prioritize required and high-priority tasks, may exceed estimates
        return _allocate_urgent(tasks, remaining_tokens, total_estimated)
    elif contract.mode == ContractMode.ECONOMICAL:
        # ECONOMICAL: Conservative allocation, reserve buffer
        return _allocate_economical(tasks, remaining_tokens, total_estimated)
    else:  # BALANCED
        # BALANCED: Proportional allocation with some buffer
        return _allocate_balanced(tasks, remaining_tokens, total_estimated)


def _allocate_urgent(
    tasks: list[Task], remaining_tokens: int, total_estimated: int
) -> list[ResourceAllocation]:
    """Allocate resources in URGENT mode - prioritize speed and critical tasks."""
    allocations = []

    # Sort tasks: required first, then by priority
    sorted_tasks = sorted(
        tasks,
        key=lambda t: (
            not t.required,  # Required tasks first
            t.priority.value,  # Then by priority
        ),
    )

    for task in sorted_tasks:
        # In URGENT mode, allocate generously to avoid bottlenecks
        if task.required or task.priority in (TaskPriority.CRITICAL, TaskPriority.HIGH):
            # Give 120% of estimate for critical tasks
            allocated = int(task.estimated_tokens * 1.2)
        else:
            # Give 100% for others
            allocated = task.estimated_tokens

        allocations.append(
            ResourceAllocation(
                task_id=task.id,
                allocated_tokens=allocated,
                allocated_time=task.estimated_time,
                expected_quality=max(0.85, task.estimated_quality),  # URGENT accepts 85% quality
            )
        )

    return allocations


def _allocate_economical(
    tasks: list[Task], remaining_tokens: int, total_estimated: int
) -> list[ResourceAllocation]:
    """Allocate resources in ECONOMICAL mode - minimize usage, reserve buffer."""
    allocations = []

    # Reserve 20% buffer
    usable_budget = int(remaining_tokens * 0.8)

    # Sort tasks: required first, then by priority
    sorted_tasks = sorted(
        tasks,
        key=lambda t: (
            not t.required,
            t.priority.value,
        ),
    )

    allocated_so_far = 0
    for task in sorted_tasks:
        if task.required:
            # Required tasks get minimum viable allocation (70% of estimate)
            allocated = int(task.estimated_tokens * 0.7)
        elif allocated_so_far < usable_budget:
            # Optional tasks get conservative allocation
            allocated = min(
                int(task.estimated_tokens * 0.6),
                usable_budget - allocated_so_far,
            )
        else:
            # Skip if budget exhausted
            allocated = 0

        allocated_so_far += allocated

        allocations.append(
            ResourceAllocation(
                task_id=task.id,
                allocated_tokens=allocated,
                allocated_time=task.estimated_time * 1.3,  # Allow more time to save resources
                expected_quality=task.estimated_quality * 0.95,  # Maintain quality
            )
        )

    return allocations


def _allocate_balanced(
    tasks: list[Task], remaining_tokens: int, total_estimated: int
) -> list[ResourceAllocation]:
    """Allocate resources in BALANCED mode - proportional with modest buffer."""
    allocations = []

    # Reserve 10% buffer
    usable_budget = int(remaining_tokens * 0.9)

    # If estimates fit within budget, use proportional allocation
    if total_estimated <= usable_budget:
        for task in tasks:
            allocations.append(
                ResourceAllocation(
                    task_id=task.id,
                    allocated_tokens=task.estimated_tokens,
                    allocated_time=task.estimated_time,
                    expected_quality=task.estimated_quality,
                )
            )
    else:
        # Scale down proportionally
        scale_factor = usable_budget / total_estimated

        for task in tasks:
            allocated = int(task.estimated_tokens * scale_factor)

            # Required tasks get at least 80% of estimate
            if task.required:
                allocated = max(allocated, int(task.estimated_tokens * 0.8))

            allocations.append(
                ResourceAllocation(
                    task_id=task.id,
                    allocated_tokens=allocated,
                    allocated_time=task.estimated_time,
                    expected_quality=task.estimated_quality * scale_factor,
                )
            )

    return allocations


def prioritize_tasks(
    tasks: list[Task],
    contract: Contract,
    current_usage: ResourceUsage | None = None,
) -> list[Task]:
    """Prioritize tasks based on contract mode and current budget state.

    Args:
        tasks: List of tasks to prioritize
        contract: The active contract
        current_usage: Optional current resource usage

    Returns:
        Tasks sorted by priority (highest first)

    Example:
        >>> tasks = [
        ...     Task(id="t1", name="Optional", priority=TaskPriority.LOW),
        ...     Task(id="t2", name="Critical", required=True)
        ... ]
        >>> prioritized = prioritize_tasks(tasks, contract)
        >>> prioritized[0].id  # Required task comes first
        't2'
    """
    # Calculate budget utilization
    utilization = 0.0
    if current_usage and contract.resources.tokens:
        utilization = current_usage.tokens / contract.resources.tokens

    # Define priority scores
    priority_scores = {
        TaskPriority.CRITICAL: 4,
        TaskPriority.HIGH: 3,
        TaskPriority.MEDIUM: 2,
        TaskPriority.LOW: 1,
    }

    def task_score(task: Task) -> tuple[int, int, int]:
        """Calculate sort key for task."""
        # Required tasks always come first
        required_score = 1 if task.required else 0

        # Base priority score
        base_score = priority_scores[task.priority]

        # Mode-specific adjustments
        if contract.mode == ContractMode.URGENT:
            # In URGENT mode, deprioritize low-quality tasks
            quality_penalty = 0 if task.estimated_quality >= 0.8 else -1
        elif contract.mode == ContractMode.ECONOMICAL:
            # In ECONOMICAL mode, penalize resource-intensive tasks
            resource_penalty = -1 if task.estimated_tokens > 5000 else 0
            quality_penalty = 0
        else:  # BALANCED
            quality_penalty = 0
            resource_penalty = 0

        # Under budget pressure, prioritize required and high-priority tasks
        if utilization > 0.7:
            # Boost required tasks even more
            required_score = 2 if task.required else 0

        final_score = (
            base_score
            + quality_penalty
            + (resource_penalty if contract.mode == ContractMode.ECONOMICAL else 0)
        )

        # Return tuple for sorting: (required, priority, negative_tokens)
        # Negative tokens to prefer lighter tasks when scores are equal
        return (required_score, final_score, -task.estimated_tokens)

    return sorted(tasks, key=task_score, reverse=True)


def estimate_quality_cost_time(
    approach: str,
    contract: Contract,
) -> tuple[float, int, float]:
    """Estimate quality-cost-time tradeoffs for different approaches.

    Args:
        approach: Approach name (e.g., "thorough", "quick", "minimal")
        contract: The active contract

    Returns:
        Tuple of (quality, tokens, time_seconds)

    Example:
        >>> contract = Contract(id="test", name="Test")
        >>> quality, tokens, time = estimate_quality_cost_time("thorough", contract)
        >>> quality > 0.9  # Thorough approach has high quality
        True
    """
    # Base estimates (for BALANCED mode)
    estimates = {
        "thorough": (0.95, 10000, 300.0),  # High quality, high cost, slow
        "standard": (0.85, 5000, 180.0),  # Medium quality, medium cost, medium
        "quick": (0.75, 3000, 60.0),  # Lower quality, lower cost, fast
        "minimal": (0.65, 1000, 30.0),  # Minimum viable, minimal cost, very fast
    }

    base_quality, base_tokens, base_time = estimates.get(approach, estimates["standard"])

    # Mode-specific adjustments
    if contract.mode == ContractMode.URGENT:
        # URGENT: Accept quality reduction for speed boost
        quality = base_quality * 0.9  # 10% quality penalty
        tokens = int(base_tokens * 1.2)  # 20% more tokens for speed
        time = base_time * 0.5  # 50% time reduction
    elif contract.mode == ContractMode.ECONOMICAL:
        # ECONOMICAL: Reduce tokens, accept time increase
        quality = base_quality  # Maintain quality
        tokens = int(base_tokens * 0.4)  # 60% token reduction
        time = base_time * 1.5  # 50% time increase
    else:  # BALANCED
        quality = base_quality
        tokens = base_tokens
        time = base_time

    return (quality, tokens, time)


def recommend_strategy(
    contract: Contract,
    current_usage: ResourceUsage | None = None,
) -> StrategyRecommendation:
    """Generate strategic recommendation based on contract mode and budget state.

    Args:
        contract: The active contract
        current_usage: Optional current resource usage

    Returns:
        Strategic recommendation with approach and risk assessment

    Example:
        >>> contract = Contract(
        ...     id="test",
        ...     name="Test",
        ...     mode=ContractMode.ECONOMICAL,
        ...     resources=ResourceConstraints(tokens=5000)
        ... )
        >>> usage = ResourceUsage(tokens=4000)
        >>> rec = recommend_strategy(contract, usage)
        >>> rec.risk_level
        'high'
    """
    # Calculate budget utilization
    utilization = 0.0
    if current_usage and contract.resources.tokens:
        utilization = current_usage.tokens / contract.resources.tokens

    warnings = []

    # Determine risk level
    if utilization < 0.3:
        risk_level = "low"
    elif utilization < 0.7:
        risk_level = "medium"
        warnings.append("Moderate budget consumption detected")
    else:
        risk_level = "high"
        warnings.append("High budget utilization - approaching limits")

    # Generate mode-specific recommendations
    if contract.mode == ContractMode.URGENT:
        if utilization > 0.9:
            approach = "Complete core objectives immediately, skip all optional tasks"
            should_continue = True
            warnings.append("Budget critical but URGENT mode - push to completion")
        elif utilization > 0.7:
            approach = "Focus on required tasks, use fastest available methods"
            should_continue = True
        else:
            approach = "Prioritize speed using parallel operations and caching"
            should_continue = True

    elif contract.mode == ContractMode.ECONOMICAL:
        if utilization > 0.8:
            approach = "Enter strict conservation mode - parametric knowledge only, no API calls"
            should_continue = False
            warnings.append("Budget exceeded threshold for ECONOMICAL mode")
        elif utilization > 0.6:
            approach = "Reduce resource usage - batch operations, minimal API calls"
            should_continue = True
            warnings.append("Approaching budget limits in ECONOMICAL mode")
        else:
            approach = "Maintain efficiency - batch operations, leverage caching"
            should_continue = True

    else:  # BALANCED
        if utilization > 0.85:
            approach = "Reduce scope - focus on highest-value tasks only"
            should_continue = True
            warnings.append("May need to deliver partial results")
        elif utilization > 0.7:
            approach = "Monitor usage closely - balance quality and efficiency"
            should_continue = True
        else:
            approach = "Standard execution - balanced resource allocation"
            should_continue = True

    return StrategyRecommendation(
        mode=contract.mode,
        budget_utilization=utilization,
        recommended_approach=approach,
        risk_level=risk_level,
        should_continue=should_continue,
        warnings=warnings,
    )
