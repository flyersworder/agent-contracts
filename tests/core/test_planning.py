"""Unit tests for strategic planning utilities."""

from agent_contracts.core import (
    Contract,
    ContractMode,
    ResourceConstraints,
    ResourceUsage,
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


class TestTask:
    """Tests for Task dataclass."""

    def test_create_minimal_task(self) -> None:
        """Test creating a task with minimal fields."""
        task = Task(id="t1", name="Task 1")
        assert task.id == "t1"
        assert task.name == "Task 1"
        assert task.estimated_tokens == 0
        assert task.estimated_time == 0.0
        assert task.estimated_quality == 0.8
        assert task.priority == TaskPriority.MEDIUM
        assert task.required is False

    def test_create_full_task(self) -> None:
        """Test creating a task with all fields."""
        task = Task(
            id="t2",
            name="Critical Task",
            estimated_tokens=5000,
            estimated_time=120.0,
            estimated_quality=0.95,
            priority=TaskPriority.CRITICAL,
            required=True,
        )
        assert task.id == "t2"
        assert task.estimated_tokens == 5000
        assert task.estimated_time == 120.0
        assert task.estimated_quality == 0.95
        assert task.priority == TaskPriority.CRITICAL
        assert task.required is True


class TestPlanResourceAllocation:
    """Tests for plan_resource_allocation function."""

    def test_allocation_without_token_limit(self) -> None:
        """Test allocation when contract has no token limit."""
        contract = Contract(id="test", name="Test")  # No resource constraints
        tasks = [
            Task(id="t1", name="Task 1", estimated_tokens=1000),
            Task(id="t2", name="Task 2", estimated_tokens=2000),
        ]

        allocations = plan_resource_allocation(tasks, contract)

        # Should allocate exactly as estimated
        assert len(allocations) == 2
        assert allocations[0].allocated_tokens == 1000
        assert allocations[1].allocated_tokens == 2000

    def test_urgent_mode_allocation(self) -> None:
        """Test resource allocation in URGENT mode."""
        contract = Contract(
            id="urgent-test",
            name="Urgent Test",
            mode=ContractMode.URGENT,
            resources=ResourceConstraints(tokens=10000),
        )
        tasks = [
            Task(id="t1", name="Required Task", estimated_tokens=3000, required=True),
            Task(id="t2", name="High Priority", estimated_tokens=2000, priority=TaskPriority.HIGH),
            Task(id="t3", name="Low Priority", estimated_tokens=1000, priority=TaskPriority.LOW),
        ]

        allocations = plan_resource_allocation(tasks, contract)

        # URGENT mode allocates generously (120% for critical, 100% for others)
        assert allocations[0].allocated_tokens == int(3000 * 1.2)  # Required: 120%
        assert allocations[1].allocated_tokens == int(2000 * 1.2)  # High: 120%
        assert allocations[2].allocated_tokens == 1000  # Low: 100%

        # Quality should be at least 85% in URGENT mode
        assert all(a.expected_quality >= 0.85 for a in allocations)

    def test_economical_mode_allocation(self) -> None:
        """Test resource allocation in ECONOMICAL mode."""
        contract = Contract(
            id="eco-test",
            name="Economical Test",
            mode=ContractMode.ECONOMICAL,
            resources=ResourceConstraints(tokens=10000),
        )
        tasks = [
            Task(
                id="t1",
                name="Required Task",
                estimated_tokens=4000,
                estimated_time=100.0,
                required=True,
            ),
            Task(
                id="t2",
                name="Optional Task",
                estimated_tokens=3000,
                estimated_time=80.0,
                required=False,
            ),
        ]

        allocations = plan_resource_allocation(tasks, contract)

        # ECONOMICAL mode reserves 20% buffer and allocates conservatively
        # (usable budget would be 10000 * 0.8 = 8000)

        # Required task gets 70% of estimate
        assert allocations[0].allocated_tokens == int(4000 * 0.7)

        # Optional task gets 60% of estimate
        assert allocations[1].allocated_tokens == int(3000 * 0.6)

        # Time should be extended in ECONOMICAL mode (1.3x)
        assert allocations[0].allocated_time == tasks[0].estimated_time * 1.3

    def test_balanced_mode_allocation(self) -> None:
        """Test resource allocation in BALANCED mode."""
        contract = Contract(
            id="balanced-test",
            name="Balanced Test",
            mode=ContractMode.BALANCED,
            resources=ResourceConstraints(tokens=10000),
        )
        tasks = [
            Task(id="t1", name="Task 1", estimated_tokens=3000),
            Task(id="t2", name="Task 2", estimated_tokens=2000),
        ]

        allocations = plan_resource_allocation(tasks, contract)

        # BALANCED mode allocates proportionally when estimates fit
        assert allocations[0].allocated_tokens == 3000
        assert allocations[1].allocated_tokens == 2000

    def test_balanced_mode_allocation_over_budget(self) -> None:
        """Test BALANCED mode allocation when estimates exceed budget."""
        contract = Contract(
            id="balanced-test",
            name="Balanced Test",
            mode=ContractMode.BALANCED,
            resources=ResourceConstraints(tokens=5000),
        )
        tasks = [
            Task(id="t1", name="Required Task", estimated_tokens=6000, required=True),
            Task(id="t2", name="Optional Task", estimated_tokens=4000, required=False),
        ]

        allocations = plan_resource_allocation(tasks, contract)

        # Total estimated: 10000, but only 5000 available (after 10% buffer: 4500)
        # Should scale down but ensure required tasks get at least 80%
        assert allocations[0].allocated_tokens >= int(6000 * 0.8)

    def test_allocation_with_current_usage(self) -> None:
        """Test allocation when some budget is already used."""
        contract = Contract(
            id="test",
            name="Test",
            mode=ContractMode.BALANCED,
            resources=ResourceConstraints(tokens=10000),
        )
        current_usage = ResourceUsage(tokens=3000)
        tasks = [
            Task(id="t1", name="Task 1", estimated_tokens=2000),
            Task(id="t2", name="Task 2", estimated_tokens=2000),
        ]

        allocations = plan_resource_allocation(tasks, contract, current_usage)

        # Should allocate from remaining budget (10000 - 3000 = 7000)
        # After 10% buffer: 6300
        total_allocated = sum(a.allocated_tokens for a in allocations)
        assert total_allocated <= 7000


class TestPrioritizeTasks:
    """Tests for prioritize_tasks function."""

    def test_required_tasks_first(self) -> None:
        """Test that required tasks always come first."""
        contract = Contract(id="test", name="Test")
        tasks = [
            Task(id="t1", name="Optional Low", priority=TaskPriority.LOW),
            Task(id="t2", name="Required Medium", priority=TaskPriority.MEDIUM, required=True),
            Task(id="t3", name="Optional High", priority=TaskPriority.HIGH),
        ]

        prioritized = prioritize_tasks(tasks, contract)

        # Required task should be first regardless of priority
        assert prioritized[0].id == "t2"

    def test_priority_ordering(self) -> None:
        """Test that tasks are ordered by priority when none are required."""
        contract = Contract(id="test", name="Test")
        tasks = [
            Task(id="t1", name="Low", priority=TaskPriority.LOW),
            Task(id="t2", name="Critical", priority=TaskPriority.CRITICAL),
            Task(id="t3", name="High", priority=TaskPriority.HIGH),
            Task(id="t4", name="Medium", priority=TaskPriority.MEDIUM),
        ]

        prioritized = prioritize_tasks(tasks, contract)

        # Should be ordered: CRITICAL, HIGH, MEDIUM, LOW
        assert prioritized[0].priority == TaskPriority.CRITICAL
        assert prioritized[1].priority == TaskPriority.HIGH
        assert prioritized[2].priority == TaskPriority.MEDIUM
        assert prioritized[3].priority == TaskPriority.LOW

    def test_urgent_mode_deprioritizes_low_quality(self) -> None:
        """Test URGENT mode deprioritizes low-quality tasks."""
        contract = Contract(
            id="test",
            name="Test",
            mode=ContractMode.URGENT,
        )
        tasks = [
            Task(id="t1", name="High quality", priority=TaskPriority.MEDIUM, estimated_quality=0.9),
            Task(id="t2", name="Low quality", priority=TaskPriority.MEDIUM, estimated_quality=0.5),
        ]

        prioritized = prioritize_tasks(tasks, contract)

        # High quality task should come first in URGENT mode
        assert prioritized[0].id == "t1"

    def test_economical_mode_deprioritizes_expensive(self) -> None:
        """Test ECONOMICAL mode deprioritizes resource-intensive tasks."""
        contract = Contract(
            id="test",
            name="Test",
            mode=ContractMode.ECONOMICAL,
        )
        tasks = [
            Task(id="t1", name="Expensive", priority=TaskPriority.MEDIUM, estimated_tokens=10000),
            Task(id="t2", name="Cheap", priority=TaskPriority.MEDIUM, estimated_tokens=1000),
        ]

        prioritized = prioritize_tasks(tasks, contract)

        # Cheaper task should come first in ECONOMICAL mode
        assert prioritized[0].id == "t2"

    def test_budget_pressure_boosts_required_tasks(self) -> None:
        """Test that high budget utilization further prioritizes required tasks."""
        contract = Contract(
            id="test",
            name="Test",
            resources=ResourceConstraints(tokens=10000),
        )
        current_usage = ResourceUsage(tokens=8000)  # 80% utilization
        tasks = [
            Task(id="t1", name="Required", required=True, priority=TaskPriority.LOW),
            Task(id="t2", name="High priority", priority=TaskPriority.HIGH),
        ]

        prioritized = prioritize_tasks(tasks, contract, current_usage)

        # Required task should be heavily prioritized under budget pressure
        assert prioritized[0].id == "t1"


class TestEstimateQualityCostTime:
    """Tests for estimate_quality_cost_time function."""

    def test_thorough_approach(self) -> None:
        """Test estimates for thorough approach."""
        contract = Contract(id="test", name="Test", mode=ContractMode.BALANCED)

        quality, tokens, time = estimate_quality_cost_time("thorough", contract)

        # Thorough should have high quality, high cost, slow
        assert quality >= 0.9
        assert tokens >= 8000  # High token usage
        assert time >= 200  # Slower

    def test_minimal_approach(self) -> None:
        """Test estimates for minimal approach."""
        contract = Contract(id="test", name="Test", mode=ContractMode.BALANCED)

        quality, tokens, time = estimate_quality_cost_time("minimal", contract)

        # Minimal should have lower quality, low cost, fast
        assert quality <= 0.7
        assert tokens <= 1500  # Low token usage
        assert time <= 50  # Fast

    def test_urgent_mode_tradeoffs(self) -> None:
        """Test that URGENT mode trades quality for speed."""
        contract_urgent = Contract(id="test", name="Test", mode=ContractMode.URGENT)
        contract_balanced = Contract(id="test2", name="Test2", mode=ContractMode.BALANCED)

        quality_u, tokens_u, time_u = estimate_quality_cost_time("standard", contract_urgent)
        quality_b, tokens_b, time_b = estimate_quality_cost_time("standard", contract_balanced)

        # URGENT: Lower quality, more tokens, less time
        assert quality_u < quality_b
        assert tokens_u > tokens_b
        assert time_u < time_b

    def test_economical_mode_tradeoffs(self) -> None:
        """Test that ECONOMICAL mode trades speed for cost savings."""
        contract_eco = Contract(id="test", name="Test", mode=ContractMode.ECONOMICAL)
        contract_balanced = Contract(id="test2", name="Test2", mode=ContractMode.BALANCED)

        quality_e, tokens_e, time_e = estimate_quality_cost_time("standard", contract_eco)
        quality_b, tokens_b, time_b = estimate_quality_cost_time("standard", contract_balanced)

        # ECONOMICAL: Same quality, fewer tokens, more time
        assert quality_e == quality_b
        assert tokens_e < tokens_b
        assert time_e > time_b

    def test_unknown_approach_defaults_to_standard(self) -> None:
        """Test that unknown approach defaults to standard."""
        contract = Contract(id="test", name="Test", mode=ContractMode.BALANCED)

        quality1, tokens1, time1 = estimate_quality_cost_time("unknown", contract)
        quality2, tokens2, time2 = estimate_quality_cost_time("standard", contract)

        assert quality1 == quality2
        assert tokens1 == tokens2
        assert time1 == time2


class TestRecommendStrategy:
    """Tests for recommend_strategy function."""

    def test_low_utilization_balanced(self) -> None:
        """Test recommendation at low utilization in BALANCED mode."""
        contract = Contract(
            id="test",
            name="Test",
            mode=ContractMode.BALANCED,
            resources=ResourceConstraints(tokens=10000),
        )
        usage = ResourceUsage(tokens=2000)  # 20% utilization

        rec = recommend_strategy(contract, usage)

        assert rec.mode == ContractMode.BALANCED
        assert rec.budget_utilization == 0.2
        assert rec.risk_level == "low"
        assert rec.should_continue is True
        assert len(rec.warnings) == 0
        assert "standard" in rec.recommended_approach.lower()

    def test_moderate_utilization(self) -> None:
        """Test recommendation at moderate utilization."""
        contract = Contract(
            id="test",
            name="Test",
            mode=ContractMode.BALANCED,
            resources=ResourceConstraints(tokens=10000),
        )
        usage = ResourceUsage(tokens=5000)  # 50% utilization

        rec = recommend_strategy(contract, usage)

        assert rec.budget_utilization == 0.5
        assert rec.risk_level == "medium"
        assert rec.should_continue is True
        assert len(rec.warnings) >= 1
        assert "moderate" in rec.warnings[0].lower()

    def test_high_utilization_balanced(self) -> None:
        """Test recommendation at high utilization in BALANCED mode."""
        contract = Contract(
            id="test",
            name="Test",
            mode=ContractMode.BALANCED,
            resources=ResourceConstraints(tokens=10000),
        )
        usage = ResourceUsage(tokens=9000)  # 90% utilization

        rec = recommend_strategy(contract, usage)

        assert rec.budget_utilization == 0.9
        assert rec.risk_level == "high"
        assert rec.should_continue is True
        assert len(rec.warnings) >= 1
        assert (
            "scope" in rec.recommended_approach.lower()
            or "partial" in rec.recommended_approach.lower()
        )

    def test_urgent_mode_high_utilization(self) -> None:
        """Test URGENT mode recommendation at high utilization."""
        contract = Contract(
            id="test",
            name="Test",
            mode=ContractMode.URGENT,
            resources=ResourceConstraints(tokens=10000),
        )
        usage = ResourceUsage(tokens=9500)  # 95% utilization

        rec = recommend_strategy(contract, usage)

        assert rec.mode == ContractMode.URGENT
        assert rec.should_continue is True  # URGENT keeps going
        assert (
            "core" in rec.recommended_approach.lower()
            or "immediately" in rec.recommended_approach.lower()
        )

    def test_economical_mode_high_utilization(self) -> None:
        """Test ECONOMICAL mode recommendation at high utilization."""
        contract = Contract(
            id="test",
            name="Test",
            mode=ContractMode.ECONOMICAL,
            resources=ResourceConstraints(tokens=10000),
        )
        usage = ResourceUsage(tokens=8500)  # 85% utilization

        rec = recommend_strategy(contract, usage)

        assert rec.mode == ContractMode.ECONOMICAL
        assert rec.should_continue is False  # ECONOMICAL stops early
        assert "conservation" in rec.recommended_approach.lower()
        assert len(rec.warnings) >= 1

    def test_economical_mode_moderate_utilization(self) -> None:
        """Test ECONOMICAL mode recommendation at moderate utilization."""
        contract = Contract(
            id="test",
            name="Test",
            mode=ContractMode.ECONOMICAL,
            resources=ResourceConstraints(tokens=10000),
        )
        usage = ResourceUsage(tokens=6500)  # 65% utilization

        rec = recommend_strategy(contract, usage)

        assert rec.should_continue is True
        assert (
            "batch" in rec.recommended_approach.lower()
            or "minimal" in rec.recommended_approach.lower()
        )

    def test_no_usage_provided(self) -> None:
        """Test recommendation when no usage is provided."""
        contract = Contract(
            id="test",
            name="Test",
            mode=ContractMode.BALANCED,
            resources=ResourceConstraints(tokens=10000),
        )

        rec = recommend_strategy(contract, None)

        # Should assume 0% utilization
        assert rec.budget_utilization == 0.0
        assert rec.risk_level == "low"
        assert rec.should_continue is True

    def test_no_token_constraint(self) -> None:
        """Test recommendation when contract has no token constraint."""
        contract = Contract(
            id="test",
            name="Test",
            mode=ContractMode.BALANCED,
            # No resource constraints
        )
        usage = ResourceUsage(tokens=5000)

        rec = recommend_strategy(contract, usage)

        # Should default to 0% utilization
        assert rec.budget_utilization == 0.0
        assert rec.risk_level == "low"


class TestResourceAllocation:
    """Tests for ResourceAllocation dataclass."""

    def test_create_allocation(self) -> None:
        """Test creating a resource allocation."""
        allocation = ResourceAllocation(
            task_id="t1",
            allocated_tokens=5000,
            allocated_time=120.0,
            expected_quality=0.85,
        )

        assert allocation.task_id == "t1"
        assert allocation.allocated_tokens == 5000
        assert allocation.allocated_time == 120.0
        assert allocation.expected_quality == 0.85


class TestStrategyRecommendation:
    """Tests for StrategyRecommendation dataclass."""

    def test_create_recommendation(self) -> None:
        """Test creating a strategy recommendation."""
        rec = StrategyRecommendation(
            mode=ContractMode.ECONOMICAL,
            budget_utilization=0.75,
            recommended_approach="Reduce usage",
            risk_level="high",
            should_continue=True,
            warnings=["Budget approaching limit"],
        )

        assert rec.mode == ContractMode.ECONOMICAL
        assert rec.budget_utilization == 0.75
        assert rec.recommended_approach == "Reduce usage"
        assert rec.risk_level == "high"
        assert rec.should_continue is True
        assert len(rec.warnings) == 1
