"""Unit tests for core contract data structures."""

from datetime import datetime, timedelta

import pytest

from agent_contracts.core import (
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


class TestResourceConstraints:
    """Tests for ResourceConstraints validation and creation."""

    def test_create_empty_constraints(self) -> None:
        """Test creating constraints with all defaults (unlimited)."""
        constraints = ResourceConstraints()
        assert constraints.tokens is None
        assert constraints.api_calls is None
        assert constraints.cost_usd is None

    def test_create_with_valid_values(self) -> None:
        """Test creating constraints with valid positive values."""
        constraints = ResourceConstraints(tokens=10000, api_calls=50, cost_usd=5.0, memory_mb=500.0)
        assert constraints.tokens == 10000
        assert constraints.api_calls == 50
        assert constraints.cost_usd == 5.0
        assert constraints.memory_mb == 500.0

    def test_negative_values_raise_error(self) -> None:
        """Test that negative resource values raise ValueError."""
        with pytest.raises(ValueError, match="tokens must be non-negative"):
            ResourceConstraints(tokens=-100)

        with pytest.raises(ValueError, match="cost_usd must be non-negative"):
            ResourceConstraints(cost_usd=-1.0)

    def test_zero_values_allowed(self) -> None:
        """Test that zero values are valid (no budget)."""
        constraints = ResourceConstraints(tokens=0, api_calls=0, cost_usd=0.0)
        assert constraints.tokens == 0
        assert constraints.api_calls == 0
        assert constraints.cost_usd == 0.0

    def test_immutability(self) -> None:
        """Test that ResourceConstraints is immutable (frozen dataclass)."""
        constraints = ResourceConstraints(tokens=1000)
        with pytest.raises(AttributeError):
            constraints.tokens = 2000  # type: ignore


class TestContractMode:
    """Tests for ContractMode enum and strategic modes."""

    def test_contract_mode_values(self) -> None:
        """Test that ContractMode has all three strategic modes."""
        assert ContractMode.URGENT.value == "urgent"
        assert ContractMode.BALANCED.value == "balanced"
        assert ContractMode.ECONOMICAL.value == "economical"

    def test_contract_default_mode(self) -> None:
        """Test that Contract defaults to BALANCED mode."""
        contract = Contract(id="test", name="Test Contract")
        assert contract.mode == ContractMode.BALANCED

    def test_contract_with_urgent_mode(self) -> None:
        """Test creating contract with URGENT mode."""
        contract = Contract(
            id="urgent-task",
            name="Urgent Task",
            mode=ContractMode.URGENT,
            resources=ResourceConstraints(tokens=20000),  # Higher budget for speed
        )
        assert contract.mode == ContractMode.URGENT
        assert contract.resources.tokens == 20000

    def test_contract_with_economical_mode(self) -> None:
        """Test creating contract with ECONOMICAL mode."""
        contract = Contract(
            id="batch-job",
            name="Batch Processing",
            mode=ContractMode.ECONOMICAL,
            resources=ResourceConstraints(tokens=5000),  # Lower budget for cost savings
        )
        assert contract.mode == ContractMode.ECONOMICAL
        assert contract.resources.tokens == 5000

    def test_contract_mode_in_repr(self) -> None:
        """Test that contract mode is represented in string format."""
        contract_urgent = Contract(id="u1", name="Urgent", mode=ContractMode.URGENT)
        contract_eco = Contract(id="e1", name="Eco", mode=ContractMode.ECONOMICAL)

        # Mode should be accessible
        assert contract_urgent.mode == ContractMode.URGENT
        assert contract_eco.mode == ContractMode.ECONOMICAL


class TestTemporalConstraints:
    """Tests for TemporalConstraints validation and creation."""

    def test_create_empty_constraints(self) -> None:
        """Test creating temporal constraints with defaults."""
        constraints = TemporalConstraints()
        assert constraints.deadline is None
        assert constraints.max_duration is None
        assert constraints.deadline_type == DeadlineType.HARD

    def test_create_with_deadline(self) -> None:
        """Test creating constraints with absolute deadline."""
        deadline = datetime(2025, 12, 31, 23, 59, 59)
        constraints = TemporalConstraints(deadline=deadline, deadline_type=DeadlineType.SOFT)
        assert constraints.deadline == deadline
        assert constraints.deadline_type == DeadlineType.SOFT

    def test_create_with_duration(self) -> None:
        """Test creating constraints with max duration."""
        duration = timedelta(hours=2)
        constraints = TemporalConstraints(max_duration=duration)
        assert constraints.max_duration == duration

    def test_negative_quality_decay_raises_error(self) -> None:
        """Test that negative quality decay raises ValueError."""
        with pytest.raises(ValueError, match="soft_deadline_quality_decay must be non-negative"):
            TemporalConstraints(soft_deadline_quality_decay=-0.1)

    def test_quality_decay_zero_allowed(self) -> None:
        """Test that zero quality decay is valid."""
        constraints = TemporalConstraints(soft_deadline_quality_decay=0.0)
        assert constraints.soft_deadline_quality_decay == 0.0


class TestInputOutputSpecification:
    """Tests for InputSpecification and OutputSpecification."""

    def test_input_spec_defaults(self) -> None:
        """Test InputSpecification with defaults."""
        spec = InputSpecification()
        assert spec.schema is None
        assert spec.constraints == {}
        assert spec.examples == []

    def test_input_spec_with_data(self) -> None:
        """Test InputSpecification with data."""
        schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        constraints = {"max_length": 1000}
        examples = [{"query": "example"}]

        spec = InputSpecification(schema=schema, constraints=constraints, examples=examples)
        assert spec.schema == schema
        assert spec.constraints == constraints
        assert spec.examples == examples

    def test_output_spec_defaults(self) -> None:
        """Test OutputSpecification with defaults."""
        spec = OutputSpecification()
        assert spec.schema is None
        assert spec.quality_criteria == {}
        assert spec.min_quality == 0.0

    def test_output_spec_min_quality_validation(self) -> None:
        """Test that min_quality must be in [0, 1]."""
        OutputSpecification(min_quality=0.0)  # Valid
        OutputSpecification(min_quality=1.0)  # Valid

        with pytest.raises(ValueError, match="min_quality must be in"):
            OutputSpecification(min_quality=-0.1)

        with pytest.raises(ValueError, match="min_quality must be in"):
            OutputSpecification(min_quality=1.1)


class TestSuccessCriterion:
    """Tests for SuccessCriterion."""

    def test_create_criterion(self) -> None:
        """Test creating a success criterion."""
        criterion = SuccessCriterion(name="completion", condition="task_done == True", weight=0.5)
        assert criterion.name == "completion"
        assert criterion.condition == "task_done == True"
        assert criterion.weight == 0.5
        assert criterion.required is False

    def test_weight_validation(self) -> None:
        """Test that weight must be in [0, 1]."""
        SuccessCriterion(name="test", condition="true", weight=0.0)
        SuccessCriterion(name="test", condition="true", weight=1.0)

        with pytest.raises(ValueError, match="weight must be in"):
            SuccessCriterion(name="test", condition="true", weight=-0.1)

        with pytest.raises(ValueError, match="weight must be in"):
            SuccessCriterion(name="test", condition="true", weight=1.5)


class TestContract:
    """Tests for Contract creation and lifecycle management."""

    def test_create_minimal_contract(self) -> None:
        """Test creating a contract with only required fields."""
        contract = Contract(id="test-1", name="Test Contract")
        assert contract.id == "test-1"
        assert contract.name == "Test Contract"
        assert contract.state == ContractState.DRAFTED
        assert contract.version == "1.0"

    def test_create_full_contract(self) -> None:
        """Test creating a contract with all fields."""
        contract = Contract(
            id="code-review-1",
            name="Code Review Agent",
            description="Automated PR review",
            version="2.0",
            inputs=InputSpecification(schema={"type": "object"}),
            outputs=OutputSpecification(min_quality=0.8),
            skills=["static_analysis", "security_scan"],
            resources=ResourceConstraints(tokens=50000, api_calls=30),
            temporal=TemporalConstraints(max_duration=timedelta(minutes=5)),
            success_criteria=[
                SuccessCriterion(name="completion", condition="done", weight=0.5, required=True)
            ],
            termination_conditions=[
                TerminationCondition(type="time_limit", condition="timeout", priority=1)
            ],
        )

        assert contract.id == "code-review-1"
        assert contract.name == "Code Review Agent"
        assert contract.description == "Automated PR review"
        assert contract.version == "2.0"
        assert len(contract.skills) == 2
        assert contract.resources.tokens == 50000
        assert contract.temporal.max_duration == timedelta(minutes=5)
        assert len(contract.success_criteria) == 1
        assert len(contract.termination_conditions) == 1

    def test_contract_state_transitions(self) -> None:
        """Test valid contract state transitions."""
        contract = Contract(id="test", name="Test")
        assert contract.state == ContractState.DRAFTED

        # DRAFTED -> ACTIVE
        contract.activate()
        assert contract.state == ContractState.ACTIVE  # type: ignore[comparison-overlap]
        assert contract.is_active()

        # Cannot activate again
        with pytest.raises(ValueError, match="Cannot activate contract"):
            contract.activate()

    def test_contract_fulfill(self) -> None:
        """Test fulfilling a contract."""
        contract = Contract(id="test", name="Test")
        contract.activate()

        contract.fulfill()
        assert contract.state == ContractState.FULFILLED
        assert contract.is_complete()

        # Cannot fulfill from non-active state
        contract2 = Contract(id="test2", name="Test2")
        with pytest.raises(ValueError, match="Cannot fulfill contract"):
            contract2.fulfill()

    def test_contract_violate(self) -> None:
        """Test violating a contract."""
        contract = Contract(id="test", name="Test")
        contract.activate()

        contract.violate(reason="Budget exceeded")
        assert contract.state == ContractState.VIOLATED
        assert contract.metadata["violation_reason"] == "Budget exceeded"
        assert contract.is_complete()

    def test_contract_expire(self) -> None:
        """Test expiring a contract."""
        contract = Contract(id="test", name="Test")
        contract.activate()

        contract.expire()
        assert contract.state == ContractState.EXPIRED
        assert contract.is_complete()

    def test_contract_terminate(self) -> None:
        """Test terminating a contract."""
        contract = Contract(id="test", name="Test")

        # Can terminate from DRAFTED
        contract.terminate(reason="User cancelled")
        assert contract.state == ContractState.TERMINATED
        assert contract.metadata["termination_reason"] == "User cancelled"
        assert contract.is_complete()

        # Can terminate from ACTIVE
        contract2 = Contract(id="test2", name="Test2")
        contract2.activate()
        contract2.terminate()
        assert contract2.state == ContractState.TERMINATED

        # Cannot terminate from terminal states
        with pytest.raises(ValueError, match="Cannot terminate contract"):
            contract2.terminate()

    def test_is_active_method(self) -> None:
        """Test is_active method."""
        contract = Contract(id="test", name="Test")
        assert not contract.is_active()

        contract.activate()
        assert contract.is_active()

        contract.fulfill()
        assert not contract.is_active()

    def test_is_complete_method(self) -> None:
        """Test is_complete method."""
        contract = Contract(id="test", name="Test")
        assert not contract.is_complete()

        contract.activate()
        assert not contract.is_complete()

        contract.fulfill()
        assert contract.is_complete()

    def test_contract_repr(self) -> None:
        """Test string representation of contract."""
        contract = Contract(id="test-123", name="My Contract", version="2.0")
        repr_str = repr(contract)

        assert "test-123" in repr_str
        assert "My Contract" in repr_str
        assert "drafted" in repr_str
        assert "2.0" in repr_str

    def test_contract_created_at_auto_set(self) -> None:
        """Test that created_at is automatically set."""
        before = datetime.now()
        contract = Contract(id="test", name="Test")
        after = datetime.now()

        assert before <= contract.created_at <= after

    def test_metadata_storage(self) -> None:
        """Test that metadata dict works correctly."""
        contract = Contract(id="test", name="Test")
        contract.metadata["custom_key"] = "custom_value"
        contract.metadata["priority"] = 10

        assert contract.metadata["custom_key"] == "custom_value"
        assert contract.metadata["priority"] == 10
