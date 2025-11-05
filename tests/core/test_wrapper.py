"""Tests for ContractAgent wrapper (Phase 2B)."""

import time
from datetime import timedelta

import pytest

from agent_contracts.core.contract import (
    Contract,
    ContractMode,
    ContractState,
    ResourceConstraints,
    TemporalConstraints,
)
from agent_contracts.core.wrapper import (
    ContractAgent,
    ContractViolationError,
    ExecutionLog,
    ExecutionResult,
)


class TestContractAgent:
    """Test ContractAgent base wrapper."""

    def test_create_contract_agent(self) -> None:
        """Test creating a ContractAgent."""
        contract = Contract(
            id="test-agent",
            name="test-agent",
            resources=ResourceConstraints(tokens=1000),
        )

        def simple_agent(x: str) -> str:
            return f"Processed: {x}"

        agent = ContractAgent(contract=contract, agent=simple_agent)

        assert agent.contract == contract
        assert agent.agent == simple_agent
        assert agent.strict_mode is True
        assert agent.enable_logging is True

    def test_execute_simple_agent(self) -> None:
        """Test executing a simple agent."""
        contract = Contract(
            id="test-exec",
            name="test-exec",
            resources=ResourceConstraints(tokens=1000),
        )

        def simple_agent(x: str) -> str:
            return f"Result: {x}"

        agent = ContractAgent(contract=contract, agent=simple_agent)
        result = agent.execute("test input")

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.output == "Result: test input"
        assert result.contract.state == ContractState.FULFILLED
        assert len(result.violations) == 0

    def test_execute_updates_contract_state(self) -> None:
        """Test that execution updates contract state."""
        contract = Contract(
            id="test-state",
            name="test-state",
            resources=ResourceConstraints(tokens=1000),
        )

        def agent(x: str) -> str:
            return "done"

        wrapped = ContractAgent(contract=contract, agent=agent)

        # Initial state
        assert contract.state == ContractState.DRAFTED

        # After execution
        result = wrapped.execute("input")
        assert contract.state == ContractState.FULFILLED
        assert result.success is True

    def test_execute_with_none_output_marks_violated(self) -> None:
        """Test that None output is considered a failure."""
        contract = Contract(
            id="test-none",
            name="test-none",
            resources=ResourceConstraints(tokens=1000),
        )

        def failing_agent(x: str) -> None:
            return None

        agent = ContractAgent(contract=contract, agent=failing_agent)
        result = agent.execute("input")

        assert result.success is False
        assert result.output is None
        assert contract.state == ContractState.VIOLATED

    def test_execute_with_exception(self) -> None:
        """Test handling of agent exceptions."""
        contract = Contract(
            id="test-exception",
            name="test-exception",
            resources=ResourceConstraints(tokens=1000),
        )

        def failing_agent(x: str) -> str:
            raise ValueError("Agent failed!")

        agent = ContractAgent(contract=contract, agent=failing_agent)
        result = agent.execute("input")

        assert result.success is False
        assert result.output is None
        assert contract.state == ContractState.VIOLATED
        assert len(result.violations) == 1
        assert "Agent failed!" in result.violations[0]

    def test_execution_log_created(self) -> None:
        """Test that execution log is created."""
        contract = Contract(
            id="test-log",
            name="test-log",
            resources=ResourceConstraints(tokens=1000, cost_usd=0.10),
        )

        def agent(x: str) -> str:
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent, enable_logging=True)
        result = wrapped.execute("input")

        assert result.execution_log is not None
        assert isinstance(result.execution_log, ExecutionLog)
        assert result.execution_log.contract_id == "test-log"
        assert result.execution_log.start_time is not None
        assert result.execution_log.end_time is not None
        assert result.execution_log.final_state == ContractState.FULFILLED

    def test_execution_log_disabled(self) -> None:
        """Test execution with logging disabled."""
        contract = Contract(
            id="test-no-log",
            name="test-no-log",
            resources=ResourceConstraints(tokens=1000),
        )

        def agent(x: str) -> str:
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent, enable_logging=False)
        result = wrapped.execute("input")

        # No log created when logging disabled
        assert result.execution_log is None
        assert result.success is True

    def test_execution_log_resource_usage(self) -> None:
        """Test that resource usage is logged."""
        contract = Contract(
            id="test-resources",
            name="test-resources",
            resources=ResourceConstraints(tokens=1000, api_calls=5, cost_usd=0.10),
        )

        def agent(x: str) -> str:
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent)
        result = wrapped.execute("input")

        usage = result.execution_log.resource_usage
        assert "tokens" in usage
        assert "api_calls" in usage
        assert "cost_usd" in usage
        assert usage["tokens"] >= 0
        assert usage["cost_usd"] >= 0

    def test_execution_log_temporal_metrics(self) -> None:
        """Test that temporal metrics are logged."""
        contract = Contract(
            id="test-temporal",
            name="test-temporal",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        def agent(x: str) -> str:
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent)
        result = wrapped.execute("input")

        metrics = result.execution_log.temporal_metrics
        assert "elapsed_seconds" in metrics
        assert "deadline_met" in metrics
        assert metrics["elapsed_seconds"] >= 0
        assert isinstance(metrics["deadline_met"], bool)

    def test_get_remaining_budget(self) -> None:
        """Test getting remaining budget."""
        contract = Contract(
            id="test-budget",
            name="test-budget",
            resources=ResourceConstraints(tokens=1000, cost_usd=0.50),
        )

        def agent(x: str) -> str:
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent)

        remaining = wrapped.get_remaining_budget()
        assert "tokens" in remaining
        assert "cost_usd" in remaining
        assert "api_calls" in remaining
        assert remaining["tokens"] == 1000
        assert remaining["cost_usd"] == 0.50

    def test_get_time_pressure(self) -> None:
        """Test getting time pressure."""
        contract = Contract(
            id="test-pressure",
            name="test-pressure",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        def agent(x: str) -> str:
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent)

        # Before starting, pressure is 0
        pressure = wrapped.get_time_pressure()
        assert pressure == 0.0

    def test_to_json_export(self) -> None:
        """Test exporting execution log to JSON."""
        contract = Contract(
            id="test-json",
            name="test-json",
            resources=ResourceConstraints(tokens=1000),
        )

        def agent(x: str) -> str:
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent)
        _result = wrapped.execute("input")

        json_data = wrapped.to_json()
        assert isinstance(json_data, dict)
        assert "contract_id" in json_data
        assert json_data["contract_id"] == "test-json"
        assert "start_time" in json_data
        assert "final_state" in json_data

    def test_to_json_without_execution(self) -> None:
        """Test JSON export before execution."""
        contract = Contract(
            id="test-no-exec",
            name="test-no-exec",
            resources=ResourceConstraints(tokens=1000),
        )

        def agent(x: str) -> str:
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent)
        json_data = wrapped.to_json()

        assert "error" in json_data

    def test_strict_mode_enabled(self) -> None:
        """Test strict mode enforcement."""
        contract = Contract(
            id="test-strict",
            name="test-strict",
            resources=ResourceConstraints(tokens=1000),
        )

        def agent(x: str) -> str:
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent, strict_mode=True)
        assert wrapped.strict_mode is True

    def test_strict_mode_disabled(self) -> None:
        """Test lenient mode."""
        contract = Contract(
            id="test-lenient",
            name="test-lenient",
            resources=ResourceConstraints(tokens=1000),
        )

        def agent(x: str) -> str:
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent, strict_mode=False)
        assert wrapped.strict_mode is False

    def test_contract_with_different_modes(self) -> None:
        """Test ContractAgent with different contract modes."""
        for mode in [ContractMode.URGENT, ContractMode.BALANCED, ContractMode.ECONOMICAL]:
            contract = Contract(
                id=f"test-{mode.value}",
                name=f"test-{mode.value}",
                mode=mode,
                resources=ResourceConstraints(tokens=1000),
            )

            def agent(x: str, _mode: ContractMode = mode) -> str:
                return f"mode: {_mode.value}"

            wrapped = ContractAgent(contract=contract, agent=agent)
            result = wrapped.execute("input")

            assert result.success is True
            assert mode.value in result.output

    def test_execution_metadata(self) -> None:
        """Test execution result metadata."""
        contract = Contract(
            id="test-metadata",
            name="test-metadata",
            resources=ResourceConstraints(tokens=1000),
        )

        def agent(x: str) -> str:
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent)
        result = wrapped.execute("input")

        assert isinstance(result.metadata, dict)
        assert "elapsed_seconds" in result.metadata
        assert result.metadata["elapsed_seconds"] >= 0

    def test_multiple_executions(self) -> None:
        """Test multiple executions of the same agent."""
        contract1 = Contract(id="exec-1", name="exec-1", resources=ResourceConstraints(tokens=1000))
        contract2 = Contract(id="exec-2", name="exec-2", resources=ResourceConstraints(tokens=2000))

        def agent(x: str) -> str:
            return f"Result: {x}"

        wrapped1 = ContractAgent(contract=contract1, agent=agent)
        wrapped2 = ContractAgent(contract=contract2, agent=agent)

        result1 = wrapped1.execute("first")
        result2 = wrapped2.execute("second")

        assert result1.success is True
        assert result2.success is True
        assert "first" in result1.output
        assert "second" in result2.output

    def test_execution_with_temporal_constraint(self) -> None:
        """Test execution with temporal constraints."""
        contract = Contract(
            id="test-temporal-exec",
            name="test-temporal-exec",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=5)),
        )

        def agent(x: str) -> str:
            time.sleep(0.1)  # Small delay
            return "result"

        wrapped = ContractAgent(contract=contract, agent=agent)
        result = wrapped.execute("input")

        assert result.success is True
        assert result.execution_log.temporal_metrics["elapsed_seconds"] >= 0.1
        assert result.execution_log.temporal_metrics["deadline_met"] is True


class TestExecutionLog:
    """Test ExecutionLog functionality."""

    def test_execution_log_to_dict(self) -> None:
        """Test converting execution log to dictionary."""
        from datetime import datetime

        log = ExecutionLog(
            contract_id="test-log",
            start_time=datetime.now(),
            end_time=datetime.now(),
            final_state=ContractState.FULFILLED,
            resource_usage={"tokens": 500, "cost_usd": 0.05},
            temporal_metrics={"elapsed_seconds": 1.5},
            events=[{"type": "start", "message": "Started"}],
            metadata={"key": "value"},
        )

        log_dict = log.to_dict()

        assert isinstance(log_dict, dict)
        assert log_dict["contract_id"] == "test-log"
        assert log_dict["final_state"] == "fulfilled"
        assert "start_time" in log_dict
        assert "end_time" in log_dict
        assert log_dict["resource_usage"]["tokens"] == 500
        assert log_dict["temporal_metrics"]["elapsed_seconds"] == 1.5
        assert len(log_dict["events"]) == 1

    def test_execution_log_with_none_end_time(self) -> None:
        """Test execution log with None end_time."""
        from datetime import datetime

        log = ExecutionLog(
            contract_id="test-log",
            start_time=datetime.now(),
            end_time=None,
            final_state=ContractState.ACTIVE,
            resource_usage={},
            temporal_metrics={},
            events=[],
            metadata={},
        )

        log_dict = log.to_dict()
        assert log_dict["end_time"] is None


class TestContractViolationError:
    """Test ContractViolationError exception."""

    def test_create_violation_error(self) -> None:
        """Test creating a violation error."""
        contract = Contract(
            id="test-violation",
            name="test-violation",
            resources=ResourceConstraints(tokens=1000),
        )

        error = ContractViolationError(
            contract=contract,
            violation_type="budget",
            message="Token budget exceeded",
        )

        assert error.contract == contract
        assert error.violation_type == "budget"
        assert "test-violation" in str(error)
        assert "budget" in str(error)
        assert "Token budget exceeded" in str(error)

    def test_raise_violation_error(self) -> None:
        """Test raising and catching violation error."""
        contract = Contract(
            id="test-raise",
            name="test-raise",
            resources=ResourceConstraints(tokens=1000),
        )

        with pytest.raises(ContractViolationError) as exc_info:
            raise ContractViolationError(
                contract=contract,
                violation_type="deadline",
                message="Deadline exceeded",
            )

        assert exc_info.value.violation_type == "deadline"
        assert "Deadline exceeded" in str(exc_info.value)


class TestContractAgentIntegration:
    """Integration tests for ContractAgent."""

    def test_full_workflow_with_logging(self) -> None:
        """Test complete workflow with execution logging."""
        # Create contract
        contract = Contract(
            id="workflow-test",
            name="workflow-test",
            resources=ResourceConstraints(tokens=5000, api_calls=10, cost_usd=0.50),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=30)),
        )

        # Define agent
        def research_agent(query: str) -> str:
            # Simulate some work
            time.sleep(0.05)
            return f"Research findings on: {query}"

        # Wrap with contract
        wrapped = ContractAgent(contract=contract, agent=research_agent)

        # Execute
        result = wrapped.execute("AI Safety")

        # Verify results
        assert result.success is True
        assert "AI Safety" in result.output
        assert contract.state == ContractState.FULFILLED

        # Verify logging
        log = result.execution_log
        assert log.contract_id == "workflow-test"
        assert log.final_state == ContractState.FULFILLED
        assert log.temporal_metrics["elapsed_seconds"] >= 0.05
        assert log.temporal_metrics["deadline_met"] is True

        # Verify JSON export
        json_data = wrapped.to_json()
        assert json_data["contract_id"] == "workflow-test"
        assert json_data["final_state"] == "fulfilled"

    def test_budget_awareness_during_execution(self) -> None:
        """Test that agent can check budget during execution."""
        contract = Contract(
            id="budget-aware-test",
            name="budget-aware-test",
            resources=ResourceConstraints(tokens=10000, cost_usd=1.0),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        execution_data = {}

        def budget_aware_agent(query: str) -> str:
            # Agent checks budget (in real scenario, would adapt strategy)
            execution_data["remaining"] = wrapped.get_remaining_budget()
            execution_data["pressure"] = wrapped.get_time_pressure()
            return f"Processed: {query}"

        wrapped = ContractAgent(contract=contract, agent=budget_aware_agent)
        result = wrapped.execute("test query")

        assert result.success is True
        assert "remaining" in execution_data
        assert "pressure" in execution_data
        assert execution_data["remaining"]["tokens"] == 10000
        assert 0.0 <= execution_data["pressure"] <= 1.0
