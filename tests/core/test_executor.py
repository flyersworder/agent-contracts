"""Tests for ContractExecutor and ExecutionResult.

This module tests the execution engine that enables Contract.execute().
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from agent_contracts.core.contract import (
    Capabilities,
    Contract,
    ContractMode,
    ContractState,
    ExecutionConfig,
    ResourceConstraints,
)
from agent_contracts.core.executor import ContractExecutor, ExecutionResult


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_create_successful_result(self) -> None:
        """Test creating a successful ExecutionResult."""
        now = datetime.now()
        result = ExecutionResult(
            success=True,
            output="Hello, world!",
            resource_usage={"tokens": 100, "cost_usd": 0.001},
            started_at=now,
            completed_at=now,
            contract_state=ContractState.FULFILLED,
        )
        assert result.success is True
        assert result.output == "Hello, world!"
        assert result.tokens_used == 100
        assert result.cost_usd == 0.001
        assert result.violations == []
        assert result.error is None

    def test_create_failed_result(self) -> None:
        """Test creating a failed ExecutionResult."""
        result = ExecutionResult(
            success=False,
            output=None,
            resource_usage={"tokens": 50},
            violations=["Token limit exceeded"],
            error="Execution failed",
            contract_state=ContractState.VIOLATED,
        )
        assert result.success is False
        assert result.output is None
        assert result.violations == ["Token limit exceeded"]
        assert result.error == "Execution failed"

    def test_duration_seconds_calculation(self) -> None:
        """Test duration_seconds property."""
        started = datetime(2024, 1, 1, 12, 0, 0)
        completed = datetime(2024, 1, 1, 12, 0, 30)
        result = ExecutionResult(
            success=True,
            output="test",
            resource_usage={},
            started_at=started,
            completed_at=completed,
        )
        assert result.duration_seconds == 30.0

    def test_duration_seconds_none_when_incomplete(self) -> None:
        """Test duration_seconds is None when times not set."""
        result = ExecutionResult(
            success=True,
            output="test",
            resource_usage={},
        )
        assert result.duration_seconds is None

    def test_tokens_used_default(self) -> None:
        """Test tokens_used returns 0 when not in usage."""
        result = ExecutionResult(
            success=True,
            output="test",
            resource_usage={},
        )
        assert result.tokens_used == 0

    def test_cost_usd_default(self) -> None:
        """Test cost_usd returns 0.0 when not in usage."""
        result = ExecutionResult(
            success=True,
            output="test",
            resource_usage={},
        )
        assert result.cost_usd == 0.0


class TestContractExecutor:
    """Tests for ContractExecutor class."""

    def test_create_executor(self) -> None:
        """Test creating a ContractExecutor."""
        contract = Contract(
            id="test",
            name="Test",
            resources=ResourceConstraints(tokens=1000),
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        assert executor.contract is contract
        assert executor.execution_config.model == "gpt-4o"
        assert executor.strict_mode is False

    def test_create_executor_strict_mode(self) -> None:
        """Test creating executor in strict mode."""
        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract, strict_mode=True)
        assert executor.strict_mode is True

    def test_extract_task_from_query(self) -> None:
        """Test extracting task description from query input."""
        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        task = executor._extract_task_description(query="What is 2+2?")
        assert task == "What is 2+2?"

    def test_extract_task_from_prompt(self) -> None:
        """Test extracting task from prompt input."""
        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        task = executor._extract_task_description(prompt="Calculate something")
        assert task == "Calculate something"

    def test_extract_task_from_messages(self) -> None:
        """Test extracting task from messages input."""
        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What is Python?"},
        ]
        task = executor._extract_task_description(messages=messages)
        assert task == "What is Python?"

    def test_extract_task_fallback_to_contract(self) -> None:
        """Test extracting task falls back to contract description."""
        contract = Contract(
            id="test",
            name="Test Task",
            description="A test contract",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        task = executor._extract_task_description()
        assert task == "A test contract"

    def test_extract_task_fallback_to_name(self) -> None:
        """Test extracting task falls back to contract name."""
        contract = Contract(
            id="test",
            name="Test Task Name",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        task = executor._extract_task_description()
        assert task == "Test Task Name"

    def test_prepare_messages_with_no_system(self) -> None:
        """Test preparing messages when no system message exists."""
        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        messages = [{"role": "user", "content": "Hello"}]
        result = executor._prepare_messages("System prompt", messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System prompt"
        assert result[1]["role"] == "user"

    def test_prepare_messages_with_existing_system(self) -> None:
        """Test preparing messages when system message exists."""
        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        messages = [
            {"role": "system", "content": "Original system"},
            {"role": "user", "content": "Hello"},
        ]
        result = executor._prepare_messages("Budget prompt", messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Budget prompt" in result[0]["content"]
        assert "Original system" in result[0]["content"]

    def test_build_llm_params(self) -> None:
        """Test building LLM parameters."""
        contract = Contract(
            id="test",
            name="Test",
            resources=ResourceConstraints(tokens=1000),
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o", temperature=0.5),
        )
        executor = ContractExecutor(contract)
        params = executor._build_llm_params()

        assert params["temperature"] == 0.5
        assert "max_tokens" in params

    def test_get_usage_dict(self) -> None:
        """Test getting usage dictionary."""
        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        usage = executor._get_usage_dict()

        assert "tokens" in usage
        assert "api_calls" in usage
        assert "cost_usd" in usage

    def test_get_adaptive_instruction(self) -> None:
        """Test getting adaptive instruction."""
        contract = Contract(
            id="test",
            name="Test",
            resources=ResourceConstraints(tokens=1000),
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
            mode=ContractMode.ECONOMICAL,
        )
        executor = ContractExecutor(contract)
        instruction = executor.get_adaptive_instruction()

        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_estimate_cost_gpt4o(self) -> None:
        """Test cost estimation for GPT-4o."""
        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        cost = executor._estimate_cost(1000, 500)

        # GPT-4o: $2.50/1M input, $10/1M output
        expected = (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost_claude(self) -> None:
        """Test cost estimation for Claude."""
        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="claude-sonnet-4-20250514"),
        )
        executor = ContractExecutor(contract)
        cost = executor._estimate_cost(1000, 500)

        # Claude: $3/1M input, $15/1M output
        expected = (1000 * 3.00 / 1_000_000) + (500 * 15.00 / 1_000_000)
        assert abs(cost - expected) < 0.0001

    def test_logging(self) -> None:
        """Test that execution logging works."""
        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        executor._log("test_event", {"key": "value"})

        assert len(executor._execution_log) == 1
        assert executor._execution_log[0]["event"] == "test_event"
        assert executor._execution_log[0]["key"] == "value"
        assert "timestamp" in executor._execution_log[0]


class TestContractExecutorWithMock:
    """Tests for ContractExecutor with mocked LLM calls."""

    @patch("litellm.completion")
    def test_run_successful_execution(self, mock_completion: MagicMock) -> None:
        """Test successful execution with mocked LiteLLM."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The answer is 4"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 50
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 20
        mock_completion.return_value = mock_response

        # Execute
        contract = Contract(
            id="test",
            name="Test",
            resources=ResourceConstraints(tokens=1000),
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        result = executor.run(query="What is 2+2?")

        # Verify
        assert result.success is True
        assert result.output == "The answer is 4"
        assert result.contract_state == ContractState.FULFILLED
        assert result.tokens_used > 0
        assert len(result.execution_log) > 0

    @patch("litellm.completion")
    def test_run_with_messages(self, mock_completion: MagicMock) -> None:
        """Test execution with messages input."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 20
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 10
        mock_completion.return_value = mock_response

        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        result = executor.run(
            messages=[
                {"role": "user", "content": "Hi"},
            ]
        )

        assert result.success is True
        assert result.output == "Hello!"

    @patch("litellm.completion")
    def test_run_with_violation(self, mock_completion: MagicMock) -> None:
        """Test execution that results in violation."""
        # Setup mock response with high token usage
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Long response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 2000  # Over budget
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 1000
        mock_completion.return_value = mock_response

        contract = Contract(
            id="test",
            name="Test",
            resources=ResourceConstraints(tokens=100),  # Very tight budget
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        result = executor.run(query="Write a long essay")

        # In lenient mode, we get a result with violations
        assert result.success is False
        assert result.contract_state == ContractState.VIOLATED
        assert len(result.violations) > 0

    @patch("litellm.completion")
    def test_run_strict_mode_raises(self, mock_completion: MagicMock) -> None:
        """Test that strict mode raises on violation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 2000
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 1000
        mock_completion.return_value = mock_response

        contract = Contract(
            id="test",
            name="Test",
            resources=ResourceConstraints(tokens=100),
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract, strict_mode=True)

        with pytest.raises(RuntimeError, match="Contract violated"):
            executor.run(query="Write something")

    @patch("litellm.completion")
    def test_run_with_error(self, mock_completion: MagicMock) -> None:
        """Test execution that results in error."""
        mock_completion.side_effect = Exception("API Error")

        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract)
        result = executor.run(query="Test")

        assert result.success is False
        assert result.error == "API Error"
        assert result.contract_state == ContractState.VIOLATED

    @patch("litellm.completion")
    def test_run_strategy_in_result(self, mock_completion: MagicMock) -> None:
        """Test that strategy recommendation is included in result."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 50
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 20
        mock_completion.return_value = mock_response

        contract = Contract(
            id="test",
            name="Test",
            resources=ResourceConstraints(tokens=1000),
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
            mode=ContractMode.ECONOMICAL,
        )
        executor = ContractExecutor(contract)
        result = executor.run(query="Test")

        assert result.strategy is not None
        assert result.strategy.mode == ContractMode.ECONOMICAL


class TestContractExecute:
    """Tests for Contract.execute() method."""

    @patch("litellm.completion")
    def test_contract_execute_method(self, mock_completion: MagicMock) -> None:
        """Test the Contract.execute() convenience method."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "42"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 30
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_completion.return_value = mock_response

        contract = Contract(
            id="math",
            name="Math Helper",
            resources=ResourceConstraints(tokens=1000),
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        result = contract.execute(query="What is 6 * 7?")

        assert result.success is True
        assert result.output == "42"
        assert isinstance(result, ExecutionResult)

    def test_contract_execute_no_capabilities_raises(self) -> None:
        """Test that execute() raises when no capabilities."""
        contract = Contract(
            id="test",
            name="Test",
        )
        with pytest.raises(ValueError, match="must have capabilities defined"):
            contract.execute(query="test")
