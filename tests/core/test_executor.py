"""Tests for ContractExecutor and ContractExecutionResult.

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
from agent_contracts.core.executor import ContractExecutionResult, ContractExecutor
from agent_contracts.integrations.litellm_wrapper import ContractViolationError


class TestContractExecutionResult:
    """Tests for ContractExecutionResult dataclass."""

    def test_create_successful_result(self) -> None:
        """Test creating a successful ContractExecutionResult."""
        now = datetime.now()
        result = ContractExecutionResult(
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
        """Test creating a failed ContractExecutionResult."""
        result = ContractExecutionResult(
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
        result = ContractExecutionResult(
            success=True,
            output="test",
            resource_usage={},
            started_at=started,
            completed_at=completed,
        )
        assert result.duration_seconds == 30.0

    def test_duration_seconds_none_when_incomplete(self) -> None:
        """Test duration_seconds is None when times not set."""
        result = ContractExecutionResult(
            success=True,
            output="test",
            resource_usage={},
        )
        assert result.duration_seconds is None

    def test_tokens_used_default(self) -> None:
        """Test tokens_used returns 0 when not in usage."""
        result = ContractExecutionResult(
            success=True,
            output="test",
            resource_usage={},
        )
        assert result.tokens_used == 0

    def test_cost_usd_default(self) -> None:
        """Test cost_usd returns 0.0 when not in usage."""
        result = ContractExecutionResult(
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
        """Test building LLM parameters.

        Note: After refactoring, ContractedLLM handles max_tokens and
        response_format. The executor only passes temperature.
        """
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
        # max_tokens is now handled by ContractedLLM, not executor
        assert "max_tokens" not in params

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
    """Tests for ContractExecutor with mocked LLM calls.

    Note: We patch the completion function inside litellm_wrapper since
    ContractExecutor now uses ContractedLLM internally.
    """

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_run_successful_execution(self, mock_completion: MagicMock) -> None:
        """Test successful execution with mocked LiteLLM."""
        # Setup mock response - ContractedLLM uses dict-style .get() access
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The answer is 4"
        # Mock dict-style access for usage
        mock_response.get.side_effect = lambda k, d=None: {
            "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
            "_hidden_params": {"response_cost": 0.001},
        }.get(k, d)
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

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_run_with_messages(self, mock_completion: MagicMock) -> None:
        """Test execution with messages input."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        # Mock dict-style access for usage
        mock_response.get.side_effect = lambda k, d=None: {
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "_hidden_params": {"response_cost": 0.0005},
        }.get(k, d)
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

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_run_with_violation(self, mock_completion: MagicMock) -> None:
        """Test execution that results in violation."""
        # Setup mock response with high token usage
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Long response"
        # Mock dict-style access - high token usage to trigger violation
        mock_response.get.side_effect = lambda k, d=None: {
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000},
            "_hidden_params": {"response_cost": 0.05},
        }.get(k, d)
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

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_run_strict_mode_raises(self, mock_completion: MagicMock) -> None:
        """Test that strict mode raises on violation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        # Mock dict-style access - high token usage to trigger violation
        mock_response.get.side_effect = lambda k, d=None: {
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000},
            "_hidden_params": {"response_cost": 0.05},
        }.get(k, d)
        mock_completion.return_value = mock_response

        contract = Contract(
            id="test",
            name="Test",
            resources=ResourceConstraints(tokens=100),
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract, strict_mode=True)

        # ContractedLLM raises ContractViolationError (not RuntimeError)
        with pytest.raises(ContractViolationError, match="Contract violated"):
            executor.run(query="Write something")

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
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

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_run_strategy_in_result(self, mock_completion: MagicMock) -> None:
        """Test that strategy recommendation is included in result."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        # Mock dict-style access for usage
        mock_response.get.side_effect = lambda k, d=None: {
            "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
            "_hidden_params": {"response_cost": 0.001},
        }.get(k, d)
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


class TestSoftCutoff:
    """Tests for soft cutoff / truncation tracking in ContractExecutionResult."""

    def test_execution_result_truncated_default(self) -> None:
        """ContractExecutionResult.truncated should default to False."""
        result = ContractExecutionResult(
            success=True,
            output="full output",
            resource_usage={"tokens": 500},
        )
        assert result.truncated is False

    def test_execution_result_truncated_when_set(self) -> None:
        """ContractExecutionResult should track truncation."""
        result = ContractExecutionResult(
            success=True,
            output="partial output",
            resource_usage={"tokens": 900},
            truncated=True,
        )
        assert result.truncated is True

    def test_executor_initializes_soft_cutoff(self) -> None:
        """ContractExecutor should store the soft_cutoff parameter."""
        contract = Contract(
            id="test",
            name="Test",
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract, soft_cutoff=True)
        assert executor.soft_cutoff is True
        assert executor._truncated is False

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_soft_cutoff_streaming_timeout_returns_partial(
        self, mock_completion: MagicMock
    ) -> None:
        """Soft cutoff should return partial output on timeout."""

        # Simulate streaming that raises a timeout after yielding some chunks
        class TimeoutAfterChunks:
            def __init__(self) -> None:
                self.count = 0

            def __iter__(self):  # type: ignore[no-untyped-def]
                return self

            def __next__(self):  # type: ignore[no-untyped-def]
                self.count += 1
                if self.count <= 2:
                    chunk = MagicMock()
                    chunk.choices = [MagicMock()]
                    chunk.choices[0].delta.content = f"chunk{self.count} "
                    chunk.get.return_value = None
                    return chunk
                raise TimeoutError("Request timed out")

        mock_completion.return_value = TimeoutAfterChunks()

        contract = Contract(
            id="test-soft",
            name="Soft Cutoff Test",
            resources=ResourceConstraints(tokens=10000),
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract, soft_cutoff=True)
        result = executor.run(query="Generate a long response")

        # Should succeed with truncated partial output
        assert result.truncated is True
        assert "chunk1" in result.output
        assert "chunk2" in result.output

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_soft_cutoff_no_truncation_on_success(self, mock_completion: MagicMock) -> None:
        """Soft cutoff should not mark truncated when streaming completes."""
        # Simulate successful streaming
        chunks = []
        for i in range(3):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = f"word{i} "
            chunk.get.return_value = None
            chunks.append(chunk)

        mock_completion.return_value = iter(chunks)

        contract = Contract(
            id="test-soft-ok",
            name="Soft Cutoff No Truncation",
            resources=ResourceConstraints(tokens=10000),
            capabilities=Capabilities(),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        executor = ContractExecutor(contract, soft_cutoff=True)
        result = executor.run(query="Short response")

        assert result.truncated is False
        assert result.success is True


class TestContractExecute:
    """Tests for Contract.execute() method."""

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_contract_execute_method(self, mock_completion: MagicMock) -> None:
        """Test the Contract.execute() convenience method."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "42"
        # Mock dict-style access for usage
        mock_response.get.side_effect = lambda k, d=None: {
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "_hidden_params": {"response_cost": 0.0005},
        }.get(k, d)
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
        assert isinstance(result, ContractExecutionResult)

    def test_contract_execute_no_capabilities_raises(self) -> None:
        """Test that execute() raises when no capabilities."""
        contract = Contract(
            id="test",
            name="Test",
        )
        with pytest.raises(ValueError, match="must have capabilities defined"):
            contract.execute(query="test")
