"""Tests for LangChain integration (Phase 2B).

Note: These tests mock LangChain since it's an optional dependency.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints


class TestLangChainIntegration:
    """Test LangChain integration (mocked)."""

    @pytest.fixture(autouse=True)
    def mock_langchain(self) -> None:
        """Mock LangChain imports for testing."""
        # Mock the langchain modules
        self.mock_chain = Mock()
        self.mock_llm = Mock()

        # Patch the imports
        self.langchain_patcher = patch.dict(
            "sys.modules",
            {
                "langchain": Mock(),
                "langchain.chains": Mock(),
                "langchain.chains.base": Mock(Chain=Mock),
                "langchain.llms": Mock(),
                "langchain.prompts": Mock(),
                "langchain.schema": Mock(LLMResult=Mock),
                "langchain.callbacks": Mock(),
                "langchain.callbacks.base": Mock(BaseCallbackHandler=Mock),
            },
        )
        self.langchain_patcher.start()

        yield

        self.langchain_patcher.stop()

    def test_langchain_available_check(self) -> None:
        """Test checking if LangChain is available."""
        from agent_contracts.integrations import LANGCHAIN_AVAILABLE

        # In our test environment, it should be available (mocked)
        assert isinstance(LANGCHAIN_AVAILABLE, bool)

    def test_import_contracted_chain(self) -> None:
        """Test importing ContractedChain."""
        try:
            from agent_contracts.integrations.langchain import ContractedChain

            assert ContractedChain is not None
        except ImportError:
            # LangChain not available, which is fine
            pytest.skip("LangChain not available")

    def test_import_create_contracted_chain(self) -> None:
        """Test importing convenience function."""
        try:
            from agent_contracts.integrations.langchain import create_contracted_chain

            assert create_contracted_chain is not None
        except ImportError:
            pytest.skip("LangChain not available")


class TestContractedChainMocked:
    """Test ContractedChain with mocked LangChain."""

    def test_create_contracted_chain(self) -> None:
        """Test creating a ContractedChain."""
        # Skip if LangChain not available
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-chain",
            name="test-chain",
            resources=ResourceConstraints(tokens=1000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = ContractedChain(contract=contract, chain=mock_chain)

        assert contracted.contract == contract
        assert contracted.chain == mock_chain
        assert contracted.strict_mode is True

    def test_contracted_chain_execute(self) -> None:
        """Test executing a ContractedChain."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-exec",
            name="test-exec",
            resources=ResourceConstraints(tokens=1000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []
        mock_chain.return_value = {"text": "Result"}

        contracted = ContractedChain(contract=contract, chain=mock_chain)

        # Mock the _run_chain method
        contracted._run_chain = Mock(return_value={"text": "Test result"})

        result = contracted.execute({"input": "test"})

        assert result.success is True
        assert result.output is not None
        contracted._run_chain.assert_called_once()

    def test_contracted_chain_run_method(self) -> None:
        """Test ContractedChain run() method (LangChain-style API)."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-run",
            name="test-run",
            resources=ResourceConstraints(tokens=1000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = ContractedChain(contract=contract, chain=mock_chain)

        # Mock execute to return successful result
        mock_result = Mock()
        mock_result.success = True
        mock_result.output = {"text": "Success"}
        contracted.execute = Mock(return_value=mock_result)

        output = contracted.run(input="test")

        assert output == {"text": "Success"}
        contracted.execute.assert_called_once()

    def test_contracted_chain_callable(self) -> None:
        """Test ContractedChain is callable."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-callable",
            name="test-callable",
            resources=ResourceConstraints(tokens=1000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = ContractedChain(contract=contract, chain=mock_chain)

        # Mock execute
        mock_result = Mock()
        mock_result.success = True
        mock_result.output = {"text": "Result"}
        contracted.execute = Mock(return_value=mock_result)

        # Should be callable
        output = contracted({"input": "test"})

        assert output == {"text": "Result"}

    def test_contracted_chain_with_strict_mode(self) -> None:
        """Test ContractedChain with strict mode."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-strict",
            name="test-strict",
            resources=ResourceConstraints(tokens=1000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = ContractedChain(contract=contract, chain=mock_chain, strict_mode=True)

        assert contracted.strict_mode is True

    def test_contracted_chain_without_logging(self) -> None:
        """Test ContractedChain with logging disabled."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-no-log",
            name="test-no-log",
            resources=ResourceConstraints(tokens=1000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = ContractedChain(contract=contract, chain=mock_chain, enable_logging=False)

        assert contracted.enable_logging is False


class TestContractedChainLLMMocked:
    """Test ContractedChainLLM with mocked LangChain."""

    def test_create_contracted_chain_llm(self) -> None:
        """Test creating a ContractedChainLLM."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChainLLM

        contract = Contract(
            id="test-llm",
            name="test-llm",
            resources=ResourceConstraints(tokens=500),
        )

        mock_llm = Mock()

        contracted = ContractedChainLLM(contract=contract, llm=mock_llm)

        assert contracted.contract == contract
        assert contracted.llm == mock_llm

    def test_contracted_chain_llm_callable(self) -> None:
        """Test ContractedChainLLM is callable."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChainLLM

        contract = Contract(
            id="test-llm-call",
            name="test-llm-call",
            resources=ResourceConstraints(tokens=500),
        )

        mock_llm = Mock()

        contracted = ContractedChainLLM(contract=contract, llm=mock_llm)

        # Mock the contracted_chain
        mock_result = Mock()
        mock_result.success = True
        mock_result.output = {"text": "LLM Response"}
        contracted.contracted_chain.execute = Mock(return_value=mock_result)

        response = contracted("What is 2+2?")

        assert response == "LLM Response"

    def test_contracted_chain_llm_execute(self) -> None:
        """Test ContractedChainLLM execute() method."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChainLLM

        contract = Contract(
            id="test-llm-exec",
            name="test-llm-exec",
            resources=ResourceConstraints(tokens=500),
        )

        mock_llm = Mock()

        contracted = ContractedChainLLM(contract=contract, llm=mock_llm)

        # Mock the contracted_chain
        mock_result = Mock()
        mock_result.success = True
        mock_result.output = {"text": "Response"}
        contracted.contracted_chain.execute = Mock(return_value=mock_result)

        result = contracted.execute("test prompt")

        assert result.success is True
        assert result.output is not None


class TestCreateContractedChain:
    """Test create_contracted_chain convenience function."""

    def test_create_contracted_chain_basic(self) -> None:
        """Test creating contracted chain with convenience function."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import create_contracted_chain

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = create_contracted_chain(
            chain=mock_chain,
            resources={"tokens": 1000, "cost_usd": 0.10},
        )

        assert contracted.contract.resources.tokens == 1000
        assert contracted.contract.resources.cost_usd == 0.10

    def test_create_contracted_chain_with_temporal(self) -> None:
        """Test creating contracted chain with temporal constraints."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import create_contracted_chain

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = create_contracted_chain(
            chain=mock_chain,
            resources={"tokens": 1000},
            temporal={"max_duration": 300},  # 5 minutes
        )

        assert contracted.contract.temporal is not None

    def test_create_contracted_chain_with_custom_id(self) -> None:
        """Test creating contracted chain with custom ID."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import create_contracted_chain

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = create_contracted_chain(
            chain=mock_chain,
            resources={"tokens": 1000},
            contract_id="my-custom-chain",
        )

        assert contracted.contract.id == "my-custom-chain"

    def test_create_contracted_chain_auto_id(self) -> None:
        """Test creating contracted chain with auto-generated ID."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import create_contracted_chain

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = create_contracted_chain(
            chain=mock_chain,
            resources={"tokens": 1000},
        )

        # ID should be auto-generated
        assert contracted.contract.id.startswith("chain-")


class TestLangChainTokenTracking:
    """Test token tracking for LangChain."""

    def test_callback_setup(self) -> None:
        """Test that callback is set up for token tracking."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-callback",
            name="test-callback",
            resources=ResourceConstraints(tokens=1000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = ContractedChain(contract=contract, chain=mock_chain)

        # Callback should be added (if LangChain available)
        # In mock environment, we just verify no errors
        assert contracted.chain is not None


class TestTokenTrackingCallback:
    """Test actual token tracking through LangChain callbacks."""

    def _make_contracted_chain(self, tokens: int = 10000) -> Any:
        """Helper to create a ContractedChain with a mock chain."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-token-tracking",
            name="test-token-tracking",
            resources=ResourceConstraints(tokens=tokens),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []

        return ContractedChain(contract=contract, chain=mock_chain)

    def test_on_llm_end_extracts_openai_style_tokens(self) -> None:
        """TokenTrackingCallback should extract tokens from OpenAI-style response."""
        contracted = self._make_contracted_chain()
        callback = contracted._callback_handler

        if callback is None:
            pytest.skip("Callback handler not available in this environment")

        # Create a mock LLMResult with OpenAI-style token_usage
        mock_response = Mock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            }
        }
        mock_response.generations = []

        callback.on_llm_end(mock_response)

        assert contracted.resource_monitor.usage.tokens == 150
        assert contracted.resource_monitor.usage.api_calls == 1
        assert contracted.resource_monitor.usage.cost_usd > 0

    def test_on_llm_end_extracts_google_style_tokens(self) -> None:
        """TokenTrackingCallback should extract tokens from Google-style response."""
        contracted = self._make_contracted_chain()
        callback = contracted._callback_handler

        if callback is None:
            pytest.skip("Callback handler not available in this environment")

        # Create a mock LLMResult with Google-style usage_metadata
        mock_response = Mock()
        mock_response.llm_output = {
            "usage_metadata": {
                "prompt_token_count": 40,
                "candidates_token_count": 80,
                "total_tokens": 120,
            }
        }
        mock_response.generations = []

        callback.on_llm_end(mock_response)

        assert contracted.resource_monitor.usage.tokens == 120
        assert contracted.resource_monitor.usage.api_calls == 1

    def test_on_llm_end_extracts_from_generations_metadata(self) -> None:
        """TokenTrackingCallback should extract tokens from generation metadata."""
        contracted = self._make_contracted_chain()
        callback = contracted._callback_handler

        if callback is None:
            pytest.skip("Callback handler not available in this environment")

        # Create a mock LLMResult with tokens in generation metadata
        mock_message = Mock()
        mock_message.response_metadata = {
            "usage_metadata": {
                "total_tokens": 200,
            }
        }
        mock_generation = Mock()
        mock_generation.message = mock_message

        mock_response = Mock()
        mock_response.llm_output = None  # No llm_output
        mock_response.generations = [[mock_generation]]

        callback.on_llm_end(mock_response)

        assert contracted.resource_monitor.usage.tokens == 200
        assert contracted.resource_monitor.usage.api_calls == 1

    def test_on_llm_end_no_tokens_does_not_update(self) -> None:
        """TokenTrackingCallback should not update monitor when no tokens found."""
        contracted = self._make_contracted_chain()
        callback = contracted._callback_handler

        if callback is None:
            pytest.skip("Callback handler not available in this environment")

        # LLMResult with no token information
        mock_response = Mock()
        mock_response.llm_output = {}
        mock_response.generations = []

        callback.on_llm_end(mock_response)

        assert contracted.resource_monitor.usage.tokens == 0
        assert contracted.resource_monitor.usage.api_calls == 0

    def test_on_llm_end_accumulates_across_calls(self) -> None:
        """TokenTrackingCallback should accumulate tokens across multiple LLM calls."""
        contracted = self._make_contracted_chain()
        callback = contracted._callback_handler

        if callback is None:
            pytest.skip("Callback handler not available in this environment")

        # First call: 100 tokens
        resp1 = Mock()
        resp1.llm_output = {"token_usage": {"total_tokens": 100}}
        resp1.generations = []
        callback.on_llm_end(resp1)

        # Second call: 200 tokens
        resp2 = Mock()
        resp2.llm_output = {"token_usage": {"total_tokens": 200}}
        resp2.generations = []
        callback.on_llm_end(resp2)

        assert contracted.resource_monitor.usage.tokens == 300
        assert contracted.resource_monitor.usage.api_calls == 2


class TestRunChainIntegration:
    """Test _run_chain actually calling the underlying chain."""

    def test_run_chain_invokes_chain_with_invoke(self) -> None:
        """_run_chain should call chain.invoke when available."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-invoke",
            name="test-invoke",
            resources=ResourceConstraints(tokens=5000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []
        # chain.invoke returns a result
        mock_chain.invoke.return_value = {"text": "Invoked result"}

        contracted = ContractedChain(contract=contract, chain=mock_chain)
        result = contracted._run_chain({"query": "test"})

        assert result == {"text": "Invoked result"}
        mock_chain.invoke.assert_called_once()

    def test_run_chain_falls_back_to_call(self) -> None:
        """_run_chain should fall back to __call__ when invoke not available."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-fallback",
            name="test-fallback",
            resources=ResourceConstraints(tokens=5000),
        )

        mock_chain = Mock(spec=[])  # No invoke attribute
        mock_chain.callbacks = []
        mock_chain.return_value = {"text": "Called result"}

        contracted = ContractedChain(contract=contract, chain=mock_chain)
        result = contracted._run_chain({"query": "test"})

        assert result == {"text": "Called result"}
        mock_chain.assert_called_once_with({"query": "test"})

    def test_run_chain_extracts_usage_metadata_from_invoke_result(self) -> None:
        """_run_chain should extract usage_metadata from invoke result."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-usage-meta",
            name="test-usage-meta",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []

        # Simulate a result with usage_metadata (e.g., from ChatModel)
        mock_result = Mock()
        mock_result.usage_metadata = {
            "total_tokens": 500,
            "output_token_details": {"reasoning": 100},
        }
        mock_chain.invoke.return_value = mock_result

        contracted = ContractedChain(contract=contract, chain=mock_chain)
        contracted._run_chain({"query": "test"})

        # Tokens should have been tracked
        assert contracted.resource_monitor.usage.tokens == 500
        assert contracted.resource_monitor.usage.reasoning_tokens == 100
        assert contracted.resource_monitor.usage.text_tokens == 400
        assert contracted.resource_monitor.usage.api_calls == 1
        assert contracted.resource_monitor.usage.cost_usd > 0

    def test_run_chain_passes_callback_handler_via_config(self) -> None:
        """_run_chain should pass callback handler in config to chain.invoke."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-config",
            name="test-config",
            resources=ResourceConstraints(tokens=5000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []
        mock_chain.invoke.return_value = {"text": "result"}

        contracted = ContractedChain(contract=contract, chain=mock_chain)
        contracted._run_chain({"query": "test"})

        # Verify invoke was called with config containing callbacks
        call_kwargs = mock_chain.invoke.call_args
        config = (
            call_kwargs[1].get("config") if call_kwargs[1] else call_kwargs.kwargs.get("config")
        )
        assert config is not None
        assert "callbacks" in config
        if contracted._callback_handler is not None:
            assert contracted._callback_handler in config["callbacks"]

    def test_execute_runs_chain_end_to_end(self) -> None:
        """execute() should run the full pipeline without mocking _run_chain."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-e2e",
            name="test-e2e",
            resources=ResourceConstraints(tokens=5000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []
        mock_chain.invoke.return_value = {"text": "end-to-end result"}

        contracted = ContractedChain(contract=contract, chain=mock_chain)
        result = contracted.execute({"query": "test"})

        assert result.success is True
        assert result.output == {"text": "end-to-end result"}
        mock_chain.invoke.assert_called_once()


class TestLangChainIntegrationImportError:
    """Test LangChain integration when LangChain is not installed."""

    def test_import_error_handling(self) -> None:
        """Test that ImportError is raised gracefully when LangChain not installed."""
        # Temporarily remove langchain from sys.modules
        import sys

        langchain_modules = {k: v for k, v in sys.modules.items() if k.startswith("langchain")}

        for module in langchain_modules:
            if module in sys.modules:
                del sys.modules[module]

        try:
            # Try importing - should fail gracefully
            from agent_contracts.integrations import LANGCHAIN_AVAILABLE

            # In our test, it might be True (mocked) or False (not installed)
            assert isinstance(LANGCHAIN_AVAILABLE, bool)
        finally:
            # Restore modules
            sys.modules.update(langchain_modules)


class TestLangChainBudgetAwareness:
    """Test budget awareness injection for LangChain."""

    def test_monitored_execution_adds_budget_info(self) -> None:
        """Test that monitored execution adds budget info to inputs."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-budget-info",
            name="test-budget-info",
            resources=ResourceConstraints(tokens=1000, cost_usd=0.50),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []
        mock_chain.return_value = {"text": "Result"}

        contracted = ContractedChain(contract=contract, chain=mock_chain)

        # Create input without budget_info
        input_data: dict[str, Any] = {"query": "test"}

        # Mock _run_chain to capture inputs
        def capture_run_chain(inputs: dict[str, Any]) -> dict[str, Any]:
            return {"text": f"Processed with {inputs.get('budget_info', 'no budget')}"}

        contracted._run_chain = capture_run_chain  # type: ignore[assignment]

        # Execute through monitored_execution
        _result = contracted._monitored_execution(input_data)

        # Should have added budget_info
        assert "budget_info" in input_data
        assert "remaining_tokens" in input_data["budget_info"]
        assert "remaining_cost" in input_data["budget_info"]
        assert "time_pressure" in input_data["budget_info"]

    def test_monitored_execution_preserves_existing_budget_info(self) -> None:
        """Test that _monitored_execution does not overwrite existing budget_info."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-budget-preserve",
            name="test-budget-preserve",
            resources=ResourceConstraints(tokens=1000),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = ContractedChain(contract=contract, chain=mock_chain)

        custom_budget = {"remaining_tokens": 999, "custom_field": "preserved"}
        input_data: dict[str, Any] = {"query": "test", "budget_info": custom_budget}

        def capture_run_chain(inputs: dict[str, Any]) -> dict[str, Any]:
            return {"text": "done"}

        contracted._run_chain = capture_run_chain  # type: ignore[assignment]
        contracted._monitored_execution(input_data)

        # Original budget_info should be preserved
        assert input_data["budget_info"] is custom_budget
        assert input_data["budget_info"]["custom_field"] == "preserved"

    def test_budget_info_reflects_remaining_resources(self) -> None:
        """Test that budget_info values reflect actual remaining resources."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedChain

        contract = Contract(
            id="test-budget-values",
            name="test-budget-values",
            resources=ResourceConstraints(tokens=5000, cost_usd=1.00),
        )

        mock_chain = Mock()
        mock_chain.callbacks = []

        contracted = ContractedChain(contract=contract, chain=mock_chain)

        # Simulate some prior usage
        contracted.resource_monitor.usage.add_tokens(count=2000)
        contracted.resource_monitor.usage.add_api_call(cost=0.30)

        input_data: dict[str, Any] = {"query": "test"}

        def capture_run_chain(inputs: dict[str, Any]) -> dict[str, Any]:
            return {"text": "done"}

        contracted._run_chain = capture_run_chain  # type: ignore[assignment]
        contracted._monitored_execution(input_data)

        budget = input_data["budget_info"]
        assert budget["remaining_tokens"] == 3000  # 5000 - 2000
        assert abs(budget["remaining_cost"] - 0.70) < 0.01  # 1.00 - 0.30
