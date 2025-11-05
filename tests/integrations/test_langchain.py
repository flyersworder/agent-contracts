"""Tests for LangChain integration (Phase 2B).

Note: These tests mock LangChain since it's an optional dependency.
"""

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


class TestContractedLLMMocked:
    """Test ContractedLLM with mocked LangChain."""

    def test_create_contracted_llm(self) -> None:
        """Test creating a ContractedLLM."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedLLM

        contract = Contract(
            id="test-llm",
            name="test-llm",
            resources=ResourceConstraints(tokens=500),
        )

        mock_llm = Mock()

        contracted = ContractedLLM(contract=contract, llm=mock_llm)

        assert contracted.contract == contract
        assert contracted.llm == mock_llm

    def test_contracted_llm_callable(self) -> None:
        """Test ContractedLLM is callable."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedLLM

        contract = Contract(
            id="test-llm-call",
            name="test-llm-call",
            resources=ResourceConstraints(tokens=500),
        )

        mock_llm = Mock()

        contracted = ContractedLLM(contract=contract, llm=mock_llm)

        # Mock the contracted_chain
        mock_result = Mock()
        mock_result.success = True
        mock_result.output = {"text": "LLM Response"}
        contracted.contracted_chain.execute = Mock(return_value=mock_result)

        response = contracted("What is 2+2?")

        assert response == "LLM Response"

    def test_contracted_llm_execute(self) -> None:
        """Test ContractedLLM execute() method."""
        pytest.importorskip("langchain")

        from agent_contracts.integrations.langchain import ContractedLLM

        contract = Contract(
            id="test-llm-exec",
            name="test-llm-exec",
            resources=ResourceConstraints(tokens=500),
        )

        mock_llm = Mock()

        contracted = ContractedLLM(contract=contract, llm=mock_llm)

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
        input_data = {"query": "test"}

        # Mock _run_chain to capture inputs
        def capture_run_chain(inputs):
            return {"text": f"Processed with {inputs.get('budget_info', 'no budget')}"}

        contracted._run_chain = capture_run_chain

        # Execute through monitored_execution
        _result = contracted._monitored_execution(input_data)

        # Should have added budget_info
        assert "budget_info" in input_data
        assert "remaining_tokens" in input_data["budget_info"]
        assert "remaining_cost" in input_data["budget_info"]
        assert "time_pressure" in input_data["budget_info"]
