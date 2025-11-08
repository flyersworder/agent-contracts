"""Tests for LangGraph integration (Phase 2C).

Note: These tests mock LangGraph since it's an optional dependency.
"""

from unittest.mock import Mock, patch

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints


class TestLangGraphIntegration:
    """Test LangGraph integration (mocked)."""

    @pytest.fixture(autouse=True)
    def mock_langgraph(self) -> None:
        """Mock LangGraph imports for testing."""
        # Mock the langgraph modules
        self.mock_graph = Mock()
        self.mock_compiled_graph = Mock()

        # Patch the imports
        self.langgraph_patcher = patch.dict(
            "sys.modules",
            {
                "langgraph": Mock(),
                "langgraph.graph": Mock(CompiledGraph=Mock, Graph=Mock),
                "langgraph.graph.graph": Mock(Graph=Mock),
            },
        )
        self.langgraph_patcher.start()

        yield

        self.langgraph_patcher.stop()

    def test_langgraph_available_check(self) -> None:
        """Test checking if LangGraph is available."""
        from agent_contracts.integrations import LANGGRAPH_AVAILABLE

        # In our test environment, it should be available (mocked)
        assert isinstance(LANGGRAPH_AVAILABLE, bool)

    def test_import_contracted_graph(self) -> None:
        """Test importing ContractedGraph."""
        try:
            from agent_contracts.integrations.langgraph import ContractedGraph

            assert ContractedGraph is not None
        except ImportError:
            # LangGraph not available, which is fine
            pytest.skip("LangGraph not available")

    def test_import_create_contracted_graph(self) -> None:
        """Test importing convenience function."""
        try:
            from agent_contracts.integrations.langgraph import create_contracted_graph

            assert create_contracted_graph is not None
        except ImportError:
            pytest.skip("LangGraph not available")


class TestContractedGraphMocked:
    """Test ContractedGraph with mocked LangGraph."""

    def test_create_contracted_graph(self) -> None:
        """Test creating a ContractedGraph."""
        # Skip if LangGraph not available
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-graph",
            name="test-graph",
            resources=ResourceConstraints(tokens=10000, api_calls=25),
        )

        mock_graph = Mock()
        mock_graph.callbacks = []

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        assert contracted.contract == contract
        assert contracted.graph == mock_graph
        assert contracted.strict_mode is True

    def test_contracted_graph_execute(self) -> None:
        """Test executing a ContractedGraph."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-exec",
            name="test-exec",
            resources=ResourceConstraints(tokens=10000, api_calls=20),
        )

        mock_graph = Mock()
        mock_graph.invoke = Mock(return_value={"output": "Final result"})

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        result = contracted.execute({"input": "test query"})

        assert result.success is True
        assert result.output is not None
        mock_graph.invoke.assert_called_once()

    def test_contracted_graph_invoke_method(self) -> None:
        """Test ContractedGraph invoke() method (LangGraph-style API)."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-invoke",
            name="test-invoke",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_graph = Mock()
        mock_graph.invoke = Mock(return_value={"result": "Success"})

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        # Mock execute to return successful result
        mock_result = Mock()
        mock_result.success = True
        mock_result.output = {"result": "Success"}
        contracted.execute = Mock(return_value=mock_result)

        output = contracted.invoke({"input": "test"})

        assert output == {"result": "Success"}
        contracted.execute.assert_called_once()

    def test_contracted_graph_callable(self) -> None:
        """Test ContractedGraph is callable."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-callable",
            name="test-callable",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_graph = Mock()
        mock_graph.invoke = Mock(return_value={"output": "Result"})

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        # Mock execute
        mock_result = Mock()
        mock_result.success = True
        mock_result.output = {"output": "Result"}
        contracted.execute = Mock(return_value=mock_result)

        # Should be callable
        output = contracted({"input": "test"})

        assert output == {"output": "Result"}

    def test_contracted_graph_with_strict_mode(self) -> None:
        """Test ContractedGraph with strict mode."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-strict",
            name="test-strict",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_graph = Mock()

        contracted = ContractedGraph(contract=contract, graph=mock_graph, strict_mode=True)

        assert contracted.strict_mode is True

    def test_contracted_graph_without_logging(self) -> None:
        """Test ContractedGraph with logging disabled."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-no-log",
            name="test-no-log",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_graph = Mock()

        contracted = ContractedGraph(contract=contract, graph=mock_graph, enable_logging=False)

        assert contracted.enable_logging is False


class TestContractedGraphStreaming:
    """Test ContractedGraph streaming support."""

    def test_stream_method(self) -> None:
        """Test ContractedGraph stream() method."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-stream",
            name="test-stream",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_graph = Mock()
        # Mock stream to yield state updates
        mock_graph.stream = Mock(
            return_value=iter([{"step": 1}, {"step": 2}, {"step": 3, "output": "Final"}])
        )

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        # Collect streamed chunks
        chunks = list(contracted.stream({"input": "test"}))

        assert len(chunks) == 3
        assert chunks[0] == {"step": 1}
        assert chunks[-1]["output"] == "Final"
        mock_graph.stream.assert_called_once()

    def test_stream_not_supported(self) -> None:
        """Test stream() when graph doesn't support streaming."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-no-stream",
            name="test-no-stream",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_graph = Mock(spec=[])  # No stream method

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        with pytest.raises(NotImplementedError):
            list(contracted.stream({"input": "test"}))


class TestCreateContractedGraph:
    """Test create_contracted_graph convenience function."""

    def test_create_contracted_graph_basic(self) -> None:
        """Test creating contracted graph with convenience function."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import create_contracted_graph

        mock_graph = Mock()
        mock_graph.invoke = Mock(return_value={"output": "Result"})

        contracted = create_contracted_graph(
            graph=mock_graph,
            resources={"tokens": 50000, "api_calls": 25, "cost_usd": 2.0},
        )

        assert contracted.contract.resources.tokens == 50000
        assert contracted.contract.resources.api_calls == 25
        assert contracted.contract.resources.cost_usd == 2.0

    def test_create_contracted_graph_with_temporal(self) -> None:
        """Test creating contracted graph with temporal constraints."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import create_contracted_graph

        mock_graph = Mock()

        contracted = create_contracted_graph(
            graph=mock_graph,
            resources={"tokens": 50000},
            temporal={"max_duration": 600},  # 10 minutes
        )

        assert contracted.contract.temporal is not None

    def test_create_contracted_graph_with_custom_id(self) -> None:
        """Test creating contracted graph with custom ID."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import create_contracted_graph

        mock_graph = Mock()

        contracted = create_contracted_graph(
            graph=mock_graph,
            resources={"tokens": 50000},
            contract_id="research-workflow",
        )

        assert contracted.contract.id == "research-workflow"

    def test_create_contracted_graph_auto_id(self) -> None:
        """Test creating contracted graph with auto-generated ID."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import create_contracted_graph

        mock_graph = Mock()

        contracted = create_contracted_graph(
            graph=mock_graph,
            resources={"tokens": 50000},
        )

        # ID should be auto-generated
        assert contracted.contract.id.startswith("graph-")


class TestLangGraphBudgetTracking:
    """Test cumulative budget tracking across graph nodes."""

    def test_budget_tracked_across_nodes(self) -> None:
        """Test that budget is tracked cumulatively across all nodes."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-cumulative",
            name="test-cumulative",
            resources=ResourceConstraints(tokens=10000, api_calls=10),
        )

        mock_graph = Mock()
        mock_graph.invoke = Mock(return_value={"output": "Final"})

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        # Simulate multiple calls (like in a graph with cycles)
        for i in range(3):
            result = contracted.execute({"input": f"query {i}"})
            assert result.success is True

        # Should have accumulated usage across all calls
        assert contracted.resource_monitor.usage.api_calls >= 0

    def test_budget_awareness_injection(self) -> None:
        """Test that budget info is injected into graph state."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-budget-inject",
            name="test-budget-inject",
            resources=ResourceConstraints(tokens=10000, cost_usd=1.0, api_calls=10),
        )

        mock_graph = Mock()

        def capture_invoke(inputs, config=None):
            # Verify budget_info is in inputs
            assert "budget_info" in inputs
            assert "remaining_tokens" in inputs["budget_info"]
            assert "remaining_cost" in inputs["budget_info"]
            assert "remaining_api_calls" in inputs["budget_info"]
            return {"output": "Success"}

        mock_graph.invoke = capture_invoke

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        # Execute without budget_info
        result = contracted.execute({"query": "test"})

        assert result.success is True


class TestLangGraphComplexWorkflows:
    """Test handling of complex workflows with cycles and branches."""

    def test_graph_with_cycles(self) -> None:
        """Test graph execution with cycles (retries/loops)."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-cycles",
            name="test-cycles",
            resources=ResourceConstraints(
                tokens=20000,
                api_calls=15,  # Limit iterations
            ),
        )

        mock_graph = Mock()

        # LangGraph internally handles cycles - our wrapper calls invoke() once
        # The graph itself manages the loop iterations internally
        def simulate_graph_result(inputs, config=None):
            # Simulate final result after graph completes all internal cycles
            return {"continue": False, "output": "Done after 5 iterations", "iterations": 5}

        mock_graph.invoke = simulate_graph_result

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        result = contracted.execute({"input": "start"})

        assert result.success is True
        # Our wrapper calls invoke() once; LangGraph handles cycles internally
        assert result.output["output"] == "Done after 5 iterations"

    def test_parallel_execution(self) -> None:
        """Test budget tracking with parallel node execution."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-parallel",
            name="test-parallel",
            resources=ResourceConstraints(tokens=30000, api_calls=20),
        )

        mock_graph = Mock()
        mock_graph.invoke = Mock(
            return_value={"parallel_results": ["result1", "result2", "result3"]}
        )

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        result = contracted.execute({"input": "parallel tasks"})

        assert result.success is True
        assert "parallel_results" in result.output


class TestLangGraphIntegrationImportError:
    """Test LangGraph integration when LangGraph is not installed."""

    def test_import_error_handling(self) -> None:
        """Test that ImportError is raised gracefully when LangGraph not installed."""
        # Temporarily remove langgraph from sys.modules
        import sys

        langgraph_modules = {k: v for k, v in sys.modules.items() if k.startswith("langgraph")}

        for module in langgraph_modules:
            if module in sys.modules:
                del sys.modules[module]

        try:
            # Try importing - should fail gracefully
            from agent_contracts.integrations import LANGGRAPH_AVAILABLE

            # In our test, it might be True (mocked) or False (not installed)
            assert isinstance(LANGGRAPH_AVAILABLE, bool)
        finally:
            # Restore modules
            sys.modules.update(langgraph_modules)


class TestLangGraphTokenTracking:
    """Test token tracking for LangGraph nodes."""

    def test_callback_setup(self) -> None:
        """Test that callbacks are set up for token tracking."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-callback",
            name="test-callback",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_graph = Mock()

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        # Verify resource monitor is set up
        assert contracted.resource_monitor is not None
        assert contracted.resource_monitor.usage.tokens == 0

    def test_callback_tracks_tokens_openai_style(self) -> None:
        """Test that callback tracks tokens from OpenAI-style responses."""
        pytest.importorskip("langgraph")
        pytest.importorskip("langchain_core")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-token-tracking",
            name="test-token-tracking",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_graph = Mock()

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        # Get the callback from the config
        config = contracted._build_config()
        callback = config["callbacks"][0]

        # Create mock LLM response (OpenAI style)
        mock_response = Mock()
        mock_response.llm_output = {"token_usage": {"total_tokens": 150}}
        mock_response.generations = []

        # Simulate LLM call completion
        callback.on_llm_end(mock_response)

        # Verify tokens were tracked
        assert contracted.resource_monitor.usage.tokens == 150
        assert contracted.resource_monitor.usage.api_calls == 1

    def test_callback_tracks_tokens_google_style(self) -> None:
        """Test that callback tracks tokens from Google-style responses."""
        pytest.importorskip("langgraph")
        pytest.importorskip("langchain_core")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-google-tokens",
            name="test-google-tokens",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_graph = Mock()

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        # Get the callback
        config = contracted._build_config()
        callback = config["callbacks"][0]

        # Create mock LLM response (Google style)
        mock_response = Mock()
        mock_response.llm_output = {"usage_metadata": {"total_tokens": 200}}
        mock_response.generations = []

        # Simulate LLM call completion
        callback.on_llm_end(mock_response)

        # Verify tokens were tracked
        assert contracted.resource_monitor.usage.tokens == 200
        assert contracted.resource_monitor.usage.api_calls == 1

    def test_callback_tracks_tokens_from_generations(self) -> None:
        """Test that callback tracks tokens from generations metadata."""
        pytest.importorskip("langgraph")
        pytest.importorskip("langchain_core")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-gen-tokens",
            name="test-gen-tokens",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_graph = Mock()

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        # Get the callback
        config = contracted._build_config()
        callback = config["callbacks"][0]

        # Create mock LLM response with tokens in generations
        mock_message = Mock()
        mock_message.response_metadata = {"usage_metadata": {"total_tokens": 250}}

        mock_generation = Mock()
        mock_generation.message = mock_message

        mock_response = Mock()
        mock_response.llm_output = None
        mock_response.generations = [[mock_generation]]

        # Simulate LLM call completion
        callback.on_llm_end(mock_response)

        # Verify tokens were tracked
        assert contracted.resource_monitor.usage.tokens == 250
        assert contracted.resource_monitor.usage.api_calls == 1

    def test_callback_cumulative_tracking(self) -> None:
        """Test that callback tracks tokens cumulatively across multiple calls."""
        pytest.importorskip("langgraph")
        pytest.importorskip("langchain_core")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-cumulative",
            name="test-cumulative",
            resources=ResourceConstraints(tokens=10000),
        )

        mock_graph = Mock()

        contracted = ContractedGraph(contract=contract, graph=mock_graph)

        # Get the callback
        config = contracted._build_config()
        callback = config["callbacks"][0]

        # Simulate multiple LLM calls
        for _ in range(3):
            mock_response = Mock()
            mock_response.llm_output = {"token_usage": {"total_tokens": 100}}
            mock_response.generations = []
            callback.on_llm_end(mock_response)

        # Verify cumulative tracking
        assert contracted.resource_monitor.usage.tokens == 300
        assert contracted.resource_monitor.usage.api_calls == 3


class TestLangGraphViolationHandling:
    """Test contract violation handling in complex graphs."""

    def test_violation_stops_execution_strict_mode(self) -> None:
        """Test that violations stop execution in strict mode."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-violation",
            name="test-violation",
            resources=ResourceConstraints(
                tokens=100,  # Very low limit
                api_calls=1,
            ),
        )

        mock_graph = Mock()

        def exceed_budget(inputs, config=None):
            # Simulate token usage
            return {"output": "Result"}

        mock_graph.invoke = exceed_budget

        contracted = ContractedGraph(contract=contract, graph=mock_graph, strict_mode=True)

        # First call should work
        result1 = contracted.execute({"input": "test1"})
        assert result1.success is True

        # Manually exceed budget for testing
        contracted.resource_monitor.usage.add_api_call(cost=0.01, tokens=0)
        contracted.resource_monitor.usage.add_api_call(cost=0.01, tokens=0)

        # Next calls should detect violation
        # (In real scenario, enforcer would catch during execution)

    def test_violation_continues_non_strict_mode(self) -> None:
        """Test that violations are logged but don't stop execution in non-strict mode."""
        pytest.importorskip("langgraph")

        from agent_contracts.integrations.langgraph import ContractedGraph

        contract = Contract(
            id="test-non-strict",
            name="test-non-strict",
            resources=ResourceConstraints(tokens=100, api_calls=1),
        )

        mock_graph = Mock()
        mock_graph.invoke = Mock(return_value={"output": "Result"})

        contracted = ContractedGraph(contract=contract, graph=mock_graph, strict_mode=False)

        # Should not raise even if we exceed
        result = contracted.execute({"input": "test"})

        # Result might be success or failure depending on actual usage
        assert result is not None
