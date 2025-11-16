"""Tests for Google ADK integration.

Note: These tests use the real google-adk package since it's installed
as an optional dependency. We mock the actual LLM calls to avoid API costs.
"""

from unittest.mock import Mock, patch

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints


class TestGoogleAdkIntegration:
    """Test Google ADK integration availability."""

    def test_google_adk_available_check(self) -> None:
        """Test checking if Google ADK is available."""
        from agent_contracts.integrations import GOOGLE_ADK_AVAILABLE

        # Should be available since we installed it
        assert isinstance(GOOGLE_ADK_AVAILABLE, bool)
        assert GOOGLE_ADK_AVAILABLE is True

    def test_import_contracted_adk_agent(self) -> None:
        """Test importing ContractedAdkAgent."""
        from agent_contracts.integrations.google_adk import ContractedAdkAgent

        assert ContractedAdkAgent is not None

    def test_import_contracted_adk_multi_agent(self) -> None:
        """Test importing ContractedAdkMultiAgent."""
        from agent_contracts.integrations.google_adk import ContractedAdkMultiAgent

        assert ContractedAdkMultiAgent is not None

    def test_import_create_contracted_adk_agent(self) -> None:
        """Test importing convenience function."""
        from agent_contracts.integrations.google_adk import create_contracted_adk_agent

        assert create_contracted_adk_agent is not None


class TestContractedAdkAgent:
    """Test ContractedAdkAgent with real google-adk."""

    def test_create_contracted_adk_agent(self) -> None:
        """Test creating a ContractedAdkAgent."""
        from google.adk.agents import LlmAgent

        from agent_contracts.integrations.google_adk import ContractedAdkAgent

        contract = Contract(
            id="test-agent",
            name="test-agent",
            resources=ResourceConstraints(tokens=10000, cost_usd=1.0),
        )

        # Create a simple ADK agent
        agent = LlmAgent(
            name="test_agent",
            model="gemini-2.0-flash",
            instruction="You are a helpful assistant.",
        )

        # Wrap with contract
        contracted = ContractedAdkAgent(contract=contract, agent=agent)

        assert contracted.contract == contract
        assert contracted.agent == agent
        assert contracted.strict_mode is True
        assert contracted.runner is not None

    def test_create_with_custom_runner(self) -> None:
        """Test creating ContractedAdkAgent with custom runner."""
        from google.adk.agents import LlmAgent

        from agent_contracts.integrations.google_adk import ContractedAdkAgent

        contract = Contract(
            id="test-agent",
            name="test-agent",
            resources=ResourceConstraints(tokens=10000),
        )

        agent = LlmAgent(
            name="test_agent",
            model="gemini-2.0-flash",
            instruction="You are a helpful assistant.",
        )

        # Create custom runner mock
        mock_runner = Mock()

        contracted = ContractedAdkAgent(contract=contract, agent=agent, runner=mock_runner)

        assert contracted.runner == mock_runner

    def test_contracted_adk_agent_run_mocked(self) -> None:
        """Test running ContractedAdkAgent with mocked LLM calls."""
        from google.adk.agents import LlmAgent
        from google.genai.types import Content, GenerateContentResponseUsageMetadata, Part

        from agent_contracts.integrations.google_adk import ContractedAdkAgent

        contract = Contract(
            id="test-run",
            name="test-run",
            resources=ResourceConstraints(tokens=10000, cost_usd=1.0),
        )

        agent = LlmAgent(
            name="test_agent",
            model="gemini-2.0-flash",
            instruction="You are a helpful assistant.",
        )

        contracted = ContractedAdkAgent(contract=contract, agent=agent)

        # Mock the runner's run method to return fake events
        mock_event = Mock()
        mock_event.usageMetadata = GenerateContentResponseUsageMetadata(
            total_token_count=100,
            prompt_token_count=50,
            candidates_token_count=50,
            thoughts_token_count=0,
            cached_content_token_count=0,
        )
        mock_event.content = Content(parts=[Part(text="Hello! How can I help you?")])
        mock_event.turnComplete = True

        with patch.object(contracted.runner, "run", return_value=iter([mock_event])):
            result = contracted.run(user_id="test_user", session_id="test_session", message="Hello")

            assert result is not None
            assert "response" in result
            assert result["response"] == "Hello! How can I help you?"
            assert result["total_tokens"] == 100
            assert result["usage_metadata"]["total_tokens"] == 100
            assert result["usage_metadata"]["prompt_tokens"] == 50
            assert result["usage_metadata"]["candidates_tokens"] == 50

    def test_budget_enforcement(self) -> None:
        """Test that budget limits are enforced."""
        from google.adk.agents import LlmAgent
        from google.genai.types import Content, GenerateContentResponseUsageMetadata, Part

        from agent_contracts.integrations.google_adk import ContractedAdkAgent

        # Very low token limit
        contract = Contract(
            id="test-budget",
            name="test-budget",
            resources=ResourceConstraints(tokens=50),  # Only 50 tokens
        )

        agent = LlmAgent(
            name="test_agent",
            model="gemini-2.0-flash",
            instruction="You are a helpful assistant.",
        )

        contracted = ContractedAdkAgent(contract=contract, agent=agent, strict_mode=True)

        # Mock events that exceed budget
        def create_mock_events() -> list[Mock]:
            events = []
            # First event: 30 tokens (within budget)
            event1 = Mock()
            event1.usageMetadata = GenerateContentResponseUsageMetadata(
                total_token_count=30,
                prompt_token_count=15,
                candidates_token_count=15,
                thoughts_token_count=0,
                cached_content_token_count=0,
            )
            event1.content = Content(parts=[Part(text="Thinking...")])
            events.append(event1)

            # Second event: 30 more tokens (would exceed budget of 50)
            event2 = Mock()
            event2.usageMetadata = GenerateContentResponseUsageMetadata(
                total_token_count=30,
                prompt_token_count=15,
                candidates_token_count=15,
                thoughts_token_count=0,
                cached_content_token_count=0,
            )
            event2.content = Content(parts=[Part(text="Done!")])
            events.append(event2)

            return events

        with (
            patch.object(contracted.runner, "run", return_value=iter(create_mock_events())),
            pytest.raises(RuntimeError, match="Contract violated"),
        ):
            # Should raise RuntimeError due to budget violation
            contracted.run(user_id="test_user", session_id="test_session", message="Hello")

    def test_run_debug_convenience(self) -> None:
        """Test the run_debug convenience method."""
        from google.adk.agents import LlmAgent
        from google.genai.types import Content, GenerateContentResponseUsageMetadata, Part

        from agent_contracts.integrations.google_adk import ContractedAdkAgent

        contract = Contract(
            id="test-debug",
            name="test-debug",
            resources=ResourceConstraints(tokens=10000),
        )

        agent = LlmAgent(
            name="test_agent",
            model="gemini-2.0-flash",
            instruction="You are a helpful assistant.",
        )

        contracted = ContractedAdkAgent(contract=contract, agent=agent)

        # Mock event
        mock_event = Mock()
        mock_event.usageMetadata = GenerateContentResponseUsageMetadata(
            total_token_count=50,
            prompt_token_count=25,
            candidates_token_count=25,
            thoughts_token_count=0,
            cached_content_token_count=0,
        )
        mock_event.content = Content(parts=[Part(text="Debug response")])

        with patch.object(contracted.runner, "run", return_value=iter([mock_event])):
            result = contracted.run_debug("Test message")

            assert result["response"] == "Debug response"
            assert result["total_tokens"] == 50


class TestConvenienceFunctions:
    """Test convenience functions for creating contracted agents."""

    def test_create_contracted_adk_agent(self) -> None:
        """Test create_contracted_adk_agent convenience function."""
        from google.adk.agents import LlmAgent

        from agent_contracts.integrations.google_adk import create_contracted_adk_agent

        agent = LlmAgent(
            name="test_agent",
            model="gemini-2.0-flash",
            instruction="You are a helpful assistant.",
        )

        contracted = create_contracted_adk_agent(
            agent=agent,
            resources={"tokens": 50000, "cost_usd": 2.0, "api_calls": 25},
            temporal={"max_duration": 600},  # 10 minutes
            contract_id="my-agent",
        )

        assert contracted is not None
        assert contracted.contract.id == "my-agent"
        assert contracted.contract.resources.tokens == 50000
        assert contracted.contract.resources.cost_usd == 2.0
        assert contracted.contract.resources.api_calls == 25
        assert contracted.contract.temporal.max_duration == 600


class TestMultiAgentSupport:
    """Test multi-agent system support."""

    def test_create_multi_agent_system(self) -> None:
        """Test creating a multi-agent system with contracts."""
        from google.adk.agents import LlmAgent

        from agent_contracts.integrations.google_adk import ContractedAdkMultiAgent

        # Create sub-agents
        researcher = LlmAgent(
            name="researcher",
            model="gemini-2.0-flash",
            instruction="You research topics.",
        )

        planner = LlmAgent(
            name="planner",
            model="gemini-2.0-flash",
            instruction="You create plans.",
        )

        # Create coordinator with sub-agents
        coordinator = LlmAgent(
            name="coordinator",
            model="gemini-2.0-flash",
            instruction="You coordinate research and planning.",
            sub_agents=[researcher, planner],
        )

        contract = Contract(
            id="multi-agent-test",
            name="multi-agent-test",
            resources=ResourceConstraints(tokens=100000, cost_usd=5.0),
        )

        # Wrap coordinator with contract
        contracted = ContractedAdkMultiAgent(contract=contract, agent=coordinator)

        assert contracted is not None
        assert contracted.agent == coordinator
        assert len(contracted.agent.sub_agents) == 2
