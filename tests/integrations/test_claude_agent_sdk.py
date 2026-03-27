"""Tests for Claude Agent SDK integration.

These tests mock the SDK — no real Claude sessions needed.
"""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage

from agent_contracts.core.capabilities import Capabilities
from agent_contracts.core.contract import (
    Contract,
    ResourceConstraints,
    TemporalConstraints,
)
from agent_contracts.integrations import CLAUDE_AGENT_SDK_AVAILABLE


class TestClaudeAgentSdkImport:
    """Test import availability."""

    def test_availability_flag_exists(self) -> None:
        assert isinstance(CLAUDE_AGENT_SDK_AVAILABLE, bool)

    @pytest.mark.skipif(
        not CLAUDE_AGENT_SDK_AVAILABLE,
        reason="claude-agent-sdk not installed",
    )
    def test_import_contracted_claude_agent(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        assert ContractedClaudeAgent is not None


class TestContractedClaudeAgentInit:
    """Test constructor and options mapping."""

    def _make_contract(self, **kwargs: Any) -> Contract:
        return Contract(
            id="test",
            name="test",
            resources=ResourceConstraints(**kwargs.get("resources", {"tokens": 10000})),
            temporal=kwargs.get("temporal"),
            capabilities=kwargs.get("capabilities"),
        )

    def test_basic_init(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract()
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        assert agent.contract is contract
        assert agent.prompt == "Hello"
        assert agent.strict_mode is True

    def test_iterations_maps_to_max_turns(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract(resources={"tokens": 10000, "iterations": 5})
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        merged = agent._build_options()
        assert merged.max_turns == 5

    def test_cost_usd_maps_to_max_budget_usd(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract(resources={"tokens": 10000, "cost_usd": 3.50})
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        merged = agent._build_options()
        assert merged.max_budget_usd == 3.50

    def test_user_options_preserved(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract(resources={"tokens": 10000, "iterations": 5})
        user_options = ClaudeAgentOptions(
            permission_mode="acceptEdits",
            model="claude-sonnet-4-6",
        )
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello", options=user_options)

        merged = agent._build_options()
        assert merged.permission_mode == "acceptEdits"
        assert merged.model == "claude-sonnet-4-6"
        assert merged.max_turns == 5  # from contract

    def test_user_max_turns_not_overridden(self) -> None:
        """User's explicit max_turns takes precedence — more restrictive wins."""
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract(resources={"tokens": 10000, "iterations": 10})
        user_options = ClaudeAgentOptions(max_turns=3)
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello", options=user_options)

        merged = agent._build_options()
        assert merged.max_turns == 3

    def test_capabilities_tools_merged(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract(capabilities=Capabilities(tools=["Read", "Grep"]))
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        merged = agent._build_options()
        assert "Read" in merged.allowed_tools
        assert "Grep" in merged.allowed_tools

    def test_capabilities_instructions_prepended(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract(capabilities=Capabilities(instructions="Always be concise."))
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        merged = agent._build_options()
        assert "Always be concise." in merged.system_prompt


class TestPreToolUseHook:
    """Test PreToolUse enforcement hook."""

    def _make_agent(self, **resource_kwargs: Any) -> Any:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = Contract(
            id="test",
            name="test",
            resources=ResourceConstraints(**resource_kwargs),
        )
        return ContractedClaudeAgent(contract=contract, prompt="Hello")

    @pytest.mark.asyncio
    async def test_allows_tool_within_limits(self) -> None:
        agent = self._make_agent(tokens=10000, tool_invocations=5)
        result = await agent._pre_tool_use_hook(
            {
                "tool_name": "Read",
                "tool_input": {},
                "tool_use_id": "id1",
                "agent_id": "main",
                "agent_type": "main",
                "hook_event_name": "PreToolUse",
                "session_id": "s1",
                "transcript_path": "/tmp",
                "cwd": "/tmp",
            },
            "s1",
            None,
        )
        assert result == {} or result.get("decision") != "block"

    @pytest.mark.asyncio
    async def test_blocks_when_per_tool_limit_exceeded(self) -> None:
        agent = self._make_agent(tokens=10000, per_tool_limits={"Read": 2})
        agent._resource_monitor.usage.tool_usage_by_name["Read"] = 2
        result = await agent._pre_tool_use_hook(
            {
                "tool_name": "Read",
                "tool_input": {},
                "tool_use_id": "id1",
                "agent_id": "main",
                "agent_type": "main",
                "hook_event_name": "PreToolUse",
                "session_id": "s1",
                "transcript_path": "/tmp",
                "cwd": "/tmp",
            },
            "s1",
            None,
        )
        assert result.get("decision") == "block"
        assert "Read" in result.get("reason", "")

    @pytest.mark.asyncio
    async def test_blocks_when_aggregate_tool_limit_exceeded(self) -> None:
        agent = self._make_agent(tokens=10000, tool_invocations=3)
        agent._resource_monitor.usage.tool_invocations = 3
        result = await agent._pre_tool_use_hook(
            {
                "tool_name": "Edit",
                "tool_input": {},
                "tool_use_id": "id1",
                "agent_id": "main",
                "agent_type": "main",
                "hook_event_name": "PreToolUse",
                "session_id": "s1",
                "transcript_path": "/tmp",
                "cwd": "/tmp",
            },
            "s1",
            None,
        )
        assert result.get("decision") == "block"

    @pytest.mark.asyncio
    async def test_blocks_web_search_when_limit_exceeded(self) -> None:
        agent = self._make_agent(tokens=10000, web_searches=2)
        agent._resource_monitor.usage.web_searches = 2
        result = await agent._pre_tool_use_hook(
            {
                "tool_name": "WebSearch",
                "tool_input": {},
                "tool_use_id": "id1",
                "agent_id": "main",
                "agent_type": "main",
                "hook_event_name": "PreToolUse",
                "session_id": "s1",
                "transcript_path": "/tmp",
                "cwd": "/tmp",
            },
            "s1",
            None,
        )
        assert result.get("decision") == "block"

    @pytest.mark.asyncio
    async def test_blocks_when_past_deadline(self) -> None:
        # Set deadline to 1 hour in the past (naive datetime matches monitor's datetime.now())
        past_deadline = datetime.now() - timedelta(hours=1)
        contract = Contract(
            id="test",
            name="test",
            resources=ResourceConstraints(tokens=10000),
            temporal=TemporalConstraints(deadline=past_deadline),
        )
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")
        result = await agent._pre_tool_use_hook(
            {
                "tool_name": "Read",
                "tool_input": {},
                "tool_use_id": "id1",
                "agent_id": "main",
                "agent_type": "main",
                "hook_event_name": "PreToolUse",
                "session_id": "s1",
                "transcript_path": "/tmp",
                "cwd": "/tmp",
            },
            "s1",
            None,
        )
        assert result.get("decision") == "block"


class TestPostToolUseHook:
    """Test PostToolUse audit hook."""

    @pytest.mark.asyncio
    async def test_tracks_tool_invocations(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = Contract(id="test", name="test", resources=ResourceConstraints(tokens=10000))
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        await agent._post_tool_use_hook(
            {
                "tool_name": "Read",
                "tool_input": {},
                "tool_use_id": "id1",
                "agent_id": "main",
                "agent_type": "main",
                "hook_event_name": "PostToolUse",
                "tool_response": None,
                "session_id": "s1",
                "transcript_path": "/tmp",
                "cwd": "/tmp",
            },
            "s1",
            None,
        )
        assert agent._resource_monitor.usage.tool_invocations == 1
        assert agent._resource_monitor.usage.tool_usage_by_name["Read"] == 1

    @pytest.mark.asyncio
    async def test_tracks_web_searches(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = Contract(id="test", name="test", resources=ResourceConstraints(tokens=10000))
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        await agent._post_tool_use_hook(
            {
                "tool_name": "WebSearch",
                "tool_input": {},
                "tool_use_id": "id1",
                "agent_id": "main",
                "agent_type": "main",
                "hook_event_name": "PostToolUse",
                "tool_response": None,
                "session_id": "s1",
                "transcript_path": "/tmp",
                "cwd": "/tmp",
            },
            "s1",
            None,
        )
        assert agent._resource_monitor.usage.web_searches == 1

    @pytest.mark.asyncio
    async def test_emits_enforcement_event(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = Contract(id="test", name="test", resources=ResourceConstraints(tokens=10000))
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        events: list[Any] = []
        agent._enforcer.add_callback(lambda e: events.append(e))

        await agent._post_tool_use_hook(
            {
                "tool_name": "Edit",
                "tool_input": {},
                "tool_use_id": "id1",
                "agent_id": "main",
                "agent_type": "main",
                "hook_event_name": "PostToolUse",
                "tool_response": None,
                "session_id": "s1",
                "transcript_path": "/tmp",
                "cwd": "/tmp",
            },
            "s1",
            None,
        )
        assert len(events) == 1
        assert events[0].event_type == "tool_use"
        assert "Edit" in events[0].message

    @pytest.mark.asyncio
    async def test_tracks_multiple_tools(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = Contract(id="test", name="test", resources=ResourceConstraints(tokens=10000))
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        hook_base = {
            "tool_input": {},
            "agent_id": "main",
            "agent_type": "main",
            "hook_event_name": "PostToolUse",
            "tool_response": None,
            "session_id": "s1",
            "transcript_path": "/tmp",
            "cwd": "/tmp",
        }

        await agent._post_tool_use_hook(
            {**hook_base, "tool_name": "Read", "tool_use_id": "id1"}, "s1", None
        )
        await agent._post_tool_use_hook(
            {**hook_base, "tool_name": "Edit", "tool_use_id": "id2"}, "s1", None
        )
        await agent._post_tool_use_hook(
            {**hook_base, "tool_name": "Read", "tool_use_id": "id3"}, "s1", None
        )

        assert agent._resource_monitor.usage.tool_invocations == 3
        assert agent._resource_monitor.usage.tool_usage_by_name["Read"] == 2
        assert agent._resource_monitor.usage.tool_usage_by_name["Edit"] == 1


class TestAexecute:
    """Test async execution with mocked SDK."""

    def _make_contract(self, **resource_kwargs: Any) -> Contract:
        return Contract(id="test", name="test", resources=ResourceConstraints(**resource_kwargs))

    @pytest.mark.asyncio
    async def test_basic_execution_returns_result(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract(tokens=50000)
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        # Create mock messages
        mock_assistant = MagicMock(spec=AssistantMessage)
        mock_assistant.usage = {"input_tokens": 100, "output_tokens": 50}

        mock_result = MagicMock(spec=ResultMessage)
        mock_result.result = "Test output"

        async def mock_query(*args: Any, **kwargs: Any) -> Any:
            yield mock_assistant
            yield mock_result

        with patch("agent_contracts.integrations.claude_agent_sdk.query", mock_query):
            result = await agent.aexecute()

        assert result.output == "Test output"
        assert result.success is True
        assert agent._resource_monitor.usage.tokens == 150
        assert agent._resource_monitor.usage.api_calls == 1

    @pytest.mark.asyncio
    async def test_token_limit_violation_stops_execution(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract(tokens=100)
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        mock_assistant = MagicMock(spec=AssistantMessage)
        mock_assistant.usage = {"input_tokens": 80, "output_tokens": 80}

        mock_result = MagicMock(spec=ResultMessage)
        mock_result.result = "Partial output"

        async def mock_query(*args: Any, **kwargs: Any) -> Any:
            yield mock_assistant
            yield mock_result

        with patch("agent_contracts.integrations.claude_agent_sdk.query", mock_query):
            result = await agent.aexecute()

        assert result.success is False
        assert len(result.violations) > 0

    @pytest.mark.asyncio
    async def test_execution_log_populated(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract(tokens=50000)
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        mock_assistant = MagicMock(spec=AssistantMessage)
        mock_assistant.usage = {"input_tokens": 100, "output_tokens": 50}

        mock_result = MagicMock(spec=ResultMessage)
        mock_result.result = "Done"

        async def mock_query(*args: Any, **kwargs: Any) -> Any:
            yield mock_assistant
            yield mock_result

        with patch("agent_contracts.integrations.claude_agent_sdk.query", mock_query):
            result = await agent.aexecute()

        assert result.execution_log is not None
        assert result.execution_log.contract_id == "test"
        assert result.execution_log.resource_usage["tokens"] == 150

    @pytest.mark.asyncio
    async def test_handles_exception_during_query(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract(tokens=50000)
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        async def mock_query(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Connection lost")
            yield  # make it an async generator

        with patch("agent_contracts.integrations.claude_agent_sdk.query", mock_query):
            result = await agent.aexecute()

        assert result.success is False
        assert any("Connection lost" in v for v in result.violations)

    @pytest.mark.asyncio
    async def test_messages_without_usage_skipped(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = self._make_contract(tokens=50000)
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        # Assistant message without usage data
        mock_assistant_no_usage = MagicMock(spec=AssistantMessage)
        mock_assistant_no_usage.usage = None

        mock_assistant_with_usage = MagicMock(spec=AssistantMessage)
        mock_assistant_with_usage.usage = {"input_tokens": 200, "output_tokens": 100}

        mock_result = MagicMock(spec=ResultMessage)
        mock_result.result = "Done"

        async def mock_query(*args: Any, **kwargs: Any) -> Any:
            yield mock_assistant_no_usage
            yield mock_assistant_with_usage
            yield mock_result

        with patch("agent_contracts.integrations.claude_agent_sdk.query", mock_query):
            result = await agent.aexecute()

        assert result.success is True
        assert agent._resource_monitor.usage.tokens == 300
        assert agent._resource_monitor.usage.api_calls == 1  # only counted for message with usage


class TestExecuteSync:
    """Test synchronous execution wrapper."""

    def test_sync_execute_works(self) -> None:
        from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent

        contract = Contract(id="test", name="test", resources=ResourceConstraints(tokens=50000))
        agent = ContractedClaudeAgent(contract=contract, prompt="Hello")

        mock_assistant = MagicMock(spec=AssistantMessage)
        mock_assistant.usage = {"input_tokens": 50, "output_tokens": 25}

        mock_result = MagicMock(spec=ResultMessage)
        mock_result.result = "Sync output"

        async def mock_query(*args: Any, **kwargs: Any) -> Any:
            yield mock_assistant
            yield mock_result

        with patch("agent_contracts.integrations.claude_agent_sdk.query", mock_query):
            result = agent.execute()

        assert result.output == "Sync output"
        assert result.success is True
