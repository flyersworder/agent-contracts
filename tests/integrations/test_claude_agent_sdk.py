"""Tests for Claude Agent SDK integration.

These tests mock the SDK — no real Claude sessions needed.
"""

from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from agent_contracts.core.capabilities import Capabilities
from agent_contracts.core.contract import (
    Contract,
    ResourceConstraints,
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
