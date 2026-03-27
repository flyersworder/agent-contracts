"""Tests for Claude Agent SDK integration.

These tests mock the SDK — no real Claude sessions needed.
"""

import pytest

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
