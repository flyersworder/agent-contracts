"""Unit tests for pre/post-check hooks."""

from agent_contracts.core import (
    Contract,
    EnforcementAction,
    ResourceConstraints,
    ResourceMonitor,
)
from agent_contracts.core.enforcement import CheckContext, HookResult


class TestCheckContext:
    """Tests for CheckContext frozen dataclass."""

    def test_create_context(self) -> None:
        """Test creating a CheckContext with all fields."""
        contract = Contract(id="test", name="Test")
        monitor = ResourceMonitor(ResourceConstraints(tokens=1000))
        ctx = CheckContext(
            contract=contract,
            monitor=monitor,
            phase="pre_check",
            metadata={"integration": "litellm", "model": "gpt-4"},
        )
        assert ctx.contract == contract
        assert ctx.monitor == monitor
        assert ctx.phase == "pre_check"
        assert ctx.metadata == {"integration": "litellm", "model": "gpt-4"}

    def test_context_is_frozen(self) -> None:
        """Test that CheckContext is immutable."""
        import pytest

        contract = Contract(id="test", name="Test")
        monitor = ResourceMonitor(ResourceConstraints(tokens=1000))
        ctx = CheckContext(
            contract=contract,
            monitor=monitor,
            phase="pre_check",
            metadata={},
        )
        with pytest.raises(AttributeError):
            ctx.phase = "post_check"  # type: ignore[misc]


class TestHookResult:
    """Tests for HookResult frozen dataclass."""

    def test_default_result(self) -> None:
        """Test HookResult defaults to allow."""
        result = HookResult()
        assert result.allow is True
        assert result.reason == ""
        assert result.action == EnforcementAction.WARN

    def test_blocking_result(self) -> None:
        """Test creating a blocking HookResult."""
        result = HookResult(
            allow=False,
            reason="Topic not allowed",
            action=EnforcementAction.HARD_STOP,
        )
        assert result.allow is False
        assert result.reason == "Topic not allowed"
        assert result.action == EnforcementAction.HARD_STOP

    def test_result_is_frozen(self) -> None:
        """Test that HookResult is immutable."""
        import pytest

        result = HookResult()
        with pytest.raises(AttributeError):
            result.allow = False  # type: ignore[misc]
