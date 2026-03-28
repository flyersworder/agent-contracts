"""Unit tests for pre/post-check hooks."""

from agent_contracts.core import (
    Contract,
    EnforcementAction,
    ResourceConstraints,
    ResourceMonitor,
)
from agent_contracts.core.enforcement import (
    CheckContext,
    CheckHook,
    ContractEnforcer,
    EnforcementEvent,
    HookResult,
)


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


class TestContractEnforcerHooks:
    """Tests for pre/post-check hooks on ContractEnforcer."""

    def _make_enforcer(
        self,
        tokens: int = 1000,
        pre_check_hooks: list[CheckHook] | None = None,
        post_check_hooks: list[CheckHook] | None = None,
    ) -> ContractEnforcer:
        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=tokens))
        contract.activate()
        return ContractEnforcer(
            contract,
            strict_mode=True,
            pre_check_hooks=pre_check_hooks,
            post_check_hooks=post_check_hooks,
        )

    def test_no_hooks_backward_compatible(self) -> None:
        """Existing behavior unchanged when no hooks registered."""
        enforcer = self._make_enforcer()
        is_violated, violations = enforcer.check_constraints()
        assert is_violated is False
        assert violations == []

    def test_pre_check_hook_allows(self) -> None:
        def allow_hook(ctx: CheckContext) -> HookResult:
            return HookResult(allow=True)

        enforcer = self._make_enforcer(pre_check_hooks=[allow_hook])
        is_violated, _violations = enforcer.check_constraints()
        assert is_violated is False

    def test_pre_check_hook_blocks_with_hard_stop(self) -> None:
        def block_hook(ctx: CheckContext) -> HookResult:
            return HookResult(
                allow=False, reason="Forbidden topic", action=EnforcementAction.HARD_STOP
            )

        enforcer = self._make_enforcer(pre_check_hooks=[block_hook])
        is_violated, violations = enforcer.check_constraints()
        assert is_violated is True
        assert len(violations) == 1
        assert violations[0].resource == "hook"

    def test_pre_check_hook_warn_does_not_block(self) -> None:
        def warn_hook(ctx: CheckContext) -> HookResult:
            return HookResult(
                allow=False, reason="Budget getting high", action=EnforcementAction.WARN
            )

        events: list[EnforcementEvent] = []
        enforcer = self._make_enforcer(pre_check_hooks=[warn_hook])
        enforcer.add_callback(lambda e: events.append(e))
        is_violated, _violations = enforcer.check_constraints()
        assert is_violated is False
        assert any(e.event_type == "hook_blocked" for e in events)

    def test_pre_check_hook_soft_stop_blocks(self) -> None:
        def soft_stop_hook(ctx: CheckContext) -> HookResult:
            return HookResult(
                allow=False, reason="Graceful stop", action=EnforcementAction.SOFT_STOP
            )

        enforcer = self._make_enforcer(pre_check_hooks=[soft_stop_hook])
        is_violated, _violations = enforcer.check_constraints()
        assert is_violated is True

    def test_multiple_hooks_first_block_wins(self) -> None:
        call_order: list[str] = []

        def hook_a(ctx: CheckContext) -> HookResult:
            call_order.append("a")
            return HookResult(
                allow=False, reason="Hook A blocks", action=EnforcementAction.HARD_STOP
            )

        def hook_b(ctx: CheckContext) -> HookResult:
            call_order.append("b")
            return HookResult(allow=True)

        enforcer = self._make_enforcer(pre_check_hooks=[hook_a, hook_b])
        is_violated, _ = enforcer.check_constraints()
        assert is_violated is True
        assert call_order == ["a"]

    def test_metadata_passed_to_hooks(self) -> None:
        received_metadata: list[dict] = []

        def capture_hook(ctx: CheckContext) -> HookResult:
            received_metadata.append(ctx.metadata)
            return HookResult(allow=True)

        enforcer = self._make_enforcer(pre_check_hooks=[capture_hook])
        enforcer.check_constraints(metadata={"integration": "litellm", "model": "gpt-4"})
        assert len(received_metadata) == 1
        assert received_metadata[0]["integration"] == "litellm"
        assert received_metadata[0]["model"] == "gpt-4"

    def test_metadata_defaults_to_empty_dict(self) -> None:
        received_metadata: list[dict] = []

        def capture_hook(ctx: CheckContext) -> HookResult:
            received_metadata.append(ctx.metadata)
            return HookResult(allow=True)

        enforcer = self._make_enforcer(pre_check_hooks=[capture_hook])
        enforcer.check_constraints()
        assert received_metadata[0] == {}

    def test_post_check_hook_runs_after_constraints(self) -> None:
        post_phases: list[str] = []

        def post_hook(ctx: CheckContext) -> HookResult:
            post_phases.append(ctx.phase)
            return HookResult(allow=True)

        enforcer = self._make_enforcer(post_check_hooks=[post_hook])
        enforcer.check_constraints()
        assert post_phases == ["post_check"]

    def test_hook_exception_caught_like_callbacks(self) -> None:
        def bad_hook(ctx: CheckContext) -> HookResult:
            raise ValueError("Hook crashed")

        enforcer = self._make_enforcer(pre_check_hooks=[bad_hook])
        is_violated, _violations = enforcer.check_constraints()
        assert is_violated is False

    def test_add_remove_pre_check_hook(self) -> None:
        enforcer = self._make_enforcer()

        def my_hook(ctx: CheckContext) -> HookResult:
            return HookResult(allow=True)

        enforcer.add_pre_check_hook(my_hook)
        assert my_hook in enforcer.pre_check_hooks
        enforcer.remove_pre_check_hook(my_hook)
        assert my_hook not in enforcer.pre_check_hooks

    def test_add_remove_post_check_hook(self) -> None:
        enforcer = self._make_enforcer()

        def my_hook(ctx: CheckContext) -> HookResult:
            return HookResult(allow=True)

        enforcer.add_post_check_hook(my_hook)
        assert my_hook in enforcer.post_check_hooks
        enforcer.remove_post_check_hook(my_hook)
        assert my_hook not in enforcer.post_check_hooks

    def test_hook_receives_correct_phase(self) -> None:
        phases: list[str] = []

        def phase_hook(ctx: CheckContext) -> HookResult:
            phases.append(ctx.phase)
            return HookResult(allow=True)

        enforcer = self._make_enforcer(pre_check_hooks=[phase_hook], post_check_hooks=[phase_hook])
        enforcer.check_constraints()
        assert phases == ["pre_check", "post_check"]

    def test_throttle_does_not_block(self) -> None:
        def throttle_hook(ctx: CheckContext) -> HookResult:
            return HookResult(allow=False, reason="Slow down", action=EnforcementAction.THROTTLE)

        enforcer = self._make_enforcer(pre_check_hooks=[throttle_hook])
        is_violated, _ = enforcer.check_constraints()
        assert is_violated is False


class TestHookExports:
    """Tests for hook type exports."""

    def test_import_from_core(self) -> None:
        """Types importable from agent_contracts.core."""
        from agent_contracts.core import CheckContext, CheckHook, HookResult

        assert CheckContext is not None
        assert HookResult is not None
        assert CheckHook is not None

    def test_import_from_top_level(self) -> None:
        """Types importable from agent_contracts top-level."""
        from agent_contracts import CheckContext, CheckHook, HookResult

        assert CheckContext is not None
        assert HookResult is not None
        assert CheckHook is not None
