"""Tests for the M3c Planner+Reasoner multi-agent chamber variant.

Covers `evaluation.chamber_pipeline.agents.planner_reasoner_agents` —
plan §5.1 variant 5, the contribution-load-bearing variant. Uses the
same FakeLLM mocking strategy as test_chamber_llm_agents.py.

What these tests are explicitly about:

1. **Conservation enforcement** — A + B <= k is checked upfront with
   a clear ConservationViolationError. Violations raise BEFORE any
   LLM call (no API spend on illegal configurations).
2. **Sub-budget compliance** — Planner spends exactly A interventions;
   Reasoner spends exactly B. Cross-contamination would invalidate
   the §5.3 Pareto comparison vs llm_pc.
3. **Role handoff via prompt** — Reasoner's prompt sees Planner's
   chosen experiments via the `already_chosen` block; the system
   message frames Reasoner's role distinctly.
4. **Audit trail** — both sub-budgets are recorded as
   ContractingCapability allocations on the parent contract, per
   plan §5 line 76-77 ("delegation primitives ... are exercised").
5. **Edge cases** — A=0 (Reasoner-only), B=0 (Planner-only).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.integrations import CAUSAL_CHAMBER_AVAILABLE
from evaluation.chamber_pipeline.agents import planner_reasoner_agents
from evaluation.chamber_pipeline.llm_planner import (
    _PLANNER_SYSTEM_MESSAGE,
    _REASONER_SYSTEM_MESSAGE,
)

requires_causalchamber = pytest.mark.skipif(
    not CAUSAL_CHAMBER_AVAILABLE,
    reason="causalchamber not installed — install with pip install 'ai-agent-contracts[chambers]'",
)


# ---------------------------------------------------------------------------
# FakeLLM — mirrors the fixture in test_chamber_llm_agents.py
#
# Duplicated rather than imported so the M3c test file stands alone if
# the M3b file is ever reorganized. ~30 LOC; not worth a shared fixture
# module yet (would matter at M4 once the orchestrator tests start
# needing the same FakeLLM).
# ---------------------------------------------------------------------------


class FakeLLM:
    """Synthetic LiteLLM-shaped completion callable. See test_chamber_llm_agents.py."""

    def __init__(self, responses: list[str] | None = None, responder: Any = None) -> None:
        if (responses is None) == (responder is None):
            raise ValueError("Pass exactly one of `responses` or `responder`")
        self._responses = responses
        self._responder = responder
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, model: str, messages: list[dict[str, str]], **_: Any) -> dict:
        idx = len(self.calls)
        self.calls.append({"model": model, "messages": messages, "idx": idx})
        if self._responses is not None:
            if idx >= len(self._responses):
                raise AssertionError(
                    f"FakeLLM exhausted: agent made {idx + 1} calls but only "
                    f"{len(self._responses)} responses were scripted."
                )
            content = self._responses[idx]
        else:
            content = self._responder(idx, messages)
        return {"choices": [{"message": {"content": content}}]}


def _indexed_menu_responder(idx: int, messages: list[dict[str, str]]) -> str:
    """Return menu entry at position `idx` from the user prompt.

    Mirrors the helper in test_chamber_llm_agents.py — picks distinct
    menu entries so PC's Fisher-Z doesn't hit singular sub-correlation
    matrices on perfectly-redundant pooled data.
    """
    user_text = messages[-1]["content"]
    lines = [line.strip() for line in user_text.splitlines() if line.strip()]
    menu_entries = [line for line in lines if line.startswith(("uniform_", "exp_"))]
    if not menu_entries:
        raise RuntimeError(f"Test fixture found no menu entries in: {user_text[:300]}")
    return menu_entries[idx % len(menu_entries)]


# ---------------------------------------------------------------------------
# Conservation enforcement
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestConservationEnforcement:
    """A + B <= k must be enforced before any LLM call."""

    def test_oversubscribed_raises_conservation_error(self) -> None:
        """A=2, B=1, k=2 -> 2+1=3 > 2 -> violation."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        llm = FakeLLM(responses=[])  # exhaustion would raise — proves no LLM called
        # The framework primitive emits "Cannot allocate N 'intervene'
        # calls to '<child_name>'" on the second create_subcontract.
        # Match on the tool name (framework-stable; won't drift if the
        # surrounding wording is reworded later).
        with pytest.raises(ConservationViolationError, match="intervene"):
            planner_reasoner_agents(adapter, planner_budget=2, reasoner_budget=1, llm=llm)
        assert len(llm.calls) == 0, "Conservation must fail BEFORE any LLM call"

    def test_exact_match_is_allowed(self) -> None:
        """A + B == k is on the boundary; should run successfully."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=4)
        llm = FakeLLM(responder=_indexed_menu_responder)
        adj = planner_reasoner_agents(adapter, planner_budget=2, reasoner_budget=2, llm=llm)
        # Adjacency is well-shaped; total spend is exactly k.
        assert adj.shape == adapter.ground_truth().shape
        assert adapter._resource_monitor.usage.get_tool_usage("intervene") == 4

    def test_under_subscribed_is_allowed(self) -> None:
        """A + B < k is allowed (slack). Total spend is A + B, not k."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=10)
        llm = FakeLLM(responder=_indexed_menu_responder)
        planner_reasoner_agents(adapter, planner_budget=2, reasoner_budget=2, llm=llm)
        # Spent only A + B = 4, not k = 10.
        assert adapter._resource_monitor.usage.get_tool_usage("intervene") == 4

    def test_negative_planner_budget_raises_value_error(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=4)
        with pytest.raises(ValueError, match="non-negative"):
            planner_reasoner_agents(
                adapter, planner_budget=-1, reasoner_budget=1, llm=FakeLLM(responses=[])
            )

    def test_negative_reasoner_budget_raises_value_error(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=4)
        with pytest.raises(ValueError, match="non-negative"):
            planner_reasoner_agents(
                adapter, planner_budget=1, reasoner_budget=-2, llm=FakeLLM(responses=[])
            )


# ---------------------------------------------------------------------------
# Sub-budget compliance
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestSubBudgetCompliance:
    """Each phase spends exactly its allocated sub-budget."""

    def test_planner_spends_exactly_a(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=10)
        llm = FakeLLM(responder=_indexed_menu_responder)
        planner_reasoner_agents(adapter, planner_budget=3, reasoner_budget=2, llm=llm)

        # The first 3 LLM calls are Planner; their system messages match.
        planner_calls = [
            c for c in llm.calls if c["messages"][0]["content"] == _PLANNER_SYSTEM_MESSAGE
        ]
        assert len(planner_calls) == 3

    def test_reasoner_spends_exactly_b(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=10)
        llm = FakeLLM(responder=_indexed_menu_responder)
        planner_reasoner_agents(adapter, planner_budget=2, reasoner_budget=4, llm=llm)

        reasoner_calls = [
            c for c in llm.calls if c["messages"][0]["content"] == _REASONER_SYSTEM_MESSAGE
        ]
        assert len(reasoner_calls) == 4

    def test_total_llm_calls_equals_a_plus_b(self) -> None:
        """No graph-emission step (PC handles inference) -> exactly A+B LLM calls."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=10)
        llm = FakeLLM(responder=_indexed_menu_responder)
        planner_reasoner_agents(adapter, planner_budget=3, reasoner_budget=2, llm=llm)
        assert len(llm.calls) == 5

    def test_total_intervention_count_equals_a_plus_b(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=10)
        llm = FakeLLM(responder=_indexed_menu_responder)
        planner_reasoner_agents(adapter, planner_budget=2, reasoner_budget=3, llm=llm)
        assert adapter._resource_monitor.usage.get_tool_usage("intervene") == 5


# ---------------------------------------------------------------------------
# Role handoff via prompt
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestRoleHandoff:
    """Reasoner's prompt must reflect the Planner's prior choices."""

    def test_planner_prompt_uses_planner_system_message(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=4)
        llm = FakeLLM(responder=_indexed_menu_responder)
        planner_reasoner_agents(adapter, planner_budget=1, reasoner_budget=1, llm=llm)
        # First call is Planner.
        assert llm.calls[0]["messages"][0]["content"] == _PLANNER_SYSTEM_MESSAGE
        # Second call is Reasoner.
        assert llm.calls[1]["messages"][0]["content"] == _REASONER_SYSTEM_MESSAGE

    def test_reasoner_prompt_lists_planners_picks(self) -> None:
        """The Reasoner's user prompt must include the Planner's chosen
        experiments in the `already_chosen` block — that's the role
        handoff signal."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=4)
        llm = FakeLLM(responder=_indexed_menu_responder)
        planner_reasoner_agents(adapter, planner_budget=2, reasoner_budget=1, llm=llm)
        # Get the Planner's actual picks (first 2 LLM calls -> first 2 interventions).
        planner_picks = [
            e["data"]["experiment_name"] for e in adapter.events if e["type"] == "tool_use"
        ][:2]
        # The Reasoner's call is the 3rd (idx 2). Its user prompt must
        # contain ALL of the Planner's picks.
        reasoner_user = llm.calls[2]["messages"][1]["content"]
        for pick in planner_picks:
            assert pick in reasoner_user, (
                f"Reasoner's prompt missing Planner's pick {pick!r}; "
                f"first 500 chars of user prompt: {reasoner_user[:500]}"
            )

    def test_reasoner_does_not_reselect_planner_picks(self) -> None:
        """The Reasoner's selectable pool excludes Planner's picks even
        if the LLM tries to repeat them — fallback picks an unspent."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=4)
        menu = adapter.available_experiments()
        # Script: Planner picks menu[0], menu[1]; Reasoner LLM tries
        # menu[0] (duplicate) and menu[1] (duplicate). Both fall back
        # to RNG-seeded unspent picks.
        llm = FakeLLM(responses=[menu[0], menu[1], menu[0], menu[1]])
        planner_reasoner_agents(adapter, planner_budget=2, reasoner_budget=2, llm=llm, seed=0)
        spent = [e["data"]["experiment_name"] for e in adapter.events if e["type"] == "tool_use"]
        assert len(spent) == 4
        assert len(set(spent)) == 4, f"Expected 4 distinct spent experiments, got {spent}"


# ---------------------------------------------------------------------------
# Audit trail (delegation framework integration)
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestAuditTrail:
    """Sub-budgets recorded via ContractingCapability — the framework primitive."""

    def test_invocation_creates_two_subcontracts(self) -> None:
        """We can't easily inspect the `capability` object after the
        function returns, but we CAN check that the framework's
        subcontract creation didn't raise — which would happen if the
        per_tool_limits dict were malformed. This test serves as a
        regression guard against breaking the create_subcontract
        integration."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=4)
        llm = FakeLLM(responder=_indexed_menu_responder)
        # If create_subcontract validation fails we get ValueError or
        # ConservationViolationError; if either fires, this test surfaces it.
        adj = planner_reasoner_agents(adapter, planner_budget=2, reasoner_budget=2, llm=llm)
        assert adj.shape == adapter.ground_truth().shape


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestEdgeCases:
    """Boundary inputs that must still produce well-typed output."""

    def test_planner_only_a_only(self) -> None:
        """A=k, B=0 -> degenerates to llm_pc-with-Planner-prompt for full budget."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=3)
        llm = FakeLLM(responder=_indexed_menu_responder)
        adj = planner_reasoner_agents(adapter, planner_budget=3, reasoner_budget=0, llm=llm)
        assert adj.shape == adapter.ground_truth().shape
        # All 3 LLM calls are Planner, none Reasoner.
        assert all(c["messages"][0]["content"] == _PLANNER_SYSTEM_MESSAGE for c in llm.calls)
        assert len(llm.calls) == 3

    def test_reasoner_only_b_only(self) -> None:
        """A=0, B=k -> degenerates to Reasoner-prompted llm_pc for full budget."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=3)
        llm = FakeLLM(responder=_indexed_menu_responder)
        adj = planner_reasoner_agents(adapter, planner_budget=0, reasoner_budget=3, llm=llm)
        assert adj.shape == adapter.ground_truth().shape
        assert all(c["messages"][0]["content"] == _REASONER_SYSTEM_MESSAGE for c in llm.calls)
        assert len(llm.calls) == 3

    def test_zero_total_returns_empty_adjacency(self) -> None:
        """A=0, B=0 -> nothing to do, no LLM calls, all-zeros output."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=4)
        llm = FakeLLM(responses=[])
        adj = planner_reasoner_agents(adapter, planner_budget=0, reasoner_budget=0, llm=llm)
        assert (adj.values == 0).all()
        assert adj.shape == adapter.ground_truth().shape
        assert len(llm.calls) == 0

    def test_zero_chamber_budget_returns_empty(self) -> None:
        """k=0 means no spend possible regardless of A,B = 0,0."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=0)
        llm = FakeLLM(responses=[])
        adj = planner_reasoner_agents(adapter, planner_budget=0, reasoner_budget=0, llm=llm)
        assert (adj.values == 0).all()


# ---------------------------------------------------------------------------
# Output shape + invariants vs llm_pc_agent
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestOutputInvariants:
    """Output must align with the chamber's ground-truth node set."""

    def test_returns_node_aligned_dataframe(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=4)
        llm = FakeLLM(responder=_indexed_menu_responder)
        adj = planner_reasoner_agents(adapter, planner_budget=2, reasoner_budget=2, llm=llm)
        gt = adapter.ground_truth()
        assert list(adj.index) == list(gt.index)
        assert list(adj.columns) == list(gt.columns)
        assert adj.shape == gt.shape

    def test_default_model_propagates_to_llm(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        llm = FakeLLM(responder=_indexed_menu_responder)
        planner_reasoner_agents(adapter, planner_budget=1, reasoner_budget=1, llm=llm)
        # Plan §5: default model is openrouter/deepseek/deepseek-v4-flash.
        assert all(c["model"] == "openrouter/deepseek/deepseek-v4-flash" for c in llm.calls)

    def test_custom_model_propagates(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        llm = FakeLLM(responder=_indexed_menu_responder)
        planner_reasoner_agents(
            adapter,
            planner_budget=1,
            reasoner_budget=1,
            model="custom/test-model",
            llm=llm,
        )
        assert all(c["model"] == "custom/test-model" for c in llm.calls)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_top_level_reexport() -> None:
    """planner_reasoner_agents is importable from the package's top level."""
    from evaluation.chamber_pipeline import planner_reasoner_agents as exported

    assert exported is planner_reasoner_agents


def test_conservation_error_message_includes_numbers() -> None:
    """The framework's conservation error must carry actionable info.

    Asserted against a fixed Contract + ResourceMonitor (no chamber dep)
    so this lands in the suite's no-extras tier. Verifies that the
    second `create_subcontract` call is the one that raises (planner
    fits in 5; reasoner asks for 3 of the remaining 1).
    """
    from unittest.mock import MagicMock

    from agent_contracts.core.contract import Contract, ResourceConstraints
    from agent_contracts.core.monitor import ResourceMonitor

    parent = Contract(
        id="test-contract",
        name="Test",
        resources=ResourceConstraints(per_tool_limits={"intervene": 5}),
    )
    monitor = ResourceMonitor(parent.resources)

    # The agent only touches `adapter.contract`, `adapter._resource_monitor`,
    # and `adapter.ground_truth` (for the empty-adjacency early return on
    # exhaustion). MagicMock is fine for the rest because conservation
    # fires before those code paths.
    fake_adapter = MagicMock()
    fake_adapter.contract = parent
    fake_adapter._resource_monitor = monitor
    fake_adapter.ground_truth.return_value.index = ["x", "y"]

    with pytest.raises(ConservationViolationError) as exc:
        planner_reasoner_agents(
            fake_adapter, planner_budget=4, reasoner_budget=3, llm=FakeLLM(responses=[])
        )
    msg = str(exc.value)
    # Tool name + offending child name appear, so M4 misallocations
    # are easy to triage from the error alone.
    assert "intervene" in msg
    assert "reasoner" in msg
    # Structured fields reflect the failing call: reasoner asked for 3,
    # only 1 remained (parent limit 5 minus planner's already-allocated 4).
    assert exc.value.requested == 3
    assert exc.value.available == 1
    assert exc.value.parent_id == "test-contract"
