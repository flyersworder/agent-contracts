"""The two arms that complete the shared-record axis.

`one_shot` removes the running record entirely; `critique` adds a second agent
that holds no budget. Between them they bracket the ladder: every other
multi-agent rung DIVIDES the loop's record, and until these existed nothing
established what an undivided record was worth or what a non-dividing second
agent costs.
"""

from __future__ import annotations

from evaluation.chamber_pipeline.agents import _resolve_batch_selection
from evaluation.chamber_pipeline.orchestrator import get_spec, run_cell
from tests.evaluation.conftest import requires_causalchamber

MENU = [f"e{i}" for i in range(20)]


class TestBatchResolution:
    """A batch answer is unconstrained where a one-at-a-time answer is not."""

    def test_over_long_is_cut_to_budget_and_counted(self) -> None:
        chosen, over, short = _resolve_batch_selection(MENU[:15], MENU, 5, 0, "x")
        assert len(chosen) == 5
        assert (over, short) == (10, 0)

    def test_short_is_topped_up_and_counted(self) -> None:
        """A silent top-up would let a model that answered with one name score
        as a full-budget arm."""
        chosen, over, short = _resolve_batch_selection(["e1"], MENU, 5, 0, "x")
        assert len(chosen) == 5
        assert (over, short) == (0, 4)
        assert "e1" in chosen

    def test_off_menu_names_are_discarded_before_counting(self) -> None:
        chosen, _over, short = _resolve_batch_selection(["nope", "e1"], MENU, 2, 0, "x")
        assert set(chosen) <= set(MENU)
        assert len(chosen) == 2
        assert short == 1, "the hallucinated name left the answer one short"

    def test_the_cut_is_not_menu_order(self) -> None:
        """Same reason as the team arm's claim cap: the menu is grouped by
        variable family, so a slice keeps the head families in every seed."""
        tails = [
            sum(1 for n in _resolve_batch_selection(MENU, MENU, 5, s, "x")[0] if int(n[1:]) >= 10)
            for s in range(30)
        ]
        assert sum(tails) / len(tails) > 1.0, f"mean tail kept = {sum(tails) / len(tails)}"


@requires_causalchamber
class TestBatchArmsRun:
    @staticmethod
    def _llm(calls: dict[str, int]):
        def fake(**kw):
            calls["n"] = calls.get("n", 0) + 1
            text = kw["messages"][-1]["content"]
            if "Critique this set" in text:
                body = "Over-weighted on one family; swap one for an untouched target."
            else:
                menu = [
                    line for line in text.split("Menu:\n")[-1].split("\n\n")[0].split("\n") if line
                ]
                body = "\n".join(menu[:8])

            class M:
                content = body

            class C:
                message = M()

            resp = type("R", (), {})()
            resp.choices = [C()]
            resp.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 10})()
            return resp

        return fake

    def test_one_shot_spends_one_call_and_the_whole_budget(self) -> None:
        calls: dict[str, int] = {}
        record = run_cell(
            get_spec("one_shot"), "lt", "standard", budget_k=5, seed=0, llm=self._llm(calls)
        )
        assert record.status == "ok"
        assert calls["n"] == 1, "one call for the whole budget is the point of the arm"
        assert len((record.chosen_experiments or "").split(",")) == 5

    def test_critique_spends_three_calls_regardless_of_budget(self) -> None:
        """Propose, review, revise. Flat in k, which is what makes it the
        cheapest multi-agent arm and worth reporting on the cost axis."""
        for budget in (3, 9):
            calls: dict[str, int] = {}
            record = run_cell(
                get_spec("critique"),
                "lt",
                "standard",
                budget_k=budget,
                seed=0,
                llm=self._llm(calls),
            )
            assert record.status == "ok"
            assert calls["n"] == 3
            assert len((record.chosen_experiments or "").split(",")) == budget

    def test_an_unreadable_critique_leaves_the_proposal_standing(self) -> None:
        """A review nobody could parse did not change the plan -- it must not
        empty the basket."""

        def fake(**kw):
            text = kw["messages"][-1]["content"]
            if "Critique this set" in text or "A reviewer responded" in text:
                body = ""
            else:
                menu = [
                    line for line in text.split("Menu:\n")[-1].split("\n\n")[0].split("\n") if line
                ]
                body = "\n".join(menu[:4])

            class M:
                content = body

            class C:
                message = M()

            resp = type("R", (), {})()
            resp.choices = [C()]
            resp.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()
            return resp

        record = run_cell(get_spec("critique"), "lt", "standard", budget_k=4, seed=0, llm=fake)
        assert record.status == "ok"
        assert len((record.chosen_experiments or "").split(",")) == 4


def test_both_arms_are_registered_and_need_no_scout_calibration() -> None:
    """`scout_roles` must stay None: neither is a two-scout arm, and handing
    either one the fan-in calibration would budget it for roles it has not."""
    for name in ("one_shot", "critique"):
        spec = get_spec(name)
        assert spec.accepts_llm
        assert spec.scout_roles is None
        assert spec.extra_kwargs == ()
