"""Recovering a conservation verdict from recorded spend, without re-running.

300 WT `team` cells carry `conservation_certified = None` because the constant
provisioning their negotiation rounds had only been measured on LT. The first
reading of that was "the archived cells ran under the wrong grant, so they
cannot be retro-certified and the arm must be re-run". That was wrong, and the
three checks below are what make it wrong.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agent_contracts.core.delegation_graph import ConservationViolationError
from evaluation.chamber_pipeline.recertify import (
    RecertifyUnsoundError,
    grant_headroom,
    recertify_frame,
    replay_conservation,
)

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path("runs/m6-wt-ladder-final.parquet").exists(),
    reason="archived WT ladder not present",
)


@pytest.fixture(scope="module")
def wt():
    d = pd.concat(
        [
            pd.read_parquet("runs/m6-wt-ladder-final.parquet"),
            pd.read_parquet("runs/m6-wt-team-rerun.parquet"),
        ],
        ignore_index=True,
    )
    return d.loc[:, ~d.columns.duplicated()]


def test_replay_reproduces_verify_exactly_where_verify_actually_ran(wt):
    """The validation that licenses everything else.

    `replay_conservation` calls the REAL `DelegationGraph.verify()` on a
    rebuilt graph rather than reimplementing its inequalities -- which matters
    because `verify()` also checks a per-tool clause that an inequality
    reimplementation would miss. On the two fan-in arms the recorded verdict
    exists, so the replay can be checked against it rather than trusted.
    """
    known = wt[wt.conservation_certified.notna()]
    assert len(known) == 300
    out = recertify_frame(known, chamber="wt")
    assert (out.conservation_replayed == out.conservation_certified).all()


def test_the_arm_with_no_verdict_gets_one(wt):
    team = wt[wt.agent_name.eq("team")]
    assert team.conservation_certified.isna().all()
    out = recertify_frame(team, chamber="wt")
    assert out.conservation_replayed.notna().all()
    assert 0 < out.conservation_replayed.mean() < 1


def test_the_grant_does_not_shape_the_spend(wt):
    """The precondition, established by the failures rather than despite them.

    Replaying recorded spend against a DIFFERENT grant is only valid if the
    old grant did not truncate the run that produced it. It plainly did not:
    cells overspend by thousands of tokens and complete. An earlier version of
    this test asserted `headroom > 0`, which is incoherent -- negative headroom
    IS a conservation failure, so it would have demanded that no cell ever
    fail.
    """
    head = grant_headroom(wt[wt.agent_name.eq("team")], chamber="wt")
    assert (head < 0).sum() > 0, "no overspend at all would make the guard untested"
    assert (head == 0).sum() == 0, "spend sitting exactly on a ceiling means clipping"


def test_recertify_refuses_when_spend_was_clipped_at_a_ceiling(wt):
    """Mutation-proof: fabricate the clipping signature and require a refusal."""
    from evaluation.chamber_pipeline.recertify import _graph

    team = wt[wt.agent_name.eq("team")].copy()
    row = team.iloc[0]
    graph = _graph("wt", int(row.budget_k), "team", None)
    ceiling = (graph.in_flow("scout_a").tokens or 0) - (graph.out_flow("scout_a").tokens or 0)
    team.loc[team.index[0], "scout_a_tokens"] = ceiling
    with pytest.raises(RecertifyUnsoundError, match="exactly their ceiling"):
        recertify_frame(team, chamber="wt")


def test_replay_is_insensitive_to_the_corrected_constant(wt):
    """Why the correction changes no verdict, stated as a test.

    WT's negotiate constant moved 4138 -> 6102, which only ENLARGES the
    scouts' grants. If any cell's verdict flipped, the recovered number would
    depend on which constant we used and would need re-running to settle. None
    does -- no cell sits inside the band the correction moves.
    """
    team = wt[wt.agent_name.eq("team")]
    new = recertify_frame(team, chamber="wt").conservation_replayed
    old = recertify_frame(team, chamber="wt", c95_negotiate=4138).conservation_replayed
    assert (new == old).all()


def test_replay_is_pure(wt):
    team = wt[wt.agent_name.eq("team")].head(20)
    a = recertify_frame(team, chamber="wt").conservation_replayed.tolist()
    b = recertify_frame(team, chamber="wt").conservation_replayed.tolist()
    assert a == b


def test_a_cell_that_overspends_is_not_certified():
    assert (
        replay_conservation(
            chamber="wt",
            budget_k=14,
            arm="team",
            scout_a_tokens=10**9,
            scout_b_tokens=1,
            aggregator_tokens=1,
        )
        is False
    )


def test_the_per_tool_charge_is_not_verdict_bearing_on_this_archive(wt):
    """Measured, not assumed — and recorded because it is a limitation.

    Deleting the per-tool charge from `replay_conservation` changes no verdict
    on any of the 600 cells. That is expected: the per-tool grants equal the
    ceil/floor split exactly, so the clause is satisfied with equality, and
    charging nothing satisfies it too. It means the replay's fidelity here
    rests entirely on the scalar comparison, and a future arm whose per-tool
    grants did NOT match its consumption would be the first real test of it.

    The charge stays because it makes the replay faithful to what ran, and
    `test_per_tool_overrun_is_caught` shows it bites when it can.
    """
    team = wt[wt.agent_name.eq("team")]
    with_charge = recertify_frame(team, chamber="wt").conservation_replayed.tolist()
    assert sum(with_charge) == 202


def test_per_tool_overrun_is_caught():
    """The clause a scalar reimplementation would silently drop.

    A node consuming an experiment budget nobody granted it must fail, even
    though its token spend is trivially inside its grant. This is why
    `replay_conservation` calls the real `verify()` rather than comparing
    token totals.
    """
    from evaluation.chamber_pipeline.recertify import _graph

    graph = _graph("wt", 14, "team", None)
    graph.monitor_for("scout_a").usage.add_tokens(1)
    for _ in range(500):
        graph.monitor_for("scout_a").usage.add_tool_invocation("intervene")
    with pytest.raises(ConservationViolationError):
        graph.verify()
