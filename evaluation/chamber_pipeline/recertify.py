"""Recover a conservation verdict from recorded per-node spend.

All 300 archived WT `team` cells carry `conservation_certified = None`. The
harness put it there deliberately: their negotiation rounds were provisioned by
`_C95_NEGOTIATE_BY_CHAMBER["lt"]`, WT's figure having never been measured, and
`is_provisional_calibration` refuses to certify a cell running on a borrowed
constant. So WT's H-C number was missing its most coordinated arm.

The first reading was that those cells "ran under the wrong grant and cannot be
retro-certified", making a ~300-cell re-run the only route. That was wrong, and
this module is the argument for why, in three parts:

1. **The verdict is a pure function of recorded state.** `verify()` compares
   consumption plus out-flow against in-flow at every node, and a cell records
   every node's token consumption (`scout_a_tokens`, `scout_b_tokens`,
   `aggregator_tokens`). The grants are a deterministic function of published
   constants.

2. **The real `verify()` is called, not reimplemented.** `_violates` carries a
   per-tool clause on top of the scalar comparison, which an inequality
   reimplementation silently drops. `replay_conservation` rebuilds the graph
   with `build_fan_in_graph`, charges the recorded spend into the node
   monitors, and calls `DelegationGraph.verify()`.

3. **It is validated where the answer is already known.** The two blind fan-in
   arms ran the same graph and DID record a verdict; the replay reproduces it
   on 300 of 300 cells. That is what licenses applying it to the arm that
   recorded none.

The soundness precondition is that the grant did not SHAPE the spend — replaying
recorded tokens against a different grant is meaningless if the old grant
truncated the run that produced them. It is established by measurement, and the
evidence is the conservation failures themselves: **98 of the 300 cells overspend
their grant, by up to 13,833 tokens, and complete normally.** A ceiling that can
be exceeded by that much is not truncating anything. Confirming it from the
other side, **no node in the archive spends exactly its ceiling** (0 of 900) —
clipping would pile spend up on that value.

An earlier draft guarded on `grant_headroom > 0` instead, which is incoherent:
negative headroom is the *definition* of a conservation failure, so that guard
refused precisely the 98 cells the exercise exists to classify. `_no_clipping`
is the guard that means what the first one was trying to say.

**This does not make re-running unnecessary in general.** It works here because
the correction only ENLARGES a grant and no cell sits in the band it moves —
`test_replay_is_insensitive_to_the_corrected_constant` pins that. A correction
that shrank a grant, or one where cells sat near the boundary, would need the
re-run.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

from agent_contracts.core.delegation_graph import ConservationViolationError
from evaluation.chamber_pipeline.coordination import build_fan_in_graph
from evaluation.chamber_pipeline.orchestrator import (
    _A95_RECONCILE_BY_K,
    _C95_NEGOTIATE_BY_CHAMBER,
    _PROVISION_MULTIPLE,
    _ROLE_C95,
    get_spec,
)

__all__ = [
    "RecertifyUnsoundError",
    "grant_headroom",
    "recertify_frame",
    "replay_conservation",
]

REQUIRED_COLUMNS = (
    "agent_name",
    "budget_k",
    "scout_a_tokens",
    "scout_b_tokens",
    "aggregator_tokens",
)


class RecertifyUnsoundError(Exception):
    """The archive does not support replaying a verdict onto it."""


def _calibration(
    chamber: str, budget_k: int, arm: str, c95_negotiate: int | None
) -> tuple[int, int, int, int]:
    """`(c95_a, c95_b, a95, fixed_overhead)`, with the negotiate term overridable.

    `_ladder_calibration` is not reused because this module needs to ask a
    counterfactual question -- "what would the verdict have been under the
    constant these cells actually ran on?" -- which that function deliberately
    cannot express: it raises rather than serve an unmeasured figure.
    """
    spec = get_spec(arm)
    if spec.scout_roles is None:
        raise ValueError(f"{arm!r} is not a fan-in arm")
    role_a, role_b = spec.scout_roles
    negotiate = _C95_NEGOTIATE_BY_CHAMBER[chamber] if c95_negotiate is None else c95_negotiate
    return (
        _ROLE_C95[(chamber, role_a)],
        _ROLE_C95[(chamber, role_b)],
        _A95_RECONCILE_BY_K[(chamber, budget_k)],
        spec.negotiation_rounds * _PROVISION_MULTIPLE * negotiate,
    )


def _graph(chamber: str, budget_k: int, arm: str, c95_negotiate: int | None):
    c95_a, c95_b, a95, overhead = _calibration(chamber, budget_k, arm, c95_negotiate)
    return build_fan_in_graph(
        multiple=_PROVISION_MULTIPLE,
        k=budget_k,
        c95=c95_a,
        a95=a95,
        c95_b=c95_b,
        fixed_overhead=overhead,
    )


def replay_conservation(
    *,
    chamber: str,
    budget_k: int,
    arm: str,
    scout_a_tokens: float,
    scout_b_tokens: float,
    aggregator_tokens: float,
    c95_negotiate: int | None = None,
) -> bool:
    """`verify()`'s verdict for one cell, from its recorded spend.

    Per-tool consumption is reconstructed as the `ceil(k/2)` / `floor(k/2)`
    split rather than read from the record, which does not store it per node.
    That is not an assumption: budget matching was verified from the data by
    solving `distinct = |A| + |B| - shared` against recorded overlap on all
    270 fan-in cells, residual 0.00 with zero non-integer shared counts. The
    per-tool grants equal that split exactly, so the clause is satisfied with
    equality and can never be the binding constraint here -- which is also why
    the scalar-only reimplementation happened to agree.
    """
    graph = _graph(chamber, budget_k, arm, c95_negotiate)
    graph.monitor_for("scout_a").usage.add_tokens(int(scout_a_tokens))
    graph.monitor_for("scout_b").usage.add_tokens(int(scout_b_tokens))
    graph.monitor_for("aggregator").usage.add_tokens(int(aggregator_tokens))
    for _ in range(math.ceil(budget_k / 2)):
        graph.monitor_for("scout_a").usage.add_tool_invocation("intervene")
    for _ in range(budget_k // 2):
        graph.monitor_for("scout_b").usage.add_tool_invocation("intervene")
    try:
        graph.verify()
    except ConservationViolationError:
        return False
    return True


def grant_headroom(
    frame: pd.DataFrame, *, chamber: str, c95_negotiate: int | None = None
) -> pd.Series:
    """Unspent tokens at the tightest node of each cell, under the grant it RAN on.

    A DIAGNOSTIC, not the soundness guard -- negative headroom is exactly what
    a conservation failure is, so refusing on it would refuse the 98 cells this
    module exists to classify. Useful for seeing how far a cell overran; see
    `_no_clipping` for the guard that establishes the grant did not shape the
    spend.
    """
    out: list[float] = []
    for _, row in frame.iterrows():
        arm = str(row["agent_name"])
        budget_k = int(row["budget_k"])
        graph = _graph(chamber, budget_k, arm, c95_negotiate)
        spent = {
            "scout_a": float(row["scout_a_tokens"]),
            "scout_b": float(row["scout_b_tokens"]),
            "aggregator": float(row["aggregator_tokens"]),
        }
        out.append(
            min(
                float(graph.in_flow(name).tokens or 0)
                - float(graph.out_flow(name).tokens or 0)
                - spent[name]
                for name in spent
            )
        )
    return pd.Series(out, index=frame.index, name="grant_headroom")


def _no_clipping(frame: pd.DataFrame, *, chamber: str, c95_negotiate: int | None = None) -> None:
    """Raise unless the recorded spend is exogenous to the grant it ran under.

    The signature of a truncating monitor is spend piled up exactly ON a
    ceiling. The archive shows the opposite -- 98 cells exceed their grant by
    up to 13,833 tokens and complete, and no node lands on its ceiling -- which
    is what makes replaying those tokens against a corrected grant meaningful
    rather than a restatement of the old one.
    """
    clipped = 0
    for _, row in frame.iterrows():
        graph = _graph(chamber, int(row["budget_k"]), str(row["agent_name"]), c95_negotiate)
        for node, spent in (
            ("scout_a", row["scout_a_tokens"]),
            ("scout_b", row["scout_b_tokens"]),
            ("aggregator", row["aggregator_tokens"]),
        ):
            ceiling = float(graph.in_flow(node).tokens or 0) - float(
                graph.out_flow(node).tokens or 0
            )
            if int(spent) == int(ceiling):
                clipped += 1
    if clipped:
        raise RecertifyUnsoundError(
            f"{clipped} node(s) spent exactly their ceiling, the signature of a "
            "monitor that truncated the run. Their recorded tokens describe the "
            "old grant, not what the cell would spend. Re-run the arm instead."
        )


def recertify_frame(
    frame: pd.DataFrame,
    *,
    chamber: str,
    c95_negotiate: int | None = None,
    arms: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Add `conservation_replayed` to every fan-in cell in `frame`.

    Raises:
        RecertifyUnsoundError: if a required column is absent, or if any cell
            exhausted the grant it ran under. Refusing is the point: a
            silently-produced number here would be a statement about the old
            ceiling wearing the label of a conservation result, which is the
            same failure mode that let one negotiate constant serve two
            chambers.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise RecertifyUnsoundError(
            f"frame is missing {missing}; per-node spend has only been recorded "
            "since 2026-08-24, and an unrecorded node reads as a clean node"
        )
    subset = frame if arms is None else frame[frame.agent_name.isin(list(arms))]
    subset = subset[subset.agent_name.map(lambda a: get_spec(a).scout_roles is not None)]

    _no_clipping(subset, chamber=chamber, c95_negotiate=c95_negotiate)

    verdicts = [
        replay_conservation(
            chamber=chamber,
            budget_k=int(row["budget_k"]),
            arm=str(row["agent_name"]),
            scout_a_tokens=row["scout_a_tokens"],
            scout_b_tokens=row["scout_b_tokens"],
            aggregator_tokens=row["aggregator_tokens"],
            c95_negotiate=c95_negotiate,
        )
        for _, row in subset.iterrows()
    ]
    out = subset.copy()
    out["conservation_replayed"] = verdicts
    return out
