"""Guards for the production prompt classifier.

The classifier exists so a live calibration run can attribute token spend to
the KIND of call that produced it. `_C95_NEGOTIATE` was measured on LT and
never isolated on WT, so every WT `team` cell runs on LT's figure with its
conservation voided -- 300 cells with `conservation_certified = None`. The
only way to measure it is per-call attribution, and the only safe way to
attribute is a marker set proven exclusive across EVERY builder, not the four
the test-only version covered.
"""

from __future__ import annotations

import pytest

from evaluation.chamber_pipeline.llm_planner import (
    CALL_KIND_MARKERS,
    build_adjacency_prompt,
    build_batch_select_prompt,
    build_critique_prompt,
    build_negotiate_propose_prompt,
    build_negotiate_revise_prompt,
    build_planner_select_prompt,
    build_reasoner_select_prompt,
    build_reconcile_prompt,
    build_revise_after_critique_prompt,
    build_scout_broad_prompt,
    build_scout_targeted_prompt,
    build_select_prompt,
    build_uncontracted_select_prompt,
    call_kind,
    is_negotiation,
)

MENU = ["uniform_a", "uniform_b", "uniform_c"]

# Every builder in `llm_planner`, with the kind its output must classify as.
# The four selection-role variants collapse to "select" on purpose: they
# differ only in system message, and cost attribution is about the call, not
# the persona.
ALL_BUILDERS: list[tuple[str, list[dict[str, str]], str]] = [
    ("select", build_select_prompt(MENU, 3, ["uniform_a"]), "select"),
    ("planner", build_planner_select_prompt(MENU, 3, None), "select"),
    ("reasoner", build_reasoner_select_prompt(MENU, 3, ["uniform_a"]), "select"),
    ("scout_broad", build_scout_broad_prompt(MENU, 3, None), "select"),
    ("scout_targeted", build_scout_targeted_prompt(MENU, 3, None), "select"),
    (
        "uncontracted",
        build_uncontracted_select_prompt(MENU, 3, ["uniform_a"]),
        "select_uncontracted",
    ),
    ("reconcile", build_reconcile_prompt(["uniform_a"], ["uniform_b"]), "reconcile"),
    ("batch", build_batch_select_prompt(MENU, 2), "batch_select"),
    ("critique", build_critique_prompt(MENU, 2, ["uniform_a"]), "critique"),
    (
        "revise_after_critique",
        build_revise_after_critique_prompt(MENU, 2, ["uniform_a"], "swap b for c"),
        "revise_after_critique",
    ),
    (
        "negotiate_propose",
        build_negotiate_propose_prompt(MENU, 2, "A"),
        "negotiate_propose",
    ),
    (
        "negotiate_revise",
        build_negotiate_revise_prompt(MENU, 2, ["uniform_a"], ["uniform_b"]),
        "negotiate_revise",
    ),
    ("adjacency", build_adjacency_prompt(["x", "y"], 3, None), "adjacency"),
    (
        "adjacency_summary",
        build_adjacency_prompt(["x", "y"], 3, "| exp | x | y |"),
        "adjacency",
    ),
]


@pytest.mark.parametrize("label,messages,want", ALL_BUILDERS, ids=[b[0] for b in ALL_BUILDERS])
def test_every_builder_classifies_to_its_kind(label, messages, want):
    assert call_kind(messages) == want


@pytest.mark.parametrize("label,messages,want", ALL_BUILDERS, ids=[b[0] for b in ALL_BUILDERS])
def test_exactly_one_marker_matches_each_builder(label, messages, want):
    """Exclusivity, tested directly rather than inferred from first-match.

    `call_kind` returns on its first hit, so a genuinely ambiguous prompt
    still classifies "correctly" whenever the right rule happens to sit
    first. That is how the test-only classifier let the reconcile prompt --
    whose system message says "one of two designers" -- match the negotiation
    marker for as long as it did. Counting matches makes the ambiguity itself
    the failure, independent of rule order.
    """
    body = " ".join(m["content"] for m in messages)
    hits = sorted(kind for kind, marker in CALL_KIND_MARKERS if marker in body)
    assert hits == [want], f"{label}: matched {hits}, want exactly ['{want}']"


def test_no_builder_is_unknown():
    """An unclassified call silently lands in no bucket and vanishes.

    Token attribution sums per kind; a call classified "unknown" is spend
    that exists in the cell total but in none of the calibration inputs,
    which reads as a cheaper call than it was.
    """
    assert all(call_kind(m) != "unknown" for _, m, _ in ALL_BUILDERS)


def test_unrecognised_prompt_is_unknown_not_misfiled():
    assert call_kind([{"role": "user", "content": "hello"}]) == "unknown"


def test_is_negotiation_excludes_reconcile():
    """The distinction `_C95_NEGOTIATE` is measured across.

    Reconcile is the aggregator's call and is provisioned by `a95`;
    negotiation is the scouts' and is provisioned by `c95_negotiate`.
    Merging them would fold aggregator spend into the scouts' constant.
    """
    assert is_negotiation(build_negotiate_propose_prompt(MENU, 2, "A"))
    assert is_negotiation(build_negotiate_revise_prompt(MENU, 2, ["uniform_a"], []))
    assert not is_negotiation(build_reconcile_prompt(["uniform_a"], ["uniform_b"]))
    assert not is_negotiation(build_select_prompt(MENU, 3, None))


def test_markers_are_distinct_strings():
    """No marker may be a substring of another.

    Two markers where one contains the other are not independent rules: the
    broader one matches everywhere the narrower does, so exclusivity depends
    entirely on rule order -- the property the counting test above exists to
    remove.
    """
    markers = [m for _, m in CALL_KIND_MARKERS]
    for a in markers:
        for b in markers:
            if a is not b:
                assert a not in b, f"marker {a!r} is a substring of {b!r}"
