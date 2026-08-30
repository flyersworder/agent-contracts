"""LT menu taxonomy: which VARIABLE a menu entry intervenes on.

The LT menu is 59 entries over 30 variables. Ten variables carry one entry,
eleven carry two, nine carry three (`weak` / `mid` / `strong`), and one entry
(`uniform_reference`) is purely observational. So a k=30 budget can touch
anywhere from **11 to 30 distinct variables** depending only on how the picks
are distributed -- a range the LLM arms never explore on their own (the loop's
observed range is 25-30).

That gap is why this module exists twice over: `analyze_mechanism` reads it to
count what an arm bought, and `agents` uses it to build the two arms that
manipulate coverage directly. Both must agree on the parse, so it lives in one
place.

**LT-specific by construction.** WT names (`validate_v_2`,
`validate_load_out_pressure_intake`) do not carry a strength suffix and several
name multi-component regimes, so nothing here applies to them. The agents built
on it are registered `chambers=("lt",)`.
"""

from __future__ import annotations

import random as _random
from collections import defaultdict

STRENGTHS = ("weak", "mid", "strong")
_PREFIX = "uniform_"


def experiment_variable(name: str) -> str:
    """`uniform_osr_c_strong` -> `osr_c`; `uniform_reference` -> `reference`.

    The split is on the KNOWN strength suffix, never on a token count, because
    LT variable names contain underscores (`osr_angle_1`, `diode_ir_3`) and a
    `rsplit("_", 1)` eats a real token on any entry carrying no strength.

    **Measured, so as not to overclaim**: on the current LT release the two
    rules agree on all 59 names and both yield 30 variables, because the only
    strengthless entry (`uniform_reference`) has no underscore left to lose.
    So no recorded number depends on this choice today. The suffix rule is kept
    because it is the rule that stays correct if the menu gains a strengthless
    multi-token entry, not because it rescued a live defect.
    """
    stem = name[len(_PREFIX) :] if name.startswith(_PREFIX) else name
    for strength in STRENGTHS:
        if stem.endswith("_" + strength):
            return stem[: -(len(strength) + 1)]
    return stem


def experiment_strength(name: str) -> str:
    """The intervention strength, or `none` for the observational entry."""
    for strength in STRENGTHS:
        if name.endswith("_" + strength):
            return strength
    return "none"


def group_by_variable(menu: list[str]) -> dict[str, list[str]]:
    """Menu entries bucketed by the variable they intervene on, menu order kept."""
    groups: dict[str, list[str]] = defaultdict(list)
    for name in menu:
        groups[experiment_variable(name)].append(name)
    return dict(groups)


def coverage_ordered(
    menu: list[str],
    budget: int,
    seed: int,
    *,
    maximize: bool,
    exclude_strengths: tuple[str, ...] = (),
) -> list[str]:
    """Exactly `min(budget, len(menu))` names, chosen to widen or narrow coverage.

    `maximize=True` takes one entry from every variable before taking a second
    from any, which attains `min(budget, n_variables)` -- the optimum, since no
    selection of `budget` entries can touch more variables than it has picks.

    `maximize=False` takes variables fattest-first and exhausts each before
    moving on, which attains the minimum for the same reason read backwards:
    spending a pick on a fresh variable is the only way to raise the count, so
    the fewest fresh variables come from spending as many picks as possible
    inside each.

    `exclude_strengths` drops levels from the menu before ordering. It exists
    because the unrestricted manipulation is CONFOUNDED: the fattest variables
    are exactly the ones carrying a `weak` level, so `maximize=False` buys 9.0
    weak interventions where `maximize=True` buys 3.2, and across the first
    90-cell sweep `n_variables` and `n_weak` correlated at -0.89. A weak
    intervention perturbs less and carries less signal, so the two channels
    could not be told apart. Excluding `weak` narrows the LT k=30 span from
    11-30 to 15-30 and buys a manipulation that varies breadth alone.

    Both shuffle on a seeded RNG before ordering -- within each variable, and
    across variables. The menu is grouped by variable family and strength, so
    resolving ties by menu order would hand every seed the same families and
    make the arm's blind spot identical in all of them. That is the defect
    `_capped_claim` documents in `agents`, and it applies with more force here
    because these arms have no LLM to perturb the choice.
    """
    if exclude_strengths:
        menu = [n for n in menu if experiment_strength(n) not in exclude_strengths]
    rng = _random.Random(f"coverage:{seed}")
    items = []
    for variable, names in group_by_variable(menu).items():
        shuffled = list(names)
        rng.shuffle(shuffled)
        items.append((variable, shuffled))
    rng.shuffle(items)

    chosen: list[str] = []
    if maximize:
        depth = 0
        while len(chosen) < budget:
            progressed = False
            for _, names in items:
                if depth >= len(names):
                    continue
                chosen.append(names[depth])
                progressed = True
                if len(chosen) == budget:
                    return chosen
            if not progressed:
                break
            depth += 1
        return chosen

    # Stable sort: variables of equal size keep the shuffled order above.
    items.sort(key=lambda kv: -len(kv[1]))
    for _, names in items:
        for name in names:
            chosen.append(name)
            if len(chosen) == budget:
                return chosen
    return chosen


__all__ = [
    "STRENGTHS",
    "coverage_ordered",
    "experiment_strength",
    "experiment_variable",
    "group_by_variable",
]
