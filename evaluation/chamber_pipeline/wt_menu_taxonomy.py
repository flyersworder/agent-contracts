"""WT menu taxonomy: which VARIABLE a wind-tunnel menu entry intervenes on.

The LT parse (`menu_taxonomy`) splits on a known strength suffix. WT names
carry no strength, so a different rule is needed — and the rule cannot be a
token count, because WT variable names contain underscores
(`osr_downwind`, `load_out`, `pressure_intake`).

**The rule: strip `validate_`, then take the LONGEST node name that is a
prefix of the remainder.** That is the manipulated variable; anything after it
names the regime or the co-varied sensor. `validate_load_out_pressure_intake`
resolves to `load_out`, not to `load` and not to `load_out_pressure_intake`.

Two guards, and on TODAY's menu they are redundant with each other — a fact
worth stating so neither is removed as dead weight. The underscore boundary
alone separates `osr_in` from `osr_intake` (`osr_intake` does not start with
`osr_in_`), and longest-match alone would resolve the same pair correctly
without it. Each is load-bearing only in a case the current menu does not
contain: longest-match when one node name plus an underscore prefixes another
(`a_b` vs `a_b_c`), the boundary when a node name is a bare prefix of an entry
(`load` vs `loadX`). Both cases are pinned in the tests, which were themselves
mutation-checked after a first version passed with either guard removed.

**How WT differs from LT, and why it matters for what a coverage arm means.**
On LT the multi-entry variables are the ones carrying weak/mid/strong, so
"maximise distinct variables" trades intervention strength for breadth. On WT
the only multi-entry variables are `hatch` (3), `load_in` (3) and `load_out`
(4) — and those are precisely the highest out-degree drivers in the ground
truth (6, 8 and 8 edges). Every one of the other 18 entries is a single-entry
apparatus setting with out-degree 1. So on WT, maximising distinct variables
spreads the budget across trivial settings and *away* from the real drivers.
The arm is the same rule; what it buys is structurally opposite. Expect it to
behave differently, and do not read an LT-calibrated intuition onto it.
"""

from __future__ import annotations

import random as _random
from collections import defaultdict

_PREFIX = "validate_"


def experiment_variable(name: str, node_names: list[str]) -> str:
    """Return the variable `name` intervenes on, by longest node-name prefix.

    Args:
        name: A menu entry, e.g. `validate_load_out_pressure_intake`.
        node_names: The chamber's node names — the candidate variables.

    Returns:
        The matched node name.

    Raises:
        ValueError: If no node name prefixes the entry. Raising rather than
            falling back to the raw stem is deliberate: a silent fallback
            would invent a variable, inflate the distinct count, and make a
            coverage arm look better than it is.
    """
    stem = name[len(_PREFIX) :] if name.startswith(_PREFIX) else name
    matches = [n for n in node_names if stem == n or stem.startswith(n + "_")]
    if not matches:
        raise ValueError(
            f"no node name prefixes menu entry {name!r} (stem {stem!r}); "
            "the WT parse cannot invent a variable without inflating the "
            "distinct-variable count that coverage arms are scored on"
        )
    return max(matches, key=len)


def group_by_variable(menu: list[str], node_names: list[str]) -> dict[str, list[str]]:
    """Map variable -> the menu entries that intervene on it, order preserved."""
    groups: dict[str, list[str]] = defaultdict(list)
    for entry in menu:
        groups[experiment_variable(entry, node_names)].append(entry)
    return dict(groups)


def coverage_ordered(
    menu: list[str],
    budget: int,
    seed: int,
    node_names: list[str],
    *,
    maximize: bool,
) -> list[str]:
    """Order `budget` entries to maximise or minimise distinct variables.

    Mirrors `menu_taxonomy.coverage_ordered` exactly, minus the strength
    filter, which has no WT analogue. Both the within-variable order and the
    variable order are shuffled from `seed` so that ties are broken
    reproducibly without favouring the menu's own ordering.
    """
    rng = _random.Random(f"wtcoverage:{seed}")
    items = []
    for variable, names in group_by_variable(menu, node_names).items():
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

    items.sort(key=lambda kv: -len(kv[1]))
    for _, names in items:
        for name in names:
            chosen.append(name)
            if len(chosen) == budget:
                return chosen
    return chosen
