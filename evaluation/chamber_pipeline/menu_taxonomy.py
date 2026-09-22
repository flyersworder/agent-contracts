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


def repair_infeasible_split(
    owner: dict[str, str],
    groups: dict[str, list[str]],
    claimed_a: set[str],
    claimed_b: set[str],
    budget_a: int,
    budget_b: int,
    seed_tag: str,
) -> int:
    """Move whole variables between pools until each exceeds its budget.

    Runs only where the claims-first deal left a pool at or below its budget,
    and returns the number of variables moved, so 0 means the deal was
    feasible and was left exactly as dealt. On LT this never fires at k=30 (a
    scout taking the 15 fattest variables still leaves 20 entries against a
    budget of 15) but does at k=45: each scout needs more than 22 of the 59
    entries, the negotiation's claims cover most of the menu, and A wins
    ties, so a lopsided claim can leave B with 10 entries. Measured on the
    2026-09-21 pre-launch probe; without the repair the arm is undefined in
    those cells and the cells that survive are a selected sample.

    Donation order, least-contested first, each tier in a seeded shuffle:
    unclaimed variables (moving one overrides no claim), then variables BOTH
    scouts claimed (the receiver wanted it too), then variables only the
    donor claimed. A move is taken only if the donor stays above its own
    budget. Mutates `owner`; the caller's feasibility check still raises if
    no sequence of moves suffices.
    """
    budgets = {"a": budget_a, "b": budget_b}
    claimed = {"a": claimed_a, "b": claimed_b}

    def size(side: str) -> int:
        return sum(len(groups[v]) for v, o in owner.items() if o == side)

    rng = _random.Random(seed_tag)
    moved = 0
    for receiver, donor in (("a", "b"), ("b", "a")):
        if size(receiver) > budgets[receiver]:
            continue
        candidates = sorted(v for v, o in owner.items() if o == donor)
        rng.shuffle(candidates)

        def tier(v: str, donor: str = donor, receiver: str = receiver) -> int:
            if v not in claimed[donor] and v not in claimed[receiver]:
                return 0
            return 1 if v in claimed[receiver] else 2

        candidates.sort(key=tier)  # stable: keeps the shuffle within a tier
        for variable in candidates:
            if size(receiver) > budgets[receiver]:
                break
            if size(donor) - len(groups[variable]) <= budgets[donor]:
                continue
            owner[variable] = receiver
            moved += 1
    return moved


def partition_pools_by_variable(
    menu: list[str],
    claim_a: list[str],
    claim_b: list[str],
    budget_a: int,
    budget_b: int,
    seed: int,
    stats: dict[str, int] | None = None,
) -> tuple[set[str], set[str]]:
    """Split the menu so every entry of a variable lands in ONE scout's pool.

    The name-level partition `team_agents` ships makes the two pools disjoint
    as SETS OF EXPERIMENTS, which is what `overlap_frac` measures and why it
    reads 0.0 in every cell. It does not stop both scouts buying the same
    VARIABLE at different strengths, and measurement says that is where the
    cost is: `team` reaches 23.4 distinct variables against the loop's 27.9,
    duplicating 5.6 -- **statistically indistinguishable from splitting the
    menu at random** (null model: 4.11 +- 1.51 over 8,000 draws). The
    negotiation partitions competently, over the wrong object.

    Partitioning by variable makes cross-scout duplication structurally
    impossible rather than merely discouraged. Everything else about the arm
    -- the propose/revise calls, A-wins-ties, the budgets -- is held fixed, so
    a contrast against `team` isolates the granularity of the split and
    nothing else.

    Assignment order, and each step exists for a measured reason:

    1. **Claims first, mapped to variables.** A variable claimed by either
       scout goes to that scout; one claimed by both goes to A, matching the
       name-level backstop it replaces.
    2. **Free variables dealt greedily to whichever pool holds fewer
       ENTRIES.** Not alternately: variables carry one to three entries each,
       so alternating deals a balanced count of variables and an unbalanced
       count of picks. The shuffle is seeded, because the menu is grouped by
       family and an unshuffled deal hands the same families to the same scout
       in every seed.
    3. **An infeasible deal is repaired** by `repair_infeasible_split`,
       which moves the least-contested variables until both pools exceed
       their budgets; the count lands in `stats["partition_repairs"]` when a
       dict is passed. Added 2026-09-21 for k=45, where the claims alone
       left a pool below budget; a deal that was feasible is untouched.
    4. **Feasibility is asserted, not hoped for.** A pool at or below its
       budget makes the selection loop inert -- every name gets queried and
       the LLM's choice cannot matter -- which is the degeneracy the
       name-level code reserves shortfalls to avoid. On LT the worst case is
       comfortable (a scout taking the 15 fattest variables leaves 20 entries
       for the other 15), but "comfortable on today's menu" is not a
       guarantee, so it raises.

    Returns:
        `(pool_a, pool_b)`, disjoint, together covering the whole menu.

    Raises:
        ValueError: If either pool cannot exceed its budget.
    """
    groups = group_by_variable(menu)
    owner: dict[str, str] = {}
    for name in claim_a:
        owner[experiment_variable(name)] = "a"
    for name in claim_b:
        owner.setdefault(experiment_variable(name), "b")

    free = [v for v in groups if v not in owner]
    _random.Random(f"varsplit:{seed}").shuffle(free)
    size = {
        "a": sum(len(groups[v]) for v, o in owner.items() if o == "a"),
        "b": sum(len(groups[v]) for v, o in owner.items() if o == "b"),
    }
    for variable in free:
        side = "a" if size["a"] <= size["b"] else "b"
        owner[variable] = side
        size[side] += len(groups[variable])

    repairs = repair_infeasible_split(
        owner,
        groups,
        {experiment_variable(n) for n in claim_a},
        {experiment_variable(n) for n in claim_b},
        budget_a,
        budget_b,
        f"varsplit-repair:{seed}",
    )
    if stats is not None:
        stats["partition_repairs"] = repairs

    pool_a = {n for v, o in owner.items() if o == "a" for n in groups[v]}
    pool_b = {n for v, o in owner.items() if o == "b" for n in groups[v]}
    for label, pool, budget in (("a", pool_a, budget_a), ("b", pool_b, budget_b)):
        if len(pool) <= budget:
            raise ValueError(
                f"variable partition left scout_{label} a pool of {len(pool)} "
                f"entries against a budget of {budget}; at or below budget the "
                "selection loop is inert because every name gets queried"
            )
    return pool_a, pool_b


def partition_pools_by_variable_n(
    menu: list[str],
    budgets: tuple[int, ...],
    seed: int,
) -> tuple[set[str], ...]:
    """`partition_pools_by_variable` for n BLIND scouts (no claims).

    Same seeded shuffle, same greedy deal of each variable to the pool with
    the fewest ENTRIES (ties to the earlier scout), same feasibility check.
    With empty claims and two budgets it returns exactly what the two-pool
    function returns, which is what makes the blind two-scout varsplit the
    clean control for the three-scout one.
    """
    n = len(budgets)
    groups = group_by_variable(menu)
    free = list(groups)
    _random.Random(f"varsplit:{seed}").shuffle(free)
    owner: dict[str, int] = {}
    size = [0] * n
    for variable in free:
        i = min(range(n), key=lambda j: (size[j], j))
        owner[variable] = i
        size[i] += len(groups[variable])
    pools = tuple({m for v, o in owner.items() if o == i for m in groups[v]} for i in range(n))
    for i, (pool, budget) in enumerate(zip(pools, budgets, strict=True)):
        if len(pool) <= budget:
            raise ValueError(
                f"variable partition left scout_{chr(ord('a') + i)} a pool of "
                f"{len(pool)} entries against a budget of {budget}; at or below "
                "budget the selection loop is inert because every name gets queried"
            )
    return pools


__all__ = [
    "STRENGTHS",
    "coverage_ordered",
    "experiment_strength",
    "experiment_variable",
    "group_by_variable",
    "partition_pools_by_variable",
    "partition_pools_by_variable_n",
    "repair_infeasible_split",
]
