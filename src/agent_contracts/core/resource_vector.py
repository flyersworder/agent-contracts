"""Resource arithmetic for flow-conservation delegation.

A ``ResourceVector`` is an amount across every conserved resource dimension.
``None`` in a scalar dimension means *unbounded*, never zero.

Per-tool semantics mirror ``AllocationRecord``: only tools the budget side
explicitly constrains participate in comparison. A tool absent from a budget
is unconstrained; a tool absent from an amount counts as zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, cast

from agent_contracts.core.contract import ResourceConstraints

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agent_contracts.core.monitor import ResourceUsage

# Absolute tolerance for cost-axis (USD, float) conservation comparisons.
# `agent_contracts.core.delegation` defines the identical constant for the
# tree conservation law (0.3.2 fix). It is duplicated here rather than
# imported: `delegation.py` is intentionally untouched by the DAG
# generalization (see CHANGELOG 0.4.0, "core/delegation.py is untouched"),
# and importing its private constant would couple this module to that one's
# internals. `test_cost_epsilon_matches_delegation_module` pins the two
# constants equal so they cannot silently drift apart.
_COST_EPSILON = 1e-9


def _add(a: Any, b: Any) -> Any:
    """Add two values, treating None as unbounded."""
    if a is None or b is None:
        return None
    return a + b


def _sub(a: Any, b: Any) -> Any:
    """Subtract two values, treating None as unbounded."""
    if a is None:
        return None
    if b is None:
        raise ValueError("cannot subtract an unbounded amount from a finite budget")
    return a - b


def _le(a: Any, b: Any, tolerance: float = 0.0) -> bool:
    """Compare two values, treating None as unbounded.

    ``tolerance`` absorbs IEEE-754 representation error on the cost axis
    (e.g. 0.1 + 0.1 + 0.1 != 0.3). The integer-valued axes must keep exact
    comparison and pass the default of 0.0.
    """
    if b is None:
        return True
    if a is None:
        return False
    return bool(a <= b + tolerance)


@dataclass(frozen=True)
class ResourceVector:
    """An amount across conserved resource dimensions. ``None`` == unbounded."""

    tokens: int | None = None
    cost_usd: float | None = None
    tool_invocations: int | None = None
    iterations: int | None = None
    per_tool: Mapping[str, int] = field(default_factory=dict)

    ZERO: ClassVar[ResourceVector]

    def __add__(self, other: ResourceVector) -> ResourceVector:
        per_tool = dict(self.per_tool)
        for name, count in other.per_tool.items():
            per_tool[name] = per_tool.get(name, 0) + count
        return ResourceVector(
            tokens=cast("int | None", _add(self.tokens, other.tokens)),
            cost_usd=cast("float | None", _add(self.cost_usd, other.cost_usd)),
            tool_invocations=cast(
                "int | None", _add(self.tool_invocations, other.tool_invocations)
            ),
            iterations=cast("int | None", _add(self.iterations, other.iterations)),
            per_tool=per_tool,
        )

    def __sub__(self, other: ResourceVector) -> ResourceVector:
        per_tool = dict(self.per_tool)
        for name, count in other.per_tool.items():
            per_tool[name] = per_tool.get(name, 0) - count
        return ResourceVector(
            tokens=cast("int | None", _sub(self.tokens, other.tokens)),
            cost_usd=cast("float | None", _sub(self.cost_usd, other.cost_usd)),
            tool_invocations=cast(
                "int | None", _sub(self.tool_invocations, other.tool_invocations)
            ),
            iterations=cast("int | None", _sub(self.iterations, other.iterations)),
            per_tool=per_tool,
        )

    def __le__(self, other: ResourceVector) -> bool:
        scalars_ok = (
            _le(self.tokens, other.tokens)
            and _le(self.cost_usd, other.cost_usd, tolerance=_COST_EPSILON)
            and _le(self.tool_invocations, other.tool_invocations)
            and _le(self.iterations, other.iterations)
        )
        if not scalars_ok:
            return False
        # Only tools the budget side constrains participate.
        return all(self.per_tool.get(name, 0) <= limit for name, limit in other.per_tool.items())

    def is_finite(self) -> bool:
        """True when no scalar dimension is unbounded."""
        return all(
            value is not None
            for value in (self.tokens, self.cost_usd, self.tool_invocations, self.iterations)
        )

    @classmethod
    def from_constraints(cls, rc: ResourceConstraints) -> ResourceVector:
        return cls(
            tokens=rc.tokens,
            cost_usd=rc.cost_usd,
            tool_invocations=rc.tool_invocations,
            iterations=rc.iterations,
            per_tool=dict(rc.per_tool_limits),
        )

    @classmethod
    def from_usage(cls, usage: ResourceUsage) -> ResourceVector:
        return cls(
            tokens=usage.tokens,
            cost_usd=usage.cost_usd,
            tool_invocations=usage.tool_invocations,
            iterations=usage.iterations,
            per_tool=dict(usage.tool_usage_by_name),
        )

    def to_constraints(self) -> ResourceConstraints:
        return ResourceConstraints(
            tokens=self.tokens,
            cost_usd=self.cost_usd,
            tool_invocations=self.tool_invocations,
            iterations=self.iterations,
            per_tool_limits=dict(self.per_tool),
        )


ResourceVector.ZERO = ResourceVector(
    tokens=0, cost_usd=0.0, tool_invocations=0, iterations=0, per_tool={}
)
