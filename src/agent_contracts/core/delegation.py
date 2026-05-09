"""Contract delegation with conservation laws (Whitepaper Section 6).

This module implements contracting as an agent capability, allowing agents to
create subcontracts and delegate work to other agents while respecting
conservation laws that ensure hierarchical budget discipline.

The key insight is that contracting itself is a capability: an agent with
this capability can spawn sub-agents with their own contracts, enabling
recursive delegation and dynamic team formation.

Conservation Law:
    For any parent contract with budget B, if it creates child contracts
    with budgets b_1, b_2, ..., b_k, the following must hold:

        Σ b_i ≤ B - used

    where 'used' is the parent's own consumption.

Example:
    >>> from agent_contracts import Contract, ResourceConstraints
    >>> from agent_contracts.core.delegation import ContractingCapability
    >>>
    >>> # Parent contract with 100K tokens
    >>> parent = Contract(
    ...     id="orchestrator",
    ...     resources=ResourceConstraints(tokens=100_000)
    ... )
    >>>
    >>> # Create delegation capability
    >>> delegator = ContractingCapability(parent)
    >>>
    >>> # Allocate budget to child agents
    >>> researcher_contract = delegator.create_subcontract(
    ...     name="researcher",
    ...     tokens=40_000,
    ...     description="Research the topic"
    ... )
    >>>
    >>> analyzer_contract = delegator.create_subcontract(
    ...     name="analyzer",
    ...     tokens=30_000,
    ...     description="Analyze findings"
    ... )
    >>>
    >>> # Check remaining budget
    >>> print(delegator.remaining_budget)  # 30_000 (100K - 40K - 30K)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.monitor import ResourceMonitor


class ConservationViolationError(Exception):
    """Raised when a budget allocation would violate conservation laws.

    Conservation laws ensure that the sum of child contract budgets
    cannot exceed the parent's remaining budget.
    """

    def __init__(
        self,
        message: str,
        requested: int,
        available: int,
        parent_id: str,
    ):
        self.requested = requested
        self.available = available
        self.parent_id = parent_id
        super().__init__(message)


@dataclass
class AllocationRecord:
    """Records a budget allocation to a child contract.

    Attributes:
        child_id: ID of the child contract
        child_name: Name of the child contract
        tokens_allocated: Tokens allocated to this child
        cost_allocated: Cost budget allocated to this child
        per_tool_limits_allocated: Per-tool call budgets allocated to this
            child (tool_name -> max_calls). Tools the parent doesn't
            constrain are still recorded here for audit-trail completeness;
            they just don't participate in conservation accounting.
        created_at: When the allocation was made
        child_contract: Reference to the child contract
    """

    child_id: str
    child_name: str
    tokens_allocated: int = 0
    cost_allocated: float = 0.0
    per_tool_limits_allocated: dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    child_contract: Contract | None = None


@dataclass
class DelegationSummary:
    """Summary of all delegations from a parent contract.

    Attributes:
        parent_id: ID of the parent contract
        parent_budget_tokens: Total token budget of parent
        parent_budget_cost: Total cost budget of parent
        parent_used_tokens: Tokens used by parent itself
        parent_used_cost: Cost used by parent itself
        total_allocated_tokens: Sum of tokens allocated to children
        total_allocated_cost: Sum of cost allocated to children
        remaining_tokens: Tokens available for further delegation
        remaining_cost: Cost available for further delegation
        allocations: List of all allocations made
        conservation_satisfied: Whether conservation law holds (across
            tokens, cost, AND per-tool limits)
        total_allocated_per_tool: Sum of per-tool budgets allocated across
            children, only for tools the parent itself constrains.
            Defaults to empty dict so callers that don't use per-tool
            delegation aren't broken by the new field.
    """

    parent_id: str
    parent_budget_tokens: int
    parent_budget_cost: float
    parent_used_tokens: int
    parent_used_cost: float
    total_allocated_tokens: int
    total_allocated_cost: float
    remaining_tokens: int
    remaining_cost: float
    allocations: list[AllocationRecord]
    conservation_satisfied: bool
    total_allocated_per_tool: dict[str, int] = field(default_factory=dict)


class ContractingCapability:
    """Capability that allows an agent to create subcontracts.

    This class implements "contracting as a capability" from the whitepaper.
    An agent with this capability can delegate work to other agents by
    creating subcontracts, with automatic enforcement of conservation laws.

    The conservation law ensures that:
        parent_used + Σ child_budgets ≤ parent_budget

    Attributes:
        parent_contract: The parent contract that governs this agent
        parent_monitor: Monitor tracking parent's resource consumption
        allocations: Record of all budget allocations to children
        reserve_ratio: Fraction of budget to reserve for coordination overhead

    Example:
        >>> capability = ContractingCapability(parent_contract, parent_monitor)
        >>> child = capability.create_subcontract("worker", tokens=10000)
        >>> print(capability.remaining_budget)
    """

    def __init__(
        self,
        parent_contract: Contract,
        parent_monitor: ResourceMonitor | None = None,
        reserve_ratio: float = 0.0,
    ):
        """Initialize contracting capability.

        Args:
            parent_contract: The parent contract providing the budget
            parent_monitor: Optional monitor tracking parent's usage.
                           If not provided, parent usage is assumed to be 0.
            reserve_ratio: Fraction of budget to reserve (0.0 to 0.5).
                          Default 0.0 means no automatic reserve.

        Raises:
            ValueError: If reserve_ratio is not in valid range
        """
        if not 0.0 <= reserve_ratio <= 0.5:
            raise ValueError(f"reserve_ratio must be between 0.0 and 0.5, got {reserve_ratio}")

        self.parent_contract = parent_contract
        self.parent_monitor = parent_monitor or ResourceMonitor(parent_contract.resources)
        self.reserve_ratio = reserve_ratio

        # Track allocations
        self._allocations: dict[str, AllocationRecord] = {}
        self._total_allocated_tokens: int = 0
        self._total_allocated_cost: float = 0.0
        # Per-tool delegated budgets, keyed by tool name. Only tools the
        # parent contract itself constrains via `per_tool_limits` are
        # subject to conservation; entries here for unconstrained tools
        # are still tracked for audit completeness but skipped during
        # conservation checks.
        self._total_allocated_per_tool: dict[str, int] = {}

    @property
    def parent_budget_tokens(self) -> int:
        """Total token budget of parent contract."""
        return self.parent_contract.resources.tokens or 0

    @property
    def parent_budget_cost(self) -> float:
        """Total cost budget of parent contract."""
        return self.parent_contract.resources.cost_usd or 0.0

    @property
    def parent_used_tokens(self) -> int:
        """Tokens consumed by parent itself."""
        return self.parent_monitor.usage.tokens

    @property
    def parent_used_cost(self) -> float:
        """Cost consumed by parent itself."""
        return self.parent_monitor.usage.cost_usd

    @property
    def reserved_tokens(self) -> int:
        """Tokens reserved for coordination overhead."""
        return int(self.parent_budget_tokens * self.reserve_ratio)

    @property
    def reserved_cost(self) -> float:
        """Cost reserved for coordination overhead."""
        return self.parent_budget_cost * self.reserve_ratio

    @property
    def remaining_tokens(self) -> int:
        """Tokens available for further delegation.

        Calculated as: parent_budget - parent_used - allocated - reserved
        """
        return max(
            0,
            (
                self.parent_budget_tokens
                - self.parent_used_tokens
                - self._total_allocated_tokens
                - self.reserved_tokens
            ),
        )

    @property
    def remaining_cost(self) -> float:
        """Cost budget available for further delegation."""
        return max(
            0.0,
            (
                self.parent_budget_cost
                - self.parent_used_cost
                - self._total_allocated_cost
                - self.reserved_cost
            ),
        )

    @property
    def remaining_budget(self) -> dict[str, int | float]:
        """Remaining budget available for delegation."""
        return {
            "tokens": self.remaining_tokens,
            "cost_usd": self.remaining_cost,
        }

    # ---------------- per-tool delegation accessors ----------------------
    #
    # These mirror the tokens / cost accessors above for the per_tool_limits
    # axis. Conservation semantics: a tool the parent contract does NOT
    # list in its `per_tool_limits` is treated as unconstrained — child
    # requests for it are allowed regardless of magnitude. This matches the
    # monitor's existing behaviour at `monitor.py:567` (a missing entry
    # means "no limit"), so the framework reads consistently across the
    # delegation and enforcement layers.

    def parent_per_tool_limit(self, tool_name: str) -> int | None:
        """Parent's per-tool limit for `tool_name`, or None if unconstrained.

        None means the parent does not budget this tool, so children may
        request any amount. A value of 0 means the parent forbids the
        tool — any positive child request will be rejected by
        `create_subcontract`.
        """
        return self.parent_contract.resources.per_tool_limits.get(tool_name)

    def parent_per_tool_used(self, tool_name: str) -> int:
        """Per-tool calls the parent itself has spent."""
        return self.parent_monitor.usage.get_tool_usage(tool_name)

    @property
    def total_allocated_per_tool(self) -> dict[str, int]:
        """Sum of per-tool budgets allocated across all children.

        Returns a copy — mutating the result does not affect internal state.
        """
        return dict(self._total_allocated_per_tool)

    def reserved_per_tool(self, tool_name: str) -> int:
        """Per-tool budget reserved for parent coordination overhead.

        Mirrors `reserved_tokens` for consistency. Floor-rounds because
        per-tool budgets are integer call counts. With reserve_ratio=0
        (the framework default) this is always 0.
        """
        limit = self.parent_per_tool_limit(tool_name)
        if limit is None:
            return 0
        return int(limit * self.reserve_ratio)

    def remaining_per_tool(self, tool_name: str) -> int | None:
        """Per-tool budget available for further delegation, or None if unconstrained.

        Calculated as: parent_limit - parent_used - allocated - reserved.
        Returns None when the parent has no constraint on this tool
        (signaling "child may request any amount").
        """
        limit = self.parent_per_tool_limit(tool_name)
        if limit is None:
            return None
        return max(
            0,
            limit
            - self.parent_per_tool_used(tool_name)
            - self._total_allocated_per_tool.get(tool_name, 0)
            - self.reserved_per_tool(tool_name),
        )

    @property
    def allocations(self) -> list[AllocationRecord]:
        """List of all budget allocations made."""
        return list(self._allocations.values())

    @property
    def child_contracts(self) -> list[Contract]:
        """List of all child contracts created."""
        return [
            alloc.child_contract
            for alloc in self._allocations.values()
            if alloc.child_contract is not None
        ]

    def can_allocate(
        self,
        tokens: int = 0,
        cost_usd: float = 0.0,
        per_tool_limits: dict[str, int] | None = None,
    ) -> bool:
        """Check if an allocation is possible without violating conservation.

        Args:
            tokens: Number of tokens to allocate
            cost_usd: Cost budget to allocate
            per_tool_limits: Optional per-tool call budgets. Tools the
                parent doesn't constrain are skipped (treated as unbounded).

        Returns:
            True if allocation would satisfy conservation law on every
            constrained axis.
        """
        if tokens > self.remaining_tokens:
            return False
        if cost_usd > self.remaining_cost:
            return False
        if per_tool_limits:
            for tool, requested in per_tool_limits.items():
                if requested <= 0:
                    continue
                remaining = self.remaining_per_tool(tool)
                if remaining is None:
                    # Parent doesn't constrain this tool — always OK.
                    continue
                if requested > remaining:
                    return False
        return True

    def create_subcontract(
        self,
        name: str,
        tokens: int = 0,
        cost_usd: float = 0.0,
        api_calls: int | None = None,
        iterations: int | None = None,
        tool_invocations: int | None = None,
        per_tool_limits: dict[str, int] | None = None,
        reasoning_tokens: int | None = None,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Contract:
        """Create a subcontract with budget allocated from parent.

        This method enforces the conservation law: the sum of all child
        budgets plus parent's own usage cannot exceed parent's total budget.

        Args:
            name: Name for the child contract (used in ID generation)
            tokens: Token budget to allocate to child
            cost_usd: Cost budget to allocate to child
            api_calls: Optional API call limit for child
            iterations: Optional iteration limit for child (maps to ADK max_llm_calls)
            tool_invocations: Optional total tool invocation limit for child
            per_tool_limits: Optional per-tool invocation limits (tool_name -> max_calls)
            reasoning_tokens: Optional reasoning/thinking token budget for child
                (communicates to agent how much thinking budget is allocated)
            description: Description of the child's task
            metadata: Optional metadata for the child contract

        Returns:
            A new Contract configured for the child agent

        Raises:
            ConservationViolationError: If allocation would violate conservation law
            ValueError: If name is empty or already used
        """
        # Validate name
        if not name:
            raise ValueError("Child contract name cannot be empty")

        child_id = f"{self.parent_contract.id}/{name}"
        if child_id in self._allocations:
            raise ValueError(f"Child contract '{name}' already exists")

        # Check conservation law for tokens
        if tokens > 0 and tokens > self.remaining_tokens:
            raise ConservationViolationError(
                message=(
                    f"Cannot allocate {tokens:,} tokens to '{name}'. "
                    f"Parent budget: {self.parent_budget_tokens:,}, "
                    f"Parent used: {self.parent_used_tokens:,}, "
                    f"Already allocated: {self._total_allocated_tokens:,}, "
                    f"Reserved: {self.reserved_tokens:,}, "
                    f"Remaining: {self.remaining_tokens:,}"
                ),
                requested=tokens,
                available=self.remaining_tokens,
                parent_id=self.parent_contract.id,
            )

        # Check conservation law for cost
        if cost_usd > 0 and cost_usd > self.remaining_cost:
            raise ConservationViolationError(
                message=(
                    f"Cannot allocate ${cost_usd:.4f} to '{name}'. "
                    f"Parent budget: ${self.parent_budget_cost:.4f}, "
                    f"Parent used: ${self.parent_used_cost:.4f}, "
                    f"Already allocated: ${self._total_allocated_cost:.4f}, "
                    f"Reserved: ${self.reserved_cost:.4f}, "
                    f"Remaining: ${self.remaining_cost:.4f}"
                ),
                requested=int(cost_usd * 10000),  # Convert to basis points for int
                available=int(self.remaining_cost * 10000),
                parent_id=self.parent_contract.id,
            )

        # Check conservation law for per-tool limits.
        #
        # Tools the parent does not constrain (`remaining_per_tool == None`)
        # are treated as unbounded — child requests for them are always
        # allowed. This matches the monitor's existing semantics at
        # monitor.py:567 ("missing entry = no limit") so the framework
        # behaves consistently across delegation and enforcement layers.
        # Sort for deterministic error messages — handy when multiple
        # tools could be at fault and the test wants a stable assertion.
        if per_tool_limits:
            for tool, requested in sorted(per_tool_limits.items()):
                if requested <= 0:
                    continue
                remaining_for_tool = self.remaining_per_tool(tool)
                if remaining_for_tool is None:
                    continue
                if requested > remaining_for_tool:
                    raise ConservationViolationError(
                        message=(
                            f"Cannot allocate {requested:,} '{tool}' calls to '{name}'. "
                            f"Parent limit: {self.parent_per_tool_limit(tool):,}, "
                            f"Parent used: {self.parent_per_tool_used(tool):,}, "
                            f"Already allocated: "
                            f"{self._total_allocated_per_tool.get(tool, 0):,}, "
                            f"Reserved: {self.reserved_per_tool(tool):,}, "
                            f"Remaining: {remaining_for_tool:,}"
                        ),
                        requested=requested,
                        available=remaining_for_tool,
                        parent_id=self.parent_contract.id,
                    )

        # Create child contract
        # Note: reasoning_tokens is stored in metadata (not ResourceConstraints)
        # because it's informational for prompts, not enforced. The actual thinking
        # budget is controlled by the model's ThinkingConfig, not token counting.
        child_resources = ResourceConstraints(
            tokens=tokens if tokens > 0 else None,
            cost_usd=cost_usd if cost_usd > 0 else None,
            api_calls=api_calls,
            iterations=iterations,
            tool_invocations=tool_invocations,
            per_tool_limits=per_tool_limits or {},
        )

        child_metadata = metadata or {}
        child_metadata["parent_id"] = self.parent_contract.id
        if reasoning_tokens is not None:
            child_metadata["reasoning_tokens"] = reasoning_tokens
        child_metadata["delegation_time"] = datetime.now().isoformat()

        child_contract = Contract(
            id=child_id,
            name=name,
            description=description,
            resources=child_resources,
            metadata=child_metadata,
        )

        # Record allocation
        per_tool_recorded: dict[str, int] = dict(per_tool_limits or {})
        allocation = AllocationRecord(
            child_id=child_id,
            child_name=name,
            tokens_allocated=tokens,
            cost_allocated=cost_usd,
            per_tool_limits_allocated=per_tool_recorded,
            child_contract=child_contract,
        )
        self._allocations[child_id] = allocation
        self._total_allocated_tokens += tokens
        self._total_allocated_cost += cost_usd
        # Bump per-tool running totals only for tools the parent constrains.
        # Recording every tool in `per_tool_limits_allocated` is good for
        # audit, but conservation accounting only matters for budgeted
        # tools — otherwise an unconstrained tool's totals would drift
        # without ever participating in `_check_conservation`.
        for tool, requested in per_tool_recorded.items():
            if requested > 0 and self.parent_per_tool_limit(tool) is not None:
                self._total_allocated_per_tool[tool] = (
                    self._total_allocated_per_tool.get(tool, 0) + requested
                )

        return child_contract

    def get_allocation(self, name: str) -> AllocationRecord | None:
        """Get allocation record for a child by name.

        Args:
            name: Name of the child contract

        Returns:
            AllocationRecord if found, None otherwise
        """
        child_id = f"{self.parent_contract.id}/{name}"
        return self._allocations.get(child_id)

    def get_child_contract(self, name: str) -> Contract | None:
        """Get child contract by name.

        Args:
            name: Name of the child contract

        Returns:
            Contract if found, None otherwise
        """
        allocation = self.get_allocation(name)
        return allocation.child_contract if allocation else None

    def release_allocation(self, name: str) -> int:
        """Release a child's allocation back to the pool.

        This is called when a child completes and returns unused budget.
        The actual tokens returned depend on what the child actually used.

        Args:
            name: Name of the child contract

        Returns:
            Number of tokens released back to pool

        Raises:
            KeyError: If child not found
        """
        child_id = f"{self.parent_contract.id}/{name}"
        if child_id not in self._allocations:
            raise KeyError(f"No allocation found for '{name}'")

        allocation = self._allocations.pop(child_id)
        self._total_allocated_tokens -= allocation.tokens_allocated
        self._total_allocated_cost -= allocation.cost_allocated
        # Release per-tool allocations symmetrically. Only the
        # parent-constrained tools were added to the running total in
        # `create_subcontract`, so we can iterate the recorded map and
        # safely subtract — non-constrained tools won't appear in
        # `_total_allocated_per_tool` so the .get(..., 0) guards them.
        for tool, requested in allocation.per_tool_limits_allocated.items():
            if requested > 0 and tool in self._total_allocated_per_tool:
                remaining_total = self._total_allocated_per_tool.get(tool, 0) - requested
                if remaining_total <= 0:
                    self._total_allocated_per_tool.pop(tool, None)
                else:
                    self._total_allocated_per_tool[tool] = remaining_total

        return allocation.tokens_allocated

    def get_summary(self) -> DelegationSummary:
        """Get a summary of all delegations.

        Returns:
            DelegationSummary with complete delegation state
        """
        return DelegationSummary(
            parent_id=self.parent_contract.id,
            parent_budget_tokens=self.parent_budget_tokens,
            parent_budget_cost=self.parent_budget_cost,
            parent_used_tokens=self.parent_used_tokens,
            parent_used_cost=self.parent_used_cost,
            total_allocated_tokens=self._total_allocated_tokens,
            total_allocated_cost=self._total_allocated_cost,
            remaining_tokens=self.remaining_tokens,
            remaining_cost=self.remaining_cost,
            allocations=list(self._allocations.values()),
            conservation_satisfied=self._check_conservation(),
            total_allocated_per_tool=dict(self._total_allocated_per_tool),
        )

    def _check_conservation(self) -> bool:
        """Verify conservation law is satisfied across all budgeted axes.

        Conservation: parent_used + Σ child_allocations ≤ parent_budget,
        checked independently for tokens, cost, and each per-tool limit
        the parent declares. A per-tool axis the parent doesn't declare
        is unconstrained and trivially satisfied.
        """
        tokens_ok = (
            self.parent_used_tokens + self._total_allocated_tokens <= self.parent_budget_tokens
        )
        cost_ok = (
            (self.parent_used_cost + self._total_allocated_cost <= self.parent_budget_cost)
            if self.parent_budget_cost > 0
            else True
        )

        per_tool_ok = True
        for tool, limit in self.parent_contract.resources.per_tool_limits.items():
            used = self.parent_per_tool_used(tool)
            allocated = self._total_allocated_per_tool.get(tool, 0)
            if used + allocated > limit:
                per_tool_ok = False
                break

        return tokens_ok and cost_ok and per_tool_ok

    def __repr__(self) -> str:
        """String representation of delegation state."""
        return (
            f"ContractingCapability("
            f"parent='{self.parent_contract.id}', "
            f"budget={self.parent_budget_tokens:,} tokens, "
            f"used={self.parent_used_tokens:,}, "
            f"allocated={self._total_allocated_tokens:,}, "
            f"remaining={self.remaining_tokens:,}, "
            f"children={len(self._allocations)})"
        )
