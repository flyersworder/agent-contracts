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
        created_at: When the allocation was made
        child_contract: Reference to the child contract
    """

    child_id: str
    child_name: str
    tokens_allocated: int = 0
    cost_allocated: float = 0.0
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
        conservation_satisfied: Whether conservation law holds
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

    def can_allocate(self, tokens: int = 0, cost_usd: float = 0.0) -> bool:
        """Check if an allocation is possible without violating conservation.

        Args:
            tokens: Number of tokens to allocate
            cost_usd: Cost budget to allocate

        Returns:
            True if allocation would satisfy conservation law
        """
        return tokens <= self.remaining_tokens and cost_usd <= self.remaining_cost

    def create_subcontract(
        self,
        name: str,
        tokens: int = 0,
        cost_usd: float = 0.0,
        api_calls: int | None = None,
        iterations: int | None = None,
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

        # Create child contract
        child_resources = ResourceConstraints(
            tokens=tokens if tokens > 0 else None,
            cost_usd=cost_usd if cost_usd > 0 else None,
            api_calls=api_calls,
            iterations=iterations,
        )

        child_metadata = metadata or {}
        child_metadata["parent_id"] = self.parent_contract.id
        child_metadata["delegation_time"] = datetime.now().isoformat()

        child_contract = Contract(
            id=child_id,
            name=name,
            description=description,
            resources=child_resources,
            metadata=child_metadata,
        )

        # Record allocation
        allocation = AllocationRecord(
            child_id=child_id,
            child_name=name,
            tokens_allocated=tokens,
            cost_allocated=cost_usd,
            child_contract=child_contract,
        )
        self._allocations[child_id] = allocation
        self._total_allocated_tokens += tokens
        self._total_allocated_cost += cost_usd

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
        )

    def _check_conservation(self) -> bool:
        """Verify conservation law is satisfied.

        Conservation: used + allocated ≤ budget
        """
        tokens_ok = (
            self.parent_used_tokens + self._total_allocated_tokens <= self.parent_budget_tokens
        )
        cost_ok = (
            (self.parent_used_cost + self._total_allocated_cost <= self.parent_budget_cost)
            if self.parent_budget_cost > 0
            else True
        )

        return tokens_ok and cost_ok

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
