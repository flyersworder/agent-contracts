"""Core contract data structures.

This module implements the formal contract definition from the whitepaper (Section 2.1):
    C = (I, O, S, R, T, Φ, Ψ)

Where:
    I: Input specification - Schema and constraints for acceptable inputs
    O: Output specification - Schema and quality criteria for outputs
    S: Skills - Set of capabilities (tools, functions, knowledge domains)
    R: Resource constraints - Multi-dimensional resource budget
    T: Temporal constraints - Time-related boundaries and deadlines
    Φ: Success criteria - Measurable conditions for contract fulfillment
    Ψ: Termination conditions - Events that end the contract
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class ContractState(Enum):
    """Contract lifecycle states (Section 3.1 of whitepaper)."""

    DRAFTED = "drafted"  # Contract defined but not active
    ACTIVE = "active"  # Agent executing within constraints
    FULFILLED = "fulfilled"  # Success criteria met
    VIOLATED = "violated"  # Constraints breached
    EXPIRED = "expired"  # Time limit reached
    TERMINATED = "terminated"  # External cancellation


class DeadlineType(Enum):
    """Types of temporal deadlines (Section 2.3 of whitepaper)."""

    HARD = "hard"  # Task must complete by deadline
    SOFT = "soft"  # Task should complete by deadline, quality degrades after


@dataclass(frozen=True)
class ResourceConstraints:
    """Multi-dimensional resource budget (Section 2.2 of whitepaper).

    Supports both lumpsum and separate token budgets for reasoning models:
    - Lumpsum: Specify only `tokens` (model decides reasoning vs text split)
    - Separate: Specify `reasoning_tokens` and/or `text_tokens` for fine control

    Attributes:
        tokens: Maximum total LLM tokens (reasoning + text) (None = unlimited)
        reasoning_tokens: Maximum tokens for internal reasoning/thinking (None = unlimited)
        text_tokens: Maximum tokens for text output (None = unlimited)
        api_calls: Maximum API calls allowed (None = unlimited)
        web_searches: Maximum web searches allowed (None = unlimited)
        tool_invocations: Maximum tool uses allowed (None = unlimited)
        memory_mb: Maximum memory in MB (None = unlimited)
        compute_seconds: Maximum CPU seconds (None = unlimited)
        cost_usd: Maximum cost in USD (None = unlimited)
    """

    tokens: int | None = None
    reasoning_tokens: int | None = None
    text_tokens: int | None = None
    api_calls: int | None = None
    web_searches: int | None = None
    tool_invocations: int | None = None
    memory_mb: float | None = None
    compute_seconds: float | None = None
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        """Validate resource constraints are non-negative."""
        for field_name, value in self.__dict__.items():
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")


@dataclass(frozen=True)
class TemporalConstraints:
    """Time-related boundaries and deadlines (Section 2.3 of whitepaper).

    Attributes:
        deadline: Absolute deadline (wall-clock time)
        max_duration: Maximum elapsed time (timedelta)
        deadline_type: Hard or soft deadline
        soft_deadline_quality_decay: Quality decay rate after soft deadline (λ)
        contract_expiration: When the contract lifecycle terminates
    """

    deadline: datetime | None = None
    max_duration: timedelta | None = None
    deadline_type: DeadlineType = DeadlineType.HARD
    soft_deadline_quality_decay: float = 0.1
    contract_expiration: datetime | None = None

    def __post_init__(self) -> None:
        """Validate temporal constraints."""
        if self.soft_deadline_quality_decay < 0:
            raise ValueError(
                f"soft_deadline_quality_decay must be non-negative, "
                f"got {self.soft_deadline_quality_decay}"
            )


@dataclass
class InputSpecification:
    """Schema and constraints for acceptable inputs (I in whitepaper).

    Attributes:
        schema: JSON schema or type definition for inputs
        constraints: Additional validation constraints
        examples: Example valid inputs
    """

    schema: dict[str, Any] | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    examples: list[Any] = field(default_factory=list)


@dataclass
class OutputSpecification:
    """Schema and quality criteria for outputs (O in whitepaper).

    Attributes:
        schema: JSON schema or type definition for outputs
        quality_criteria: Measurable quality requirements
        min_quality: Minimum acceptable quality score (0-1)
    """

    schema: dict[str, Any] | None = None
    quality_criteria: dict[str, Any] = field(default_factory=dict)
    min_quality: float = 0.0

    def __post_init__(self) -> None:
        """Validate output specification."""
        if not 0 <= self.min_quality <= 1:
            raise ValueError(f"min_quality must be in [0, 1], got {self.min_quality}")


@dataclass
class SuccessCriterion:
    """A single success criterion (element of Φ in whitepaper).

    Attributes:
        name: Human-readable name
        condition: Condition to evaluate (as string or callable)
        weight: Importance weight (0-1)
        required: Whether this criterion is mandatory
    """

    name: str
    condition: str | Any
    weight: float = 1.0
    required: bool = False

    def __post_init__(self) -> None:
        """Validate success criterion."""
        if not 0 <= self.weight <= 1:
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")


@dataclass
class TerminationCondition:
    """Events that end the contract (element of Ψ in whitepaper).

    Attributes:
        type: Type of termination condition
        condition: The specific condition to check
        priority: Priority for evaluation (higher = earlier)
    """

    type: str  # e.g., "time_limit", "resource_exhaustion", "task_completion"
    condition: str | Any
    priority: int = 0


@dataclass
class Contract:
    """Formal Agent Contract (C = (I, O, S, R, T, Φ, Ψ) from whitepaper Section 2.1).

    A contract defines the complete specification for an agent's execution,
    including inputs, outputs, skills, resource budgets, time constraints,
    success criteria, and termination conditions.

    Attributes:
        id: Unique identifier for this contract
        name: Human-readable name
        description: Contract purpose and context
        version: Contract version
        created_at: When the contract was created
        inputs: Input specification (I)
        outputs: Output specification (O)
        skills: Available capabilities (S)
        resources: Resource constraints (R)
        temporal: Temporal constraints (T)
        success_criteria: Success conditions (Φ)
        termination_conditions: Termination events (Ψ)
        state: Current contract state
        metadata: Additional contract metadata
    """

    # Required fields
    id: str
    name: str

    # Optional core fields
    description: str = ""
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)

    # Contract components (I, O, S, R, T, Φ, Ψ)
    inputs: InputSpecification = field(default_factory=InputSpecification)
    outputs: OutputSpecification = field(default_factory=OutputSpecification)
    skills: list[str] = field(default_factory=list)
    resources: ResourceConstraints = field(default_factory=ResourceConstraints)
    temporal: TemporalConstraints = field(default_factory=TemporalConstraints)
    success_criteria: list[SuccessCriterion] = field(default_factory=list)
    termination_conditions: list[TerminationCondition] = field(default_factory=list)

    # State tracking
    state: ContractState = ContractState.DRAFTED

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def activate(self) -> None:
        """Activate the contract, transitioning from DRAFTED to ACTIVE state."""
        if self.state != ContractState.DRAFTED:
            raise ValueError(f"Cannot activate contract in {self.state} state")
        object.__setattr__(self, "state", ContractState.ACTIVE)

    def fulfill(self) -> None:
        """Mark contract as fulfilled (success criteria met)."""
        if self.state != ContractState.ACTIVE:
            raise ValueError(f"Cannot fulfill contract in {self.state} state")
        object.__setattr__(self, "state", ContractState.FULFILLED)

    def violate(self, reason: str = "") -> None:
        """Mark contract as violated (constraints breached).

        Args:
            reason: Explanation of the violation
        """
        if self.state != ContractState.ACTIVE:
            raise ValueError(f"Cannot violate contract in {self.state} state")
        if reason:
            self.metadata["violation_reason"] = reason
        object.__setattr__(self, "state", ContractState.VIOLATED)

    def expire(self) -> None:
        """Mark contract as expired (time limit reached)."""
        if self.state != ContractState.ACTIVE:
            raise ValueError(f"Cannot expire contract in {self.state} state")
        object.__setattr__(self, "state", ContractState.EXPIRED)

    def terminate(self, reason: str = "") -> None:
        """Terminate contract externally.

        Args:
            reason: Explanation of the termination
        """
        if self.state not in {ContractState.DRAFTED, ContractState.ACTIVE}:
            raise ValueError(f"Cannot terminate contract in {self.state} state")
        if reason:
            self.metadata["termination_reason"] = reason
        object.__setattr__(self, "state", ContractState.TERMINATED)

    def is_active(self) -> bool:
        """Check if contract is currently active."""
        return self.state == ContractState.ACTIVE

    def is_complete(self) -> bool:
        """Check if contract has reached a terminal state."""
        return self.state in {
            ContractState.FULFILLED,
            ContractState.VIOLATED,
            ContractState.EXPIRED,
            ContractState.TERMINATED,
        }

    def __repr__(self) -> str:
        """String representation of contract."""
        return (
            f"Contract(id='{self.id}', name='{self.name}', "
            f"state={self.state.value}, version='{self.version}')"
        )
