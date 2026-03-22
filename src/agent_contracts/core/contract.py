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

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from agent_contracts.core.capabilities import (  # noqa: F401
    AgentSpec,
    Capabilities,
    CoordinationPattern,
    ExecutionConfig,
)
from agent_contracts.core.skillspec import SkillSpec

if TYPE_CHECKING:
    from agent_contracts.core.executor import ExecutionResult

logger = logging.getLogger(__name__)


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


class ContractMode(Enum):
    """Strategic contract modes for quality-cost-time tradeoffs (Section 2.4 of whitepaper).

    These modes define the optimization strategy for contract execution:

    - URGENT: Optimize for speed, relax resource constraints
      * Minimize time to completion
      * Allow higher resource usage
      * Target: ≥85% quality in ≤50% time (vs balanced)
      * Use case: Time-critical tasks, urgent deadlines

    - BALANCED: Equal priority on quality, cost, and time
      * Default mode for standard execution
      * Balanced resource utilization
      * Target: Optimal quality-cost-time balance
      * Use case: Standard production workloads

    - ECONOMICAL: Minimize costs, allow more time
      * Minimize resource consumption
      * Accept longer execution time
      * Target: ≥90% quality at ≤40% tokens (vs balanced)
      * Use case: Batch processing, cost-sensitive tasks

    The mode affects:
    - Budget allocation strategy
    - System prompts (budget-aware prompting)
    - Resource consumption patterns
    - Strategic decision-making during execution
    """

    URGENT = "urgent"
    BALANCED = "balanced"
    ECONOMICAL = "economical"


@dataclass(frozen=True)
class ResourceConstraints:
    """Multi-dimensional resource budget (Section 2.2 of whitepaper).

    Token Budget Modes:

    1. Lumpsum Mode (default, backward compatible):
       ResourceConstraints(tokens=10000)
       - Single budget for reasoning + text output combined
       - Model decides the split between thinking and output

    2. Fine-Grained Mode (explicit control):
       ResourceConstraints(reasoning_tokens=5000, text_tokens=2000)
       - Separate budgets for reasoning vs text output
       - More control over model behavior (limit verbosity, allow deep thinking, etc.)

    3. Partial Fine-Grained Mode:
       ResourceConstraints(text_tokens=1000)  # Limit output, unlimited reasoning
       ResourceConstraints(reasoning_tokens=5000)  # Limit thinking, unlimited output

    Note: Cannot mix modes - specify EITHER tokens OR (reasoning_tokens/text_tokens).

    Reasoning Effort Control:

    For reasoning models (Gemini 2.5, o1, Claude extended thinking), you can specify
    how deeply the model should think:

    - reasoning_effort="low": Quick, shallow thinking (~100-500 tokens)
    - reasoning_effort="medium": Balanced thinking (~500-2000 tokens)
    - reasoning_effort="high": Deep, thorough thinking (≥2000 tokens)

    If reasoning_effort is specified, it must be compatible with reasoning_tokens budget.
    If not specified, effort is auto-selected based on budget.

    Per-Tool Limits:

    You can set individual limits for specific tools while keeping an aggregate limit:

        ResourceConstraints(
            tool_invocations=20,  # Total limit across all tools
            per_tool_limits={
                "web_search": 5,   # Max 5 web searches via this tool
                "code_exec": 3,    # Max 3 code executions
                # Tools not in dict have no individual limit (only aggregate applies)
            }
        )

    Enforcement priority:
    1. Per-tool limit checked first (e.g., web_search ≤ 5)
    2. Then aggregate limit checked (e.g., total tool calls ≤ 20)

    Attributes:
        tokens: Maximum total LLM tokens (reasoning + text) (None = unlimited)
        reasoning_tokens: Maximum tokens for internal reasoning/thinking (None = unlimited)
        text_tokens: Maximum tokens for text output (None = unlimited)
        reasoning_effort: How deeply model should think ("low"/"medium"/"high") (None = auto)
        api_calls: Maximum API calls allowed (None = unlimited)
        iterations: Maximum agent loop iterations (None = unlimited).
            For Google ADK, this maps to RunConfig.max_llm_calls.
            For LangGraph, this maps to recursion_limit.
        web_searches: Maximum web searches allowed (None = unlimited)
        tool_invocations: Maximum tool uses allowed (None = unlimited)
        per_tool_limits: Per-tool invocation limits (tool_name -> max_calls)
        memory_mb: Maximum memory in MB (None = unlimited)
        compute_seconds: Maximum CPU seconds (None = unlimited)
        cost_usd: Maximum cost in USD (None = unlimited)
    """

    tokens: int | None = None
    reasoning_tokens: int | None = None
    text_tokens: int | None = None
    reasoning_effort: str | None = None
    api_calls: int | None = None
    iterations: int | None = None
    web_searches: int | None = None
    tool_invocations: int | None = None
    per_tool_limits: dict[str, int] = field(default_factory=dict)
    memory_mb: float | None = None
    compute_seconds: float | None = None
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        """Validate resource constraints are non-negative and mode consistency."""
        # Validate non-negative values (skip string fields like reasoning_effort and dicts)
        for field_name, value in self.__dict__.items():
            if value is not None and isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")

        # Validate per_tool_limits values are non-negative integers
        for tool_name, limit in self.per_tool_limits.items():
            if not isinstance(limit, int):
                raise ValueError(
                    f"per_tool_limits['{tool_name}'] must be an integer, got {type(limit).__name__}"
                )
            if limit < 0:
                raise ValueError(
                    f"per_tool_limits['{tool_name}'] must be non-negative, got {limit}"
                )

        # Validate token mode consistency - cannot mix lumpsum with fine-grained
        has_lumpsum = self.tokens is not None
        has_fine_grained = self.reasoning_tokens is not None or self.text_tokens is not None

        if has_lumpsum and has_fine_grained:
            raise ValueError(
                "Cannot mix token budget modes. Use EITHER:\n"
                "  - Lumpsum mode: tokens=10000\n"
                "  - Fine-grained mode: reasoning_tokens=5000, text_tokens=2000\n"
                "  - Partial fine-grained: text_tokens=1000 (or reasoning_tokens=5000)"
            )

        # Validate reasoning_effort values
        # "none" disables thinking entirely (supported by Gemini 2.5 Flash)
        if self.reasoning_effort is not None:
            valid_efforts = {"none", "low", "medium", "high"}
            if self.reasoning_effort not in valid_efforts:
                raise ValueError(
                    f"reasoning_effort must be one of {valid_efforts}, "
                    f"got '{self.reasoning_effort}'"
                )

        # Validate reasoning_effort + reasoning_tokens compatibility
        if self.reasoning_effort and self.reasoning_tokens:
            self._validate_reasoning_compatibility()

    def _validate_reasoning_compatibility(self) -> None:
        """Ensure reasoning_effort aligns with reasoning_tokens budget.

        Raises:
            ValueError: If effort level is incompatible with token budget
        """
        effort = self.reasoning_effort
        budget = self.reasoning_tokens

        # Define minimum token requirements for each effort level
        # These are empirical guidelines based on reasoning model behavior
        if effort == "high" and budget is not None and budget < 2000:
            raise ValueError(
                f"reasoning_effort='high' typically requires ≥2000 reasoning tokens, "
                f"but budget is only {budget}. Consider:\n"
                f"  - Increase reasoning_tokens to ≥2000\n"
                f"  - Reduce reasoning_effort to 'medium' or 'low'\n"
                f"  - Remove reasoning_effort (auto-select based on budget)"
            )
        elif effort == "medium" and budget is not None and budget < 500:
            raise ValueError(
                f"reasoning_effort='medium' typically requires ≥500 reasoning tokens, "
                f"but budget is only {budget}. Consider:\n"
                f"  - Increase reasoning_tokens to ≥500\n"
                f"  - Reduce reasoning_effort to 'low'\n"
                f"  - Remove reasoning_effort (auto-select based on budget)"
            )
        # "low" effort can work with any budget (even <500 tokens)

    @property
    def token_mode(self) -> str:
        """Determine active token budget mode.

        Returns:
            "lumpsum" if using combined tokens budget
            "fine_grained" if using separate reasoning/text budgets
            "none" if no token constraints specified
        """
        if self.tokens is not None:
            return "lumpsum"
        elif self.reasoning_tokens is not None or self.text_tokens is not None:
            return "fine_grained"
        else:
            return "none"

    @property
    def recommended_reasoning_effort(self) -> str | None:
        """Get recommended reasoning effort based on budget.

        Auto-selects effort level based on reasoning_tokens budget:
        - ≥2000 tokens → "high" effort (deep thinking)
        - ≥500 tokens → "medium" effort (balanced)
        - <500 tokens → "low" effort (quick thinking)

        Returns:
            Recommended effort level, or None if no reasoning_tokens specified
        """
        if self.reasoning_tokens is None:
            return None

        # Auto-select effort based on budget
        if self.reasoning_tokens >= 2000:
            return "high"
        elif self.reasoning_tokens >= 500:
            return "medium"
        else:
            return "low"


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

    This class supports structured output enforcement through LiteLLM's
    response_format parameter. You can specify either:
    - A JSON Schema dict (schema parameter)
    - A Pydantic BaseModel class (pydantic_model parameter)

    The to_response_format() method converts to LiteLLM's format.

    Attributes:
        schema: JSON schema dict for output structure (use for raw JSON Schema)
        pydantic_model: Pydantic BaseModel class for structured output
        strict: If True, enforce strict schema adherence (default: True)
        quality_criteria: Measurable quality requirements (e.g., {"accuracy": 0.9})
        min_quality: Minimum acceptable quality score (0-1)
        name: Optional name for the schema (used in response_format)

    Example:
        # Using JSON Schema:
        output = OutputSpecification(
            schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["answer", "confidence"]
            },
            name="answer_response"
        )

        # Using Pydantic (preferred):
        from pydantic import BaseModel

        class Answer(BaseModel):
            answer: str
            confidence: float

        output = OutputSpecification(pydantic_model=Answer)
    """

    schema: dict[str, Any] | None = None
    pydantic_model: type | None = None  # Pydantic BaseModel class
    strict: bool = True
    quality_criteria: dict[str, Any] = field(default_factory=dict)
    min_quality: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate output specification."""
        if not 0 <= self.min_quality <= 1:
            raise ValueError(f"min_quality must be in [0, 1], got {self.min_quality}")

        # Validate that at most one of schema/pydantic_model is provided
        if self.schema is not None and self.pydantic_model is not None:
            raise ValueError(
                "Cannot specify both 'schema' and 'pydantic_model'. Use one or the other."
            )

        # Validate pydantic_model is actually a Pydantic model
        if self.pydantic_model is not None:
            try:
                from pydantic import BaseModel

                if not (
                    isinstance(self.pydantic_model, type)
                    and issubclass(self.pydantic_model, BaseModel)
                ):
                    raise ValueError(
                        f"pydantic_model must be a Pydantic BaseModel class, "
                        f"got {type(self.pydantic_model)}"
                    )
            except ImportError as err:
                raise ValueError(
                    "pydantic_model requires the 'pydantic' package. "
                    "Install with: pip install pydantic"
                ) from err

    def has_structured_output(self) -> bool:
        """Check if structured output is configured.

        Returns:
            True if schema or pydantic_model is specified
        """
        return self.schema is not None or self.pydantic_model is not None

    def to_response_format(self) -> dict[str, Any] | type | None:
        """Convert to LiteLLM response_format parameter.

        This method returns the appropriate format for use with LiteLLM's
        completion() function. LiteLLM supports:
        - Pydantic BaseModel class (preferred - auto-converts to JSON Schema)
        - Raw JSON Schema dict with type="json_schema"

        Returns:
            response_format value for LiteLLM, or None if no schema specified

        Example:
            contract = Contract(
                id="my-contract",
                output=OutputSpecification(pydantic_model=MyModel)
            )
            response = litellm.completion(
                model="gpt-4o",
                messages=[...],
                response_format=contract.output.to_response_format()
            )
        """
        # Pydantic model - LiteLLM handles conversion automatically
        if self.pydantic_model is not None:
            return self.pydantic_model

        # JSON Schema - wrap in LiteLLM's expected format
        if self.schema is not None:
            schema_name = self.name or "response"
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": self.strict,
                    "schema": self.schema,
                },
            }

        return None

    def validate_output(self, output: str | dict[str, Any]) -> tuple[bool, str | None]:
        """Validate output against the schema.

        Args:
            output: The output to validate (JSON string or dict)

        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is None
        """
        import json

        # No schema to validate against
        if not self.has_structured_output():
            return True, None

        # Parse output if string
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {e}"

        # Validate with Pydantic model
        if self.pydantic_model is not None:
            try:
                # model_validate is available on all Pydantic BaseModel classes
                # We already validated it's a BaseModel subclass in __post_init__
                validate_fn = getattr(self.pydantic_model, "model_validate", None)
                if validate_fn is not None:
                    validate_fn(output)
                    return True, None
                return True, None  # Skip if method not found
            except Exception as e:
                return False, f"Pydantic validation failed: {e}"

        # Validate with JSON Schema
        if self.schema is not None:
            try:
                import jsonschema

                jsonschema.validate(output, self.schema)
                return True, None
            except ImportError:
                # jsonschema not installed - skip validation
                return True, None
            except Exception as e:
                # Catch all jsonschema errors (ValidationError, etc.)
                return False, f"JSON Schema validation failed: {e}"

        return True, None


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


# AgentSpec, CoordinationPattern, ExecutionConfig, and Capabilities are imported from
# agent_contracts.core.capabilities and re-exported for backward compatibility.


@dataclass
class Contract:
    """Formal Agent Contract (C = (I, O, S, R, T, Φ, Ψ) from whitepaper Section 2.1).

    A contract defines the complete specification for an agent's execution,
    including inputs, outputs, capabilities, resource budgets, time constraints,
    success criteria, and termination conditions.

    The Contract separates:
    - **Capabilities**: WHAT the agent can do (model-agnostic, portable)
    - **Execution**: HOW to run it (model-specific configuration)

    This enables a clean API where capabilities are reusable:

        # Define capabilities once (portable across models)
        research_caps = Capabilities(
            tools=["web_search", "document_reader"],
            skills=["research", "summarize"],
            instructions="You are a research assistant."
        )

        # Execute with any model
        contract = Contract(
            id="research",
            name="Research Task",
            resources=ResourceConstraints(tokens=10000, cost_usd=0.50),
            capabilities=research_caps,
            execution=ExecutionConfig(model="gpt-4o")  # or claude, gemini...
        )
        result = contract.execute(query="What is quantum computing?")

    Attributes:
        id: Unique identifier for this contract
        name: Human-readable name
        description: Contract purpose and context
        version: Contract version
        created_at: When the contract was created
        inputs: Input specification (I)
        outputs: Output specification (O)
        skills: Available capabilities as simple list (S) - deprecated, use capabilities
        capabilities: Model-agnostic capability specification (extended S)
        execution: Model-specific execution configuration
        resources: Resource constraints (R)
        temporal: Temporal constraints (T)
        success_criteria: Success conditions (Φ)
        termination_conditions: Termination events (Ψ)
        mode: Strategic mode for quality-cost-time tradeoffs
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
    skills: list[str | SkillSpec] = field(default_factory=list)  # Use capabilities
    capabilities: Capabilities | None = None  # Model-agnostic capability spec
    execution: ExecutionConfig | None = None  # Model-specific execution config
    resources: ResourceConstraints = field(default_factory=ResourceConstraints)
    temporal: TemporalConstraints = field(default_factory=TemporalConstraints)
    success_criteria: list[SuccessCriterion] = field(default_factory=list)
    termination_conditions: list[TerminationCondition] = field(default_factory=list)

    # Strategic mode (Section 2.4 - Time-Resource Tradeoff Surface)
    mode: ContractMode = ContractMode.BALANCED

    # State tracking
    state: ContractState = ContractState.DRAFTED

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize contract."""
        # If skills provided but no capabilities, create Capabilities with skills
        if self.skills and self.capabilities is None:
            object.__setattr__(self, "capabilities", Capabilities(skills=self.skills))
        # If both provided, merge skills into capabilities
        elif self.skills and self.capabilities is not None:
            # Merge skills by name, avoiding duplicates
            # SkillSpec objects take precedence over string references
            merged_skills = self._merge_skills(self.capabilities.skills, self.skills)
            # Create new Capabilities with merged skills
            new_caps = Capabilities(
                tools=self.capabilities.tools,
                mcp_servers=self.capabilities.mcp_servers,
                skills=merged_skills,
                resources=self.capabilities.resources,
                agents=self.capabilities.agents,
                coordination=self.capabilities.coordination,
                instructions=self.capabilities.instructions,
            )
            object.__setattr__(self, "capabilities", new_caps)

    @staticmethod
    def _merge_skills(
        skills1: list[str | SkillSpec], skills2: list[str | SkillSpec]
    ) -> list[str | SkillSpec]:
        """Merge two skill lists, avoiding duplicates by name.

        SkillSpec objects take precedence over string references.
        If both lists contain the same skill name, the SkillSpec version is kept.

        Args:
            skills1: First skill list (typically from capabilities)
            skills2: Second skill list (typically from contract.skills)

        Returns:
            Merged list with unique skills by name
        """
        result: dict[str, str | SkillSpec] = {}

        # Process all skills, SkillSpec takes precedence
        for skill in skills1 + skills2:
            if isinstance(skill, SkillSpec):
                # SkillSpec always replaces string reference
                result[skill.name] = skill
            else:
                # String only added if no SkillSpec exists for that name
                if skill not in result:
                    result[skill] = skill

        return list(result.values())

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

    def execute(self, **kwargs: Any) -> "ExecutionResult":
        """Execute the contract and return the result.

        This is the primary entry point for contract execution. It creates
        an executor, runs the contract with the provided inputs, and returns
        the result including any resource usage and violations.

        The executor automatically:
        - Selects the appropriate backend based on capabilities
        - Generates budget-aware prompts (integrates prompts.py)
        - Plans resource allocation (integrates planning.py)
        - Enforces resource and temporal constraints
        - Tracks execution metrics

        Args:
            **kwargs: Input arguments for the contract execution.
                Common inputs include:
                - query: Text query or prompt
                - messages: List of message dicts (for chat)
                - context: Additional context for the agent

        Returns:
            ExecutionResult containing:
                - success: Whether execution completed successfully
                - output: The agent's response/output
                - resource_usage: Tokens, cost, API calls consumed
                - violations: Any constraint violations
                - execution_log: Detailed execution trace

        Raises:
            ValueError: If contract has no capabilities defined
            RuntimeError: If execution fails (in strict mode)

        Example:
            >>> contract = Contract(
            ...     id="qa",
            ...     name="Q&A",
            ...     resources=ResourceConstraints(tokens=1000, cost_usd=0.01),
            ...     capabilities=Capabilities(model="gpt-4o")
            ... )
            >>> result = contract.execute(query="What is 2+2?")
            >>> print(result.output)
            "2 + 2 = 4"
            >>> print(result.resource_usage)
            {"tokens": 45, "cost_usd": 0.0001}
        """
        # Import here to avoid circular imports
        from agent_contracts.core.executor import ContractExecutor

        if self.capabilities is None:
            raise ValueError(
                "Contract must have capabilities defined to execute. "
                "Set capabilities=Capabilities(model='gpt-4o') or similar."
            )

        executor = ContractExecutor(self)
        return executor.run(**kwargs)
