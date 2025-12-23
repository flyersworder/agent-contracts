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

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_contracts.core.executor import ExecutionResult


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
        if self.reasoning_effort is not None:
            valid_efforts = {"low", "medium", "high"}
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
class AgentSpec:
    """Model-agnostic specification for an agent in multi-agent scenarios.

    Defines what a sub-agent can do, independent of which LLM runs it.
    The model choice is made at execution time via ExecutionConfig.

    Attributes:
        name: Unique identifier for this agent
        role: Description of agent's role in the workflow
        tools: Tools available to this agent
        skills: Skills this agent can use
        instructions: Instructions for this agent (model-agnostic)
    """

    name: str
    role: str = ""
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    instructions: str = ""


# Regex pattern for valid skill names (agentskills.io specification)
# Pattern: 1-64 chars, lowercase alphanumeric + hyphens, must start with letter,
# must end with alphanumeric (not hyphen)
_SKILL_NAME_PATTERN = re.compile(r"^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$")


@dataclass(frozen=True)
class SkillSpec:
    """Agent Skill specification following the agentskills.io open standard.

    Skills are reusable instruction packages that provide domain-specific expertise.
    They follow the progressive disclosure pattern:
    - Metadata (name, description) loads at startup for discovery
    - Full instructions load only when the skill is activated
    - References and scripts load on-demand as needed

    This implementation is compatible with the open standard adopted by:
    Microsoft (VS Code, GitHub), OpenAI, Cursor, Goose, Amp, and others.

    File Structure (when exported to filesystem):
        my-skill/
        ├── SKILL.md        # Required: frontmatter + instructions
        ├── reference.md    # Optional: detailed documentation
        ├── scripts/        # Optional: helper scripts
        └── assets/         # Optional: templates, images

    Attributes:
        name: Unique identifier (1-64 chars, lowercase alphanumeric + hyphens)
        description: What the skill does and when to use it (1-1024 chars)
        instructions: Main skill content (step-by-step guidance, examples)
        version: Semantic version string (default: "1.0.0")
        license: License name or reference (optional)
        allowed_tools: Tools the skill can use (optional, restricts access)
        references: Additional documentation files (name -> content)
        scripts: Helper scripts (name -> content or path)
        metadata: Custom key-value pairs for extensibility

    Example:
        >>> skill = SkillSpec(
        ...     name="code-reviewer",
        ...     description="Review code for best practices, security issues, "
        ...                 "and test coverage. Use when reviewing PRs or "
        ...                 "analyzing code quality.",
        ...     instructions='''
        ...     ## Instructions
        ...     1. Read the target files
        ...     2. Check for common issues:
        ...        - Error handling
        ...        - Security vulnerabilities
        ...        - Test coverage
        ...     3. Provide detailed feedback
        ...     ''',
        ...     allowed_tools=["Read", "Grep", "Glob"],
        ... )
        >>>
        >>> # Use in Capabilities
        >>> caps = Capabilities(
        ...     skills=[skill, "simple-skill-name"],  # Mix of SkillSpec and strings
        ...     tools=["web_search"],
        ... )
    """

    # Required fields (agentskills.io specification)
    name: str
    description: str

    # Main content
    instructions: str = ""

    # Optional metadata fields
    version: str = "1.0.0"
    license: str | None = None
    allowed_tools: list[str] | None = None

    # Progressive disclosure content (loaded on-demand)
    references: dict[str, str] = field(default_factory=dict)
    scripts: dict[str, str] = field(default_factory=dict)

    # Extensibility
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate skill specification against agentskills.io standard."""
        # Validate name format
        if not _SKILL_NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Skill name must be 1-64 lowercase alphanumeric characters with hyphens, "
                f"cannot start/end with hyphen or have consecutive hyphens. Got: '{self.name}'"
            )

        # Validate description length
        if len(self.description) == 0:
            raise ValueError("Skill description cannot be empty")
        if len(self.description) > 1024:
            raise ValueError(
                f"Skill description must be ≤1024 characters, got {len(self.description)}"
            )

    def to_frontmatter(self) -> str:
        """Generate YAML frontmatter for SKILL.md export.

        Returns:
            YAML frontmatter string (without --- delimiters)
        """
        lines = [
            f"name: {self.name}",
            f"description: {self.description}",
        ]

        if self.version != "1.0.0":
            lines.append(f"version: {self.version}")

        if self.license:
            lines.append(f"license: {self.license}")

        if self.allowed_tools:
            lines.append(f"allowed-tools: {' '.join(self.allowed_tools)}")

        if self.metadata:
            lines.append("metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def to_skill_md(self) -> str:
        """Generate complete SKILL.md content.

        Returns:
            Full SKILL.md file content with frontmatter and instructions
        """
        return f"---\n{self.to_frontmatter()}\n---\n\n{self.instructions}"

    @classmethod
    def from_skill_md(cls, content: str, name: str | None = None) -> "SkillSpec":
        """Parse SKILL.md content into SkillSpec.

        Args:
            content: Raw SKILL.md file content
            name: Override name (if not in frontmatter or for validation)

        Returns:
            Parsed SkillSpec instance

        Raises:
            ValueError: If content is invalid or missing required fields
        """
        # Split frontmatter and content
        if not content.startswith("---"):
            raise ValueError("SKILL.md must start with YAML frontmatter (---)")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError("SKILL.md must have closing --- for frontmatter")

        frontmatter_text = parts[1].strip()
        instructions = parts[2].strip()

        # Parse YAML frontmatter (simple parser for common cases)
        frontmatter: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        in_metadata = False

        for line in frontmatter_text.split("\n"):
            line = line.rstrip()
            if not line:
                continue

            if line.startswith("  ") and in_metadata:
                # Metadata value
                key_val = line.strip().split(":", 1)
                if len(key_val) == 2:
                    metadata[key_val[0].strip()] = key_val[1].strip()
            elif ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key == "metadata":
                    in_metadata = True
                else:
                    in_metadata = False
                    frontmatter[key] = value

        # Extract required fields
        skill_name = name or frontmatter.get("name")
        if not skill_name:
            raise ValueError("SKILL.md missing required 'name' field")

        description = frontmatter.get("description")
        if not description:
            raise ValueError("SKILL.md missing required 'description' field")

        # Extract optional fields
        allowed_tools = None
        if "allowed-tools" in frontmatter:
            allowed_tools = frontmatter["allowed-tools"].split()

        return cls(
            name=skill_name,
            description=description,
            instructions=instructions,
            version=frontmatter.get("version", "1.0.0"),
            license=frontmatter.get("license"),
            allowed_tools=allowed_tools,
            metadata=metadata if metadata else {},
        )

    @property
    def metadata_tokens(self) -> int:
        """Estimate tokens for metadata (name + description).

        Used for progressive disclosure budgeting.
        """
        # Rough estimate: ~4 chars per token
        return (len(self.name) + len(self.description)) // 4

    @property
    def instruction_tokens(self) -> int:
        """Estimate tokens for full instructions.

        Used for progressive disclosure budgeting.
        """
        return len(self.instructions) // 4


class CoordinationPattern(Enum):
    """Patterns for multi-agent coordination.

    Defines how multiple agents work together to complete a contract.
    """

    SEQUENTIAL = "sequential"  # Agents execute one after another
    PARALLEL = "parallel"  # Agents execute simultaneously
    HIERARCHICAL = "hierarchical"  # Parent delegates to children
    COLLABORATIVE = "collaborative"  # Agents negotiate and collaborate
    COMPETITIVE = "competitive"  # Agents compete, best result wins


@dataclass
class ExecutionConfig:
    """Model-specific execution configuration.

    This class contains LLM-specific settings that control how the contract
    is executed. These are separate from Capabilities because they are
    implementation details, not part of what the agent can do.

    Attributes:
        model: LLM model to use (e.g., "gpt-4o", "claude-sonnet-4-20250514")
        provider: LLM provider (auto-detected if not specified)
        temperature: LLM temperature setting (0-2)
        max_retries: Maximum retry attempts on failure
        timeout_seconds: Maximum execution time per LLM call

    Example:
        >>> config = ExecutionConfig(
        ...     model="gpt-4o",
        ...     temperature=0.7,
        ...     max_retries=3
        ... )
    """

    model: str = "gpt-4o"
    provider: str | None = None
    temperature: float = 0.7
    max_retries: int = 3
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate execution configuration."""
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")

        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")

        if self.provider is None:
            self.provider = self._detect_provider()

    def _detect_provider(self) -> str:
        """Auto-detect LLM provider from model name."""
        model_lower = self.model.lower()

        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        elif "llama" in model_lower:
            return "meta"
        elif "deepseek" in model_lower:
            return "deepseek"
        else:
            return "unknown"


@dataclass
class Capabilities:
    """Model-agnostic agent capabilities specification (extended S in whitepaper).

    This class defines WHAT an agent can access and do, independent of which
    LLM executes it. Capabilities are portable across different models because
    they use standardized interfaces:

    - **MCP (Model Context Protocol)**: Standard for exposing tools/resources
    - **Agent Skills**: Reusable, composable agent behaviors (agentskills.io standard)
    - **Tools**: Function-calling interfaces (standardized across LLMs)

    The separation from ExecutionConfig means you can:
    1. Define capabilities once
    2. Execute with different models (GPT-4, Claude, Gemini, etc.)
    3. Share capability definitions across teams/projects

    Skills can be specified as:
    - Simple strings: `["code-review", "research"]` for basic skill references
    - Rich SkillSpec objects: Full skill definitions with instructions

    Attributes:
        tools: External tools available (function names, MCP tool URIs, etc.)
        mcp_servers: MCP server URIs for extended capabilities
        skills: Agent skills - strings for simple references, SkillSpec for rich definitions
        resources: Named resources the agent can access (files, APIs, databases)
        agents: Sub-agents for multi-agent scenarios
        coordination: Pattern for multi-agent coordination
        instructions: Base instructions for the agent (model-agnostic)

    Example:
        >>> # Simple skills (backward compatible)
        >>> caps = Capabilities(
        ...     tools=["web_search", "calculator"],
        ...     skills=["code-review", "research"],
        ... )
        >>>
        >>> # Rich skill definitions (agentskills.io standard)
        >>> code_review = SkillSpec(
        ...     name="code-reviewer",
        ...     description="Review code for best practices and security",
        ...     instructions="1. Read the code\\n2. Check for issues\\n3. Report findings",
        ...     allowed_tools=["Read", "Grep"],
        ... )
        >>> caps = Capabilities(
        ...     skills=[code_review, "simple-skill"],  # Mix strings and SkillSpec
        ... )
        >>>
        >>> # Lookup skills by name
        >>> skill = caps.get_skill("code-reviewer")
        >>> if skill:
        ...     print(skill.instructions)
        >>>
        >>> # Get all skill names (for discovery)
        >>> print(caps.skill_names)  # ["code-reviewer", "simple-skill"]
        >>>
        >>> # Multi-agent workflow
        >>> caps = Capabilities(
        ...     agents={
        ...         "researcher": AgentSpec(
        ...             name="researcher",
        ...             role="Gathers information from sources",
        ...             tools=["web_search", "document_reader"]
        ...         ),
        ...         "analyst": AgentSpec(
        ...             name="analyst",
        ...             role="Analyzes and synthesizes findings"
        ...         ),
        ...     },
        ...     coordination=CoordinationPattern.SEQUENTIAL
        ... )
    """

    tools: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    skills: list[str | SkillSpec] = field(default_factory=list)
    resources: dict[str, str] = field(default_factory=dict)  # name -> URI/path
    agents: dict[str, AgentSpec] | None = None
    coordination: CoordinationPattern | None = None
    instructions: str | None = None

    def __post_init__(self) -> None:
        """Validate capabilities configuration."""
        # Auto-set coordination pattern for multi-agent configs
        if self.agents and not self.coordination:
            object.__setattr__(self, "coordination", CoordinationPattern.SEQUENTIAL)

    @property
    def is_multi_agent(self) -> bool:
        """Check if this is a multi-agent configuration."""
        return self.agents is not None and len(self.agents) > 0

    @property
    def agent_count(self) -> int:
        """Get the number of agents (1 for single-agent, N for multi-agent)."""
        if self.agents:
            return len(self.agents)
        return 1

    # --- Skill helper methods ---

    @property
    def skill_names(self) -> list[str]:
        """Get all skill names for discovery/display.

        Returns:
            List of skill names (strings), extracting names from SkillSpec objects.
        """
        return [skill.name if isinstance(skill, SkillSpec) else skill for skill in self.skills]

    @property
    def skill_specs(self) -> list[SkillSpec]:
        """Get only the SkillSpec objects (not string references).

        Returns:
            List of SkillSpec objects with full skill definitions.
        """
        return [skill for skill in self.skills if isinstance(skill, SkillSpec)]

    def get_skill(self, name: str) -> SkillSpec | None:
        """Look up a skill by name.

        Args:
            name: The skill name to look up

        Returns:
            SkillSpec if found and is a rich definition, None otherwise.
            Note: Returns None for string-only skill references.
        """
        for skill in self.skills:
            if isinstance(skill, SkillSpec) and skill.name == name:
                return skill
        return None

    def has_skill(self, name: str) -> bool:
        """Check if a skill exists (by name or SkillSpec).

        Args:
            name: The skill name to check

        Returns:
            True if skill exists (as string or SkillSpec), False otherwise.
        """
        for skill in self.skills:
            if isinstance(skill, SkillSpec):
                if skill.name == name:
                    return True
            elif skill == name:
                return True
        return False

    @property
    def total_metadata_tokens(self) -> int:
        """Estimate total tokens for skill metadata (for progressive disclosure).

        This is the cost of loading skill names and descriptions for discovery,
        before any skill is activated.
        """
        total = 0
        for skill in self.skills:
            if isinstance(skill, SkillSpec):
                total += skill.metadata_tokens
            else:
                # String skills: just the name (~2-3 tokens)
                total += len(skill) // 4 + 1
        return total

    @property
    def total_instruction_tokens(self) -> int:
        """Estimate total tokens for all skill instructions.

        This would be the cost of loading ALL skills at once.
        Progressive disclosure loads instructions only when activated.
        """
        return sum(
            skill.instruction_tokens for skill in self.skills if isinstance(skill, SkillSpec)
        )


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
