"""Resource monitoring and tracking.

This module implements the runtime resource monitoring system that tracks actual
resource consumption and validates it against contract constraints.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from agent_contracts.core.contract import ResourceConstraints


@dataclass
class ResourceUsage:
    """Tracks actual resource consumption during agent execution.

    This class accumulates resource usage across multiple dimensions and provides
    methods to check if usage exceeds contract constraints.

    Supports separate tracking of reasoning vs text tokens for reasoning models.

    Attributes:
        tokens: Total tokens consumed (reasoning + text)
        reasoning_tokens: Tokens used for internal reasoning/thinking
        text_tokens: Tokens used for text output
        api_calls: Total API calls made
        web_searches: Total web searches performed
        tool_invocations: Total tool invocations
        memory_mb: Peak memory usage in MB
        compute_seconds: Total CPU time in seconds
        cost_usd: Total cost in USD
        start_time: When tracking started
        last_updated: When usage was last updated
        metadata: Additional usage metadata
    """

    tokens: int = 0
    reasoning_tokens: int = 0
    text_tokens: int = 0
    api_calls: int = 0
    web_searches: int = 0
    tool_invocations: int = 0
    memory_mb: float = 0.0
    compute_seconds: float = 0.0
    cost_usd: float = 0.0

    start_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate resource usage values are non-negative."""
        for field_name in [
            "tokens",
            "reasoning_tokens",
            "text_tokens",
            "api_calls",
            "web_searches",
            "tool_invocations",
            "memory_mb",
            "compute_seconds",
            "cost_usd",
        ]:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")

    def add_tokens(self, count: int, reasoning: int = 0, text: int = 0) -> None:
        """Add token usage with optional reasoning/text breakdown.

        Args:
            count: Total number of tokens to add (if reasoning and text not specified)
            reasoning: Number of reasoning tokens (for reasoning models)
            text: Number of text output tokens (for reasoning models)

        Raises:
            ValueError: If count is negative
        """
        if count < 0:
            raise ValueError(f"Token count must be non-negative, got {count}")
        if reasoning < 0:
            raise ValueError(f"Reasoning tokens must be non-negative, got {reasoning}")
        if text < 0:
            raise ValueError(f"Text tokens must be non-negative, got {text}")

        # If reasoning/text specified, use those; otherwise use total count
        if reasoning > 0 or text > 0:
            self.reasoning_tokens += reasoning
            self.text_tokens += text
            self.tokens += reasoning + text
        else:
            self.tokens += count

        self.last_updated = datetime.now()

    def add_api_call(self, cost: float = 0.0, tokens: int = 0) -> None:
        """Record an API call with optional cost and token information.

        Args:
            cost: Cost of this API call in USD
            tokens: Number of tokens consumed by this call

        Raises:
            ValueError: If cost or tokens are negative
        """
        if cost < 0:
            raise ValueError(f"Cost must be non-negative, got {cost}")
        if tokens < 0:
            raise ValueError(f"Tokens must be non-negative, got {tokens}")

        self.api_calls += 1
        self.cost_usd += cost
        self.tokens += tokens
        self.last_updated = datetime.now()

    def add_web_search(self) -> None:
        """Record a web search."""
        self.web_searches += 1
        self.last_updated = datetime.now()

    def add_tool_invocation(self) -> None:
        """Record a tool invocation."""
        self.tool_invocations += 1
        self.last_updated = datetime.now()

    def update_memory(self, memory_mb: float) -> None:
        """Update memory usage (tracks peak).

        Args:
            memory_mb: Current memory usage in MB

        Raises:
            ValueError: If memory_mb is negative
        """
        if memory_mb < 0:
            raise ValueError(f"Memory must be non-negative, got {memory_mb}")
        self.memory_mb = max(self.memory_mb, memory_mb)
        self.last_updated = datetime.now()

    def add_compute_time(self, seconds: float) -> None:
        """Add compute time.

        Args:
            seconds: Compute time to add in seconds

        Raises:
            ValueError: If seconds is negative
        """
        if seconds < 0:
            raise ValueError(f"Compute time must be non-negative, got {seconds}")
        self.compute_seconds += seconds
        self.last_updated = datetime.now()

    def add_cost(self, cost_usd: float) -> None:
        """Add cost.

        Args:
            cost_usd: Cost to add in USD

        Raises:
            ValueError: If cost_usd is negative
        """
        if cost_usd < 0:
            raise ValueError(f"Cost must be non-negative, got {cost_usd}")
        self.cost_usd += cost_usd
        self.last_updated = datetime.now()

    def elapsed_time(self) -> timedelta:
        """Calculate elapsed time since tracking started.

        Returns:
            Time elapsed since start_time
        """
        return datetime.now() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert usage to dictionary format.

        Returns:
            Dictionary representation of resource usage
        """
        return {
            "tokens": self.tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "text_tokens": self.text_tokens,
            "api_calls": self.api_calls,
            "web_searches": self.web_searches,
            "tool_invocations": self.tool_invocations,
            "memory_mb": self.memory_mb,
            "compute_seconds": self.compute_seconds,
            "cost_usd": self.cost_usd,
            "elapsed_seconds": self.elapsed_time().total_seconds(),
            "start_time": self.start_time.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    def __repr__(self) -> str:
        """String representation of resource usage."""
        if self.reasoning_tokens > 0 or self.text_tokens > 0:
            return (
                f"ResourceUsage(tokens={self.tokens} "
                f"[reasoning={self.reasoning_tokens}, text={self.text_tokens}], "
                f"api_calls={self.api_calls}, "
                f"cost_usd={self.cost_usd:.4f}, elapsed={self.elapsed_time()})"
            )
        return (
            f"ResourceUsage(tokens={self.tokens}, api_calls={self.api_calls}, "
            f"cost_usd={self.cost_usd:.4f}, elapsed={self.elapsed_time()})"
        )


@dataclass
class ViolationInfo:
    """Information about a constraint violation.

    Attributes:
        resource: Name of the violated resource
        limit: The constraint limit that was exceeded
        actual: The actual usage value
        timestamp: When the violation occurred
    """

    resource: str
    limit: float
    actual: float
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        """String representation of violation."""
        return f"ViolationInfo({self.resource}: {self.actual} > {self.limit})"


class ResourceMonitor:
    """Monitors resource usage and validates against constraints.

    This class provides real-time monitoring of resource consumption and checks
    whether usage exceeds contract constraints.

    Attributes:
        constraints: Resource constraints to monitor against
        usage: Current resource usage
        violations: List of detected violations
    """

    def __init__(self, constraints: ResourceConstraints) -> None:
        """Initialize resource monitor.

        Args:
            constraints: Resource constraints to enforce
        """
        self.constraints = constraints
        self.usage = ResourceUsage()
        self.violations: list[ViolationInfo] = []

    def check_constraints(self) -> list[ViolationInfo]:
        """Check if current usage violates any constraints.

        Returns:
            List of violations (empty if all constraints satisfied)
        """
        violations: list[ViolationInfo] = []

        # Check each resource constraint
        if self.constraints.tokens is not None and self.usage.tokens > self.constraints.tokens:
            violations.append(
                ViolationInfo(
                    resource="tokens", limit=self.constraints.tokens, actual=self.usage.tokens
                )
            )

        # Check reasoning tokens separately if specified
        if (
            self.constraints.reasoning_tokens is not None
            and self.usage.reasoning_tokens > self.constraints.reasoning_tokens
        ):
            violations.append(
                ViolationInfo(
                    resource="reasoning_tokens",
                    limit=self.constraints.reasoning_tokens,
                    actual=self.usage.reasoning_tokens,
                )
            )

        # Check text tokens separately if specified
        if (
            self.constraints.text_tokens is not None
            and self.usage.text_tokens > self.constraints.text_tokens
        ):
            violations.append(
                ViolationInfo(
                    resource="text_tokens",
                    limit=self.constraints.text_tokens,
                    actual=self.usage.text_tokens,
                )
            )

        if (
            self.constraints.api_calls is not None
            and self.usage.api_calls > self.constraints.api_calls
        ):
            violations.append(
                ViolationInfo(
                    resource="api_calls",
                    limit=self.constraints.api_calls,
                    actual=self.usage.api_calls,
                )
            )

        if (
            self.constraints.web_searches is not None
            and self.usage.web_searches > self.constraints.web_searches
        ):
            violations.append(
                ViolationInfo(
                    resource="web_searches",
                    limit=self.constraints.web_searches,
                    actual=self.usage.web_searches,
                )
            )

        if (
            self.constraints.tool_invocations is not None
            and self.usage.tool_invocations > self.constraints.tool_invocations
        ):
            violations.append(
                ViolationInfo(
                    resource="tool_invocations",
                    limit=self.constraints.tool_invocations,
                    actual=self.usage.tool_invocations,
                )
            )

        if (
            self.constraints.memory_mb is not None
            and self.usage.memory_mb > self.constraints.memory_mb
        ):
            violations.append(
                ViolationInfo(
                    resource="memory_mb",
                    limit=self.constraints.memory_mb,
                    actual=self.usage.memory_mb,
                )
            )

        if (
            self.constraints.compute_seconds is not None
            and self.usage.compute_seconds > self.constraints.compute_seconds
        ):
            violations.append(
                ViolationInfo(
                    resource="compute_seconds",
                    limit=self.constraints.compute_seconds,
                    actual=self.usage.compute_seconds,
                )
            )

        if (
            self.constraints.cost_usd is not None
            and self.usage.cost_usd > self.constraints.cost_usd
        ):
            violations.append(
                ViolationInfo(
                    resource="cost_usd", limit=self.constraints.cost_usd, actual=self.usage.cost_usd
                )
            )

        return violations

    def is_violated(self) -> bool:
        """Check if any constraints are currently violated.

        Returns:
            True if any constraint is violated, False otherwise
        """
        return len(self.check_constraints()) > 0

    def record_violation(self, violation: ViolationInfo) -> None:
        """Record a constraint violation.

        Args:
            violation: The violation information to record
        """
        self.violations.append(violation)

    def get_usage_percentage(self) -> dict[str, float]:
        """Calculate usage as percentage of constraints.

        Returns:
            Dictionary mapping resource names to usage percentages (0-100+)
            Resources without constraints are excluded
        """
        percentages: dict[str, float] = {}

        if self.constraints.tokens is not None and self.constraints.tokens > 0:
            percentages["tokens"] = (self.usage.tokens / self.constraints.tokens) * 100

        if self.constraints.reasoning_tokens is not None and self.constraints.reasoning_tokens > 0:
            percentages["reasoning_tokens"] = (
                self.usage.reasoning_tokens / self.constraints.reasoning_tokens
            ) * 100

        if self.constraints.text_tokens is not None and self.constraints.text_tokens > 0:
            percentages["text_tokens"] = (
                self.usage.text_tokens / self.constraints.text_tokens
            ) * 100

        if self.constraints.api_calls is not None and self.constraints.api_calls > 0:
            percentages["api_calls"] = (self.usage.api_calls / self.constraints.api_calls) * 100

        if self.constraints.web_searches is not None and self.constraints.web_searches > 0:
            percentages["web_searches"] = (
                self.usage.web_searches / self.constraints.web_searches
            ) * 100

        if self.constraints.tool_invocations is not None and self.constraints.tool_invocations > 0:
            percentages["tool_invocations"] = (
                self.usage.tool_invocations / self.constraints.tool_invocations
            ) * 100

        if self.constraints.memory_mb is not None and self.constraints.memory_mb > 0:
            percentages["memory_mb"] = (self.usage.memory_mb / self.constraints.memory_mb) * 100

        if self.constraints.compute_seconds is not None and self.constraints.compute_seconds > 0:
            percentages["compute_seconds"] = (
                self.usage.compute_seconds / self.constraints.compute_seconds
            ) * 100

        if self.constraints.cost_usd is not None and self.constraints.cost_usd > 0:
            percentages["cost_usd"] = (self.usage.cost_usd / self.constraints.cost_usd) * 100

        return percentages

    def reset(self) -> None:
        """Reset usage tracking and clear violations."""
        self.usage = ResourceUsage()
        self.violations = []

    def __repr__(self) -> str:
        """String representation of monitor."""
        violated = "VIOLATED" if self.is_violated() else "OK"
        return f"ResourceMonitor(status={violated}, usage={self.usage})"
