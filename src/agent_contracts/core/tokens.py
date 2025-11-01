"""Token counting and cost tracking for LLM interactions.

This module provides utilities for counting tokens and estimating costs for various
LLM providers and models.
"""

from dataclasses import dataclass
from typing import Any

# Model pricing database (USD per token)
# Prices are as of November 2025 - should be updated periodically
MODEL_PRICING = {
    # OpenAI models (per 1M tokens)
    "gpt-4": {"input": 30.0 / 1_000_000, "output": 60.0 / 1_000_000},
    "gpt-4-32k": {"input": 60.0 / 1_000_000, "output": 120.0 / 1_000_000},
    "gpt-4-turbo": {"input": 10.0 / 1_000_000, "output": 30.0 / 1_000_000},
    "gpt-4o": {"input": 2.5 / 1_000_000, "output": 10.0 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.6 / 1_000_000},
    "gpt-3.5-turbo": {"input": 0.5 / 1_000_000, "output": 1.5 / 1_000_000},
    # Anthropic models (per 1M tokens)
    "claude-3-opus": {"input": 15.0 / 1_000_000, "output": 75.0 / 1_000_000},
    "claude-3-sonnet": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-3-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
    "claude-3.5-sonnet": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-3.5-haiku": {"input": 0.8 / 1_000_000, "output": 4.0 / 1_000_000},
}


@dataclass
class TokenCount:
    """Token count breakdown for a completion.

    Attributes:
        input_tokens: Number of tokens in the input/prompt
        output_tokens: Number of tokens in the output/completion
        total_tokens: Total tokens (input + output)
    """

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens."""
        return self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        """String representation of token count."""
        return (
            f"TokenCount(input={self.input_tokens}, "
            f"output={self.output_tokens}, total={self.total_tokens})"
        )


@dataclass
class CostEstimate:
    """Cost estimate for a completion.

    Attributes:
        input_cost: Cost of input tokens
        output_cost: Cost of output tokens
        total_cost: Total cost (input + output)
        model: Model name used for pricing
    """

    input_cost: float
    output_cost: float
    model: str

    @property
    def total_cost(self) -> float:
        """Calculate total cost."""
        return self.input_cost + self.output_cost

    def __repr__(self) -> str:
        """String representation of cost estimate."""
        return (
            f"CostEstimate(input=${self.input_cost:.6f}, "
            f"output=${self.output_cost:.6f}, total=${self.total_cost:.6f}, "
            f"model={self.model})"
        )


class TokenCounter:
    """Counts tokens and estimates costs for LLM completions.

    This class provides methods to count tokens in text and estimate costs
    based on model pricing.
    """

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text using simple heuristics.

        This provides a rough estimate based on character count. For accurate
        token counting, use provider-specific tokenizers.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated number of tokens
        """
        # Rough estimate: ~4 characters per token (works reasonably for English)
        # This is a simplified approximation
        return max(1, len(text) // 4)

    @staticmethod
    def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
        """Estimate token count for a list of messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Estimated total token count for all messages
        """
        total = 0

        # Account for message formatting overhead (~3 tokens per message)
        total += len(messages) * 3

        # Count content tokens
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total += TokenCounter.estimate_tokens(content)
            elif isinstance(content, list):
                # Handle multimodal content (e.g., images)
                for item in content:
                    if isinstance(item, dict):
                        text_content = item.get("text", "")
                        if text_content:
                            total += TokenCounter.estimate_tokens(text_content)
                        # Images typically count as ~85 tokens (for low detail)
                        # or 170 tokens per 512x512 tile (for high detail)
                        if item.get("type") == "image_url":
                            total += 85  # Conservative estimate

            # Account for role tokens
            role = message.get("role", "")
            total += len(role) // 4

        return total

    @staticmethod
    def get_model_pricing(model: str) -> dict[str, float] | None:
        """Get pricing information for a model.

        Args:
            model: Model name

        Returns:
            Dictionary with 'input' and 'output' pricing per token, or None if not found
        """
        # Normalize model name (remove version suffixes, etc.)
        model_lower = model.lower()

        # Try exact match first
        if model_lower in MODEL_PRICING:
            return MODEL_PRICING[model_lower]

        # Try prefix match for versioned models
        for known_model in MODEL_PRICING:
            if model_lower.startswith(known_model):
                return MODEL_PRICING[known_model]

        return None

    @staticmethod
    def calculate_cost(
        token_count: TokenCount, model: str, custom_pricing: dict[str, float] | None = None
    ) -> CostEstimate:
        """Calculate cost for a completion.

        Args:
            token_count: Token count breakdown
            model: Model name
            custom_pricing: Optional custom pricing override (dict with 'input' and 'output' keys)

        Returns:
            Cost estimate

        Raises:
            ValueError: If model pricing is not found and custom_pricing is not provided
        """
        pricing = custom_pricing or TokenCounter.get_model_pricing(model)

        if pricing is None:
            raise ValueError(
                f"Pricing not found for model '{model}'. "
                f"Please provide custom_pricing or use a known model."
            )

        input_cost = token_count.input_tokens * pricing["input"]
        output_cost = token_count.output_tokens * pricing["output"]

        return CostEstimate(input_cost=input_cost, output_cost=output_cost, model=model)

    @staticmethod
    def estimate_completion_cost(
        input_text: str | list[dict[str, Any]],
        output_text: str,
        model: str,
        custom_pricing: dict[str, float] | None = None,
    ) -> tuple[TokenCount, CostEstimate]:
        """Estimate tokens and cost for a completion.

        Args:
            input_text: Input text or messages
            output_text: Output text
            model: Model name
            custom_pricing: Optional custom pricing override

        Returns:
            Tuple of (token_count, cost_estimate)
        """
        # Count tokens
        if isinstance(input_text, str):
            input_tokens = TokenCounter.estimate_tokens(input_text)
        else:
            input_tokens = TokenCounter.count_messages_tokens(input_text)

        output_tokens = TokenCounter.estimate_tokens(output_text)

        token_count = TokenCount(input_tokens=input_tokens, output_tokens=output_tokens)

        # Calculate cost
        cost_estimate = TokenCounter.calculate_cost(token_count, model, custom_pricing)

        return token_count, cost_estimate


# Convenience functions
def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Args:
        text: Text to count tokens for

    Returns:
        Estimated number of tokens
    """
    return TokenCounter.estimate_tokens(text)


def estimate_cost(
    input_text: str | list[dict[str, Any]],
    output_text: str,
    model: str,
    custom_pricing: dict[str, float] | None = None,
) -> float:
    """Estimate cost for a completion.

    Args:
        input_text: Input text or messages
        output_text: Output text
        model: Model name
        custom_pricing: Optional custom pricing override

    Returns:
        Estimated total cost in USD
    """
    _, cost_estimate = TokenCounter.estimate_completion_cost(
        input_text, output_text, model, custom_pricing
    )
    return cost_estimate.total_cost
