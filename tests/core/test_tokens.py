"""Unit tests for token counting and cost tracking."""

import pytest

from agent_contracts.core import (
    CostEstimate,
    TokenCount,
    TokenCounter,
    estimate_cost,
    estimate_tokens,
)


class TestTokenCount:
    """Tests for TokenCount data class."""

    def test_create_token_count(self) -> None:
        """Test creating a token count."""
        count = TokenCount(input_tokens=100, output_tokens=50)
        assert count.input_tokens == 100
        assert count.output_tokens == 50
        assert count.total_tokens == 150

    def test_total_tokens_property(self) -> None:
        """Test total_tokens is calculated correctly."""
        count = TokenCount(input_tokens=75, output_tokens=25)
        assert count.total_tokens == 100

    def test_repr(self) -> None:
        """Test string representation."""
        count = TokenCount(input_tokens=100, output_tokens=50)
        repr_str = repr(count)
        assert "input=100" in repr_str
        assert "output=50" in repr_str
        assert "total=150" in repr_str


class TestCostEstimate:
    """Tests for CostEstimate data class."""

    def test_create_cost_estimate(self) -> None:
        """Test creating a cost estimate."""
        cost = CostEstimate(input_cost=0.001, output_cost=0.002, model="gpt-4")
        assert cost.input_cost == 0.001
        assert cost.output_cost == 0.002
        assert cost.total_cost == 0.003
        assert cost.model == "gpt-4"

    def test_total_cost_property(self) -> None:
        """Test total_cost is calculated correctly."""
        cost = CostEstimate(input_cost=0.0015, output_cost=0.0025, model="claude-3-opus")
        assert cost.total_cost == 0.004

    def test_repr(self) -> None:
        """Test string representation."""
        cost = CostEstimate(input_cost=0.001, output_cost=0.002, model="gpt-4")
        repr_str = repr(cost)
        assert "input=$0.001000" in repr_str
        assert "output=$0.002000" in repr_str
        assert "total=$0.003000" in repr_str
        assert "gpt-4" in repr_str


class TestTokenCounter:
    """Tests for TokenCounter class."""

    def test_estimate_tokens_simple(self) -> None:
        """Test simple token estimation."""
        # Roughly 4 characters per token
        text = "This is a test message"  # 22 chars -> ~5 tokens
        tokens = TokenCounter.estimate_tokens(text)
        assert tokens >= 4
        assert tokens <= 8

    def test_estimate_tokens_empty(self) -> None:
        """Test token estimation for empty string."""
        tokens = TokenCounter.estimate_tokens("")
        assert tokens == 1  # Minimum 1 token

    def test_estimate_tokens_long_text(self) -> None:
        """Test token estimation for longer text."""
        text = "a" * 400  # 400 chars -> ~100 tokens
        tokens = TokenCounter.estimate_tokens(text)
        assert tokens == 100

    def test_count_messages_tokens_simple(self) -> None:
        """Test counting tokens in simple messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        tokens = TokenCounter.count_messages_tokens(messages)
        # 2 messages * 3 overhead + content tokens
        assert tokens >= 6  # At least message overhead
        assert tokens <= 20  # Reasonable upper bound

    def test_count_messages_tokens_empty(self) -> None:
        """Test counting tokens in empty message list."""
        tokens = TokenCounter.count_messages_tokens([])
        assert tokens == 0

    def test_count_messages_tokens_multimodal(self) -> None:
        """Test counting tokens with multimodal content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "url": "https://example.com/image.jpg"},
                ],
            }
        ]
        tokens = TokenCounter.count_messages_tokens(messages)
        # Should include text tokens + image tokens (~85) + overhead
        assert tokens >= 90  # At least image + some text

    def test_get_model_pricing_known_model(self) -> None:
        """Test getting pricing for known models."""
        pricing = TokenCounter.get_model_pricing("gpt-4")
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] > 0
        assert pricing["output"] > 0

    def test_get_model_pricing_unknown_model(self) -> None:
        """Test getting pricing for unknown model."""
        pricing = TokenCounter.get_model_pricing("unknown-model-xyz")
        assert pricing is None

    def test_get_model_pricing_case_insensitive(self) -> None:
        """Test that model name matching is case-insensitive."""
        pricing1 = TokenCounter.get_model_pricing("GPT-4")
        pricing2 = TokenCounter.get_model_pricing("gpt-4")
        assert pricing1 == pricing2

    def test_get_model_pricing_versioned(self) -> None:
        """Test getting pricing for versioned model names."""
        # Should match "gpt-4" pricing for "gpt-4-0613"
        pricing = TokenCounter.get_model_pricing("gpt-4-0613")
        assert pricing is not None
        assert pricing["input"] > 0

    def test_calculate_cost_gpt4(self) -> None:
        """Test cost calculation for GPT-4."""
        token_count = TokenCount(input_tokens=1000, output_tokens=500)
        cost = TokenCounter.calculate_cost(token_count, "gpt-4")

        assert cost.model == "gpt-4"
        assert cost.input_cost > 0
        assert cost.output_cost > 0
        assert cost.total_cost > 0
        # GPT-4: input $30/1M tokens, output $60/1M tokens
        assert abs(cost.input_cost - 0.03) < 0.001
        assert abs(cost.output_cost - 0.03) < 0.001

    def test_calculate_cost_claude(self) -> None:
        """Test cost calculation for Claude."""
        token_count = TokenCount(input_tokens=1000, output_tokens=500)
        cost = TokenCounter.calculate_cost(token_count, "claude-3-opus")

        assert cost.model == "claude-3-opus"
        # Claude Opus: input $15/1M tokens, output $75/1M tokens
        assert abs(cost.input_cost - 0.015) < 0.001
        assert abs(cost.output_cost - 0.0375) < 0.001

    def test_calculate_cost_custom_pricing(self) -> None:
        """Test cost calculation with custom pricing."""
        token_count = TokenCount(input_tokens=1000, output_tokens=500)
        custom_pricing = {"input": 0.00001, "output": 0.00002}

        cost = TokenCounter.calculate_cost(token_count, "custom-model", custom_pricing)

        assert cost.model == "custom-model"
        assert cost.input_cost == 0.01  # 1000 * 0.00001
        assert cost.output_cost == 0.01  # 500 * 0.00002

    def test_calculate_cost_unknown_model_raises_error(self) -> None:
        """Test that unknown model without custom pricing raises error."""
        token_count = TokenCount(input_tokens=1000, output_tokens=500)

        with pytest.raises(ValueError, match="Pricing not found"):
            TokenCounter.calculate_cost(token_count, "unknown-model")

    def test_estimate_completion_cost_with_string(self) -> None:
        """Test estimating completion cost with string input."""
        input_text = "What is the capital of France?"
        output_text = "The capital of France is Paris."

        token_count, cost = TokenCounter.estimate_completion_cost(
            input_text, output_text, "gpt-4o-mini"
        )

        assert token_count.input_tokens > 0
        assert token_count.output_tokens > 0
        assert cost.total_cost > 0
        assert cost.model == "gpt-4o-mini"

    def test_estimate_completion_cost_with_messages(self) -> None:
        """Test estimating completion cost with messages input."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        output_text = "2 + 2 equals 4."

        token_count, cost = TokenCounter.estimate_completion_cost(messages, output_text, "gpt-4o")

        assert token_count.input_tokens > 0
        assert token_count.output_tokens > 0
        assert cost.total_cost > 0

    def test_estimate_completion_cost_realistic_scenario(self) -> None:
        """Test realistic completion cost estimation."""
        # Simulate a code review task
        input_text = """
        Review this Python function for potential issues:

        def process_data(data):
            result = []
            for item in data:
                if item > 0:
                    result.append(item * 2)
            return result
        """

        output_text = """
        This function looks good overall, but here are some suggestions:
        1. Add type hints for better code clarity
        2. Consider using list comprehension for better performance
        3. Add docstring to document the function's purpose

        Improved version:
        def process_data(data: list[int]) -> list[int]:
            '''Double all positive numbers in the input list.'''
            return [item * 2 for item in data if item > 0]
        """

        token_count, cost = TokenCounter.estimate_completion_cost(input_text, output_text, "gpt-4o")

        # Verify reasonable token counts
        assert 50 < token_count.input_tokens < 200
        assert 100 < token_count.output_tokens < 300
        assert cost.total_cost > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_estimate_tokens_function(self) -> None:
        """Test estimate_tokens convenience function."""
        tokens = estimate_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10

    def test_estimate_cost_function_with_string(self) -> None:
        """Test estimate_cost convenience function with string input."""
        cost = estimate_cost("Hello", "Hi there", "gpt-4o-mini")
        assert cost > 0
        assert cost < 0.001  # Should be very cheap for short text

    def test_estimate_cost_function_with_messages(self) -> None:
        """Test estimate_cost convenience function with messages."""
        messages = [{"role": "user", "content": "Test"}]
        cost = estimate_cost(messages, "Response", "gpt-4o-mini")
        assert cost > 0


class TestModelPricing:
    """Tests for model pricing database."""

    def test_all_openai_models_have_pricing(self) -> None:
        """Test that all major OpenAI models have pricing."""
        openai_models = [
            "gpt-4",
            "gpt-4-32k",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        ]

        for model in openai_models:
            pricing = TokenCounter.get_model_pricing(model)
            assert pricing is not None, f"Missing pricing for {model}"
            assert pricing["input"] > 0
            assert pricing["output"] > 0

    def test_all_claude_models_have_pricing(self) -> None:
        """Test that all major Claude models have pricing."""
        claude_models = [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-haiku",
        ]

        for model in claude_models:
            pricing = TokenCounter.get_model_pricing(model)
            assert pricing is not None, f"Missing pricing for {model}"
            assert pricing["input"] > 0
            assert pricing["output"] > 0

    def test_output_more_expensive_than_input(self) -> None:
        """Test that output tokens are generally more expensive than input."""
        for model in ["gpt-4", "claude-3-opus", "gpt-4o"]:
            pricing = TokenCounter.get_model_pricing(model)
            assert pricing is not None
            assert pricing["output"] > pricing["input"], f"Output should cost more for {model}"

    def test_pricing_values_reasonable(self) -> None:
        """Test that pricing values are in reasonable ranges."""
        for model in ["gpt-4", "claude-3-opus"]:
            pricing = TokenCounter.get_model_pricing(model)
            assert pricing is not None

            # Prices should be in reasonable range (not zero, not absurdly high)
            assert 0.0000001 < pricing["input"] < 0.001  # $0.1 to $1000 per 1M tokens
            assert 0.0000001 < pricing["output"] < 0.001


class TestIntegration:
    """Integration tests for token counting and cost tracking."""

    def test_full_workflow_simple_completion(self) -> None:
        """Test complete workflow for a simple completion."""
        # Step 1: Estimate tokens and cost
        prompt = "Explain quantum computing in simple terms."
        response = "Quantum computing uses quantum bits that can be 0 and 1 at the same time."

        token_count, cost = TokenCounter.estimate_completion_cost(prompt, response, "gpt-4o-mini")

        # Step 2: Verify token counts
        assert token_count.input_tokens > 0
        assert token_count.output_tokens > 0
        assert token_count.total_tokens == token_count.input_tokens + token_count.output_tokens

        # Step 3: Verify cost calculation
        assert cost.total_cost > 0
        assert cost.input_cost > 0
        assert cost.output_cost > 0
        assert abs(cost.total_cost - (cost.input_cost + cost.output_cost)) < 0.000001

        # Step 4: Verify reasonable values
        assert cost.total_cost < 0.01  # Should be very cheap for short text with mini model

    def test_cost_comparison_across_models(self) -> None:
        """Test that costs vary appropriately across different models."""
        prompt = "Write a short story about a robot."
        response = "Once upon a time, there was a robot named Bob..."

        # Compare costs across models
        models = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
        costs = []

        for model in models:
            _, cost = TokenCounter.estimate_completion_cost(prompt, response, model)
            costs.append(cost.total_cost)

        # Verify that more advanced models cost more
        # gpt-4o-mini < gpt-4o < gpt-4
        assert costs[0] < costs[1] < costs[2]
