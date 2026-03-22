"""Tests for shared token extraction utilities."""

from agent_contracts.integrations._token_utils import (
    estimate_cost,
    extract_tokens_from_llm_result,
)


class TestExtractTokensFromLlmResult:
    def test_openai_style_token_usage(self) -> None:
        """Extract from llm_output['token_usage']."""
        assert (
            extract_tokens_from_llm_result(llm_output={"token_usage": {"total_tokens": 150}}) == 150
        )

    def test_google_style_usage_metadata(self) -> None:
        """Extract from llm_output['usage_metadata']."""
        assert (
            extract_tokens_from_llm_result(llm_output={"usage_metadata": {"total_tokens": 200}})
            == 200
        )

    def test_generations_metadata(self) -> None:
        """Extract from generations response metadata."""
        assert (
            extract_tokens_from_llm_result(
                generations_metadata={"usage_metadata": {"total_tokens": 300}}
            )
            == 300
        )

    def test_no_token_data(self) -> None:
        """Return 0 when no token data found."""
        assert extract_tokens_from_llm_result(llm_output={}) == 0
        assert extract_tokens_from_llm_result(llm_output=None) == 0
        assert extract_tokens_from_llm_result() == 0


class TestEstimateCost:
    def test_fallback_rate(self) -> None:
        """Should use fallback rate for unknown models."""
        cost = estimate_cost(total_tokens=1_000_000, model=None)
        assert cost == 1_000_000 * 0.00000015

    def test_known_model_with_separate_tokens(self) -> None:
        """Should use per-direction pricing when input/output provided."""
        cost = estimate_cost(
            total_tokens=0,
            model="gpt-4",
            input_tokens=500,
            output_tokens=500,
        )
        assert cost > 0

    def test_known_model_total_tokens_only(self) -> None:
        """Should use average rate when only total_tokens provided for known model."""
        from agent_contracts.core.tokens import MODEL_PRICING

        model = "gpt-4"
        pricing = MODEL_PRICING[model]
        expected_avg_rate = (pricing["input"] + pricing["output"]) / 2
        cost = estimate_cost(total_tokens=1000, model=model)
        assert cost == 1000 * expected_avg_rate

    def test_zero_tokens(self) -> None:
        """Zero tokens should return zero cost."""
        assert estimate_cost(total_tokens=0) == 0.0
