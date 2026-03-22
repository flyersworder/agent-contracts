"""Shared token extraction and cost estimation utilities for integrations.

Centralizes logic that was duplicated across LangChain and LangGraph integrations.
"""

from __future__ import annotations

from typing import Any

from agent_contracts.core.tokens import MODEL_PRICING

# Default fallback rate when model is unknown (~Gemini 2.5 Flash rate)
_DEFAULT_COST_PER_TOKEN = 0.00000015


def extract_tokens_from_llm_result(
    llm_output: dict[str, Any] | None = None,
    generations_metadata: dict[str, Any] | None = None,
) -> int:
    """Extract total token count from LLM response metadata.

    Checks multiple locations where different providers store token usage:
    1. OpenAI-style: llm_output["token_usage"]["total_tokens"]
    2. Google-style: llm_output["usage_metadata"]["total_tokens"]
    3. Generations metadata: usage_metadata in response_metadata

    Args:
        llm_output: The llm_output dict from LLMResult
        generations_metadata: Response metadata from generations[0][0].message

    Returns:
        Total token count, or 0 if not found
    """
    if llm_output:
        if "token_usage" in llm_output:
            return int(llm_output["token_usage"].get("total_tokens", 0))
        if "usage_metadata" in llm_output:
            return int(llm_output["usage_metadata"].get("total_tokens", 0))

    if generations_metadata and "usage_metadata" in generations_metadata:
        return int(generations_metadata["usage_metadata"].get("total_tokens", 0))

    return 0


def estimate_cost(
    total_tokens: int,
    model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> float:
    """Estimate cost for token usage.

    Uses MODEL_PRICING from tokens.py when available, falls back to default rate.
    When input_tokens and output_tokens are provided separately, applies the
    correct per-direction rate.

    Args:
        total_tokens: Total tokens consumed (used as fallback)
        model: Model name for pricing lookup
        input_tokens: Input tokens (for accurate per-direction pricing)
        output_tokens: Output tokens (for accurate per-direction pricing)

    Returns:
        Estimated cost in USD
    """
    if model and model in MODEL_PRICING:
        pricing = MODEL_PRICING[model]
        if input_tokens is not None and output_tokens is not None:
            return input_tokens * pricing["input"] + output_tokens * pricing["output"]
        avg_rate = (pricing["input"] + pricing["output"]) / 2
        return total_tokens * avg_rate

    return total_tokens * _DEFAULT_COST_PER_TOKEN
