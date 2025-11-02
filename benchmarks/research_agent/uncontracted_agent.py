"""Uncontracted research agent (baseline for comparison)."""

from typing import Any

from litellm import completion

from benchmarks.research_agent.agent import ResearchAgent


class UncontractedResearchAgent(ResearchAgent):
    """Research agent with no contract enforcement (baseline).

    This agent:
    - Uses high reasoning effort for ALL steps (wasteful)
    - Has no budget awareness
    - No resource optimization

    This serves as the baseline to demonstrate the value of contracts.
    """

    def __init__(self, model: str = "gemini/gemini-2.5-flash-preview-09-2025") -> None:
        """Initialize uncontracted research agent.

        Args:
            model: LLM model to use
        """
        super().__init__(model=model)

    def _call_llm(self, messages: list[dict[str, str]], step_type: str) -> Any:
        """Call LLM with high reasoning effort (no optimization).

        Args:
            messages: Messages to send to LLM
            step_type: Type of step (ignored - uses high effort everywhere)

        Returns:
            LLM response
        """
        # No contract enforcement - just call litellm directly
        # Use high reasoning effort for everything (wasteful!)
        return completion(
            model=self.model,
            messages=messages,
            reasoning_effort="high",  # Always use high effort (no optimization)
            temperature=0,  # Deterministic for reproducibility
        )
