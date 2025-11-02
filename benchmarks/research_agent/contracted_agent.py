"""Contract-enforced research agent with strategic budget allocation."""

from typing import Any

from agent_contracts import Contract, ContractedLLM, ResourceConstraints
from benchmarks.research_agent.agent import ResearchAgent


class ContractedResearchAgent(ResearchAgent):
    """Research agent with contract-enforced resource budgets.

    This agent uses different contracts for each step:
    - Decompose: Low reasoning effort (simple planning)
    - Research: High reasoning effort (deep analysis)
    - Synthesize: Medium reasoning effort (integration)

    This demonstrates strategic resource allocation based on task complexity.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash-preview-09-2025",
        strict_mode: bool = False,
    ) -> None:
        """Initialize contracted research agent.

        Args:
            model: LLM model to use
            strict_mode: Whether to enforce contracts strictly (default False for benchmarking)
        """
        super().__init__(model=model)
        self.strict_mode = strict_mode

    def _call_llm(self, messages: list[dict[str, str]], step_type: str) -> Any:
        """Call LLM with appropriate contract based on step type.

        Args:
            messages: Messages to send to LLM
            step_type: Type of step (decompose, research, synthesize)

        Returns:
            LLM response
        """
        # Create a fresh contract for each call (contracts have state)
        if step_type == "decompose":
            contract = self._default_planning_contract()
        elif step_type == "research":
            contract = self._default_exploration_contract()
        elif step_type == "synthesize":
            contract = self._default_synthesis_contract()
        else:
            raise ValueError(f"Unknown step type: {step_type}")

        # Create fresh ContractedLLM with auto_start
        with ContractedLLM(contract=contract, strict_mode=self.strict_mode) as llm:
            response = llm.completion(model=self.model, messages=messages, temperature=0)

        return response

    def _default_planning_contract(self) -> Contract:
        """Create default contract for planning step.

        Returns:
            Contract with MEDIUM reasoning effort for quality decomposition
        """
        return Contract(
            id="planning",
            name="Question Decomposition",
            description="Decompose research question into sub-questions",
            resources=ResourceConstraints(
                reasoning_tokens=1200,  # MEDIUM effort - quality planning (was 500)
                text_tokens=300,  # List of 3-5 questions
                api_calls=1,
                cost_usd=0.015,  # Increased to match higher token budget
            ),
        )

    def _default_exploration_contract(self) -> Contract:
        """Create default contract for exploration step.

        Returns:
            Contract with high reasoning effort for deep research
        """
        return Contract(
            id="exploration",
            name="Deep Research",
            description="Research a sub-question in depth",
            resources=ResourceConstraints(
                reasoning_tokens=2500,  # High effort - deep technical analysis
                text_tokens=500,  # Detailed answer (3-5 paragraphs)
                api_calls=1,
                cost_usd=0.03,
            ),
        )

    def _default_synthesis_contract(self) -> Contract:
        """Create default contract for synthesis step.

        Returns:
            Contract with HIGH reasoning effort for quality synthesis
        """
        return Contract(
            id="synthesis",
            name="Answer Synthesis",
            description="Synthesize sub-answers into final answer",
            resources=ResourceConstraints(
                reasoning_tokens=2500,  # HIGH effort - complex technical synthesis (was 1200)
                text_tokens=800,  # Comprehensive final answer
                api_calls=1,
                cost_usd=0.03,  # Increased cost budget to match higher tokens
            ),
        )
