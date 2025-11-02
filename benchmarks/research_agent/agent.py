"""Base research agent with decompose-research-synthesize workflow."""

from dataclasses import dataclass, field
from typing import Any, cast

from litellm import completion


@dataclass
class ResearchStep:
    """A single step in the research workflow.

    Attributes:
        step_type: Type of step (decompose, research, synthesize)
        input_text: Input to this step
        output_text: Output from this step
        tokens_used: Tokens consumed in this step
        cost_usd: Cost of this step
        reasoning_tokens: Reasoning tokens used (if available)
        text_tokens: Text output tokens (if available)
    """

    step_type: str
    input_text: str
    output_text: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    reasoning_tokens: int = 0
    text_tokens: int = 0


@dataclass
class ResearchResult:
    """Result of a research workflow.

    Attributes:
        question: Original research question
        subquestions: Decomposed sub-questions
        subanswers: Answers to each sub-question
        final_answer: Synthesized final answer
        steps: All steps taken during research
        total_tokens: Total tokens consumed
        total_cost: Total cost in USD
        total_reasoning_tokens: Total reasoning tokens used
        total_text_tokens: Total text output tokens used
        api_calls: Number of API calls made
    """

    question: str
    subquestions: list[str] = field(default_factory=list)
    subanswers: list[str] = field(default_factory=list)
    final_answer: str = ""
    steps: list[ResearchStep] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    total_reasoning_tokens: int = 0
    total_text_tokens: int = 0
    api_calls: int = 0

    def add_step(self, step: ResearchStep) -> None:
        """Add a step and update totals."""
        self.steps.append(step)
        self.total_tokens += step.tokens_used
        self.total_cost += step.cost_usd
        self.total_reasoning_tokens += step.reasoning_tokens
        self.total_text_tokens += step.text_tokens
        self.api_calls += 1


class ResearchAgent:
    """Base research agent with multi-step workflow.

    This agent implements a decompose-research-synthesize pattern:
    1. Decompose: Break question into sub-questions
    2. Research: Answer each sub-question independently
    3. Synthesize: Combine answers into final response

    This is an abstract base class. Subclasses should implement:
    - _call_llm() to control how LLM calls are made
    """

    def __init__(self, model: str = "gemini/gemini-2.5-flash-preview-09-2025") -> None:
        """Initialize research agent.

        Args:
            model: LLM model to use
        """
        self.model = model

    def research(self, question: str) -> ResearchResult:
        """Execute full research workflow.

        Args:
            question: Research question to answer

        Returns:
            ResearchResult with all steps and final answer
        """
        result = ResearchResult(question=question)

        # Step 1: Decompose question into sub-questions
        subquestions = self._decompose(question, result)
        result.subquestions = subquestions

        # Step 2: Research each sub-question
        subanswers = []
        for subq in subquestions:
            answer = self._research_subquestion(subq, result)
            subanswers.append(answer)
        result.subanswers = subanswers

        # Step 3: Synthesize final answer
        final_answer = self._synthesize(question, subquestions, subanswers, result)
        result.final_answer = final_answer

        return result

    def _decompose(self, question: str, result: ResearchResult) -> list[str]:
        """Decompose question into sub-questions.

        Args:
            question: Main research question
            result: ResearchResult to update

        Returns:
            List of sub-questions
        """
        prompt = f"""You are a research assistant. Your task is to break down a complex research question into 3-5 focused sub-questions that, when answered together, will provide a comprehensive response to the main question.

Main Question: {question}

Generate 3-5 sub-questions, one per line. Each sub-question should be:
- Specific and focused on one aspect
- Answerable independently
- Relevant to the main question

Output format (just the questions, no numbering):
<sub-question>
<sub-question>
<sub-question>
..."""

        response = self._call_llm(
            messages=[{"role": "user", "content": prompt}],
            step_type="decompose",
        )

        # Extract sub-questions from response
        content = response["choices"][0]["message"]["content"]
        subquestions = [q.strip() for q in content.strip().split("\n") if q.strip()]

        # Record step
        usage = response.get("usage", {})
        step = ResearchStep(
            step_type="decompose",
            input_text=question,
            output_text=content,
            tokens_used=usage.get("total_tokens", 0),
            cost_usd=response.get("_hidden_params", {}).get("response_cost", 0),
            reasoning_tokens=self._extract_reasoning_tokens(usage),
            text_tokens=self._extract_text_tokens(usage),
        )
        result.add_step(step)

        return subquestions

    def _research_subquestion(self, subquestion: str, result: ResearchResult) -> str:
        """Research a single sub-question.

        Args:
            subquestion: Sub-question to answer
            result: ResearchResult to update

        Returns:
            Answer to the sub-question
        """
        prompt = f"""You are a research assistant. Provide a detailed, accurate answer to the following question. Include key facts, explanations, and relevant context.

Question: {subquestion}

Provide a comprehensive answer (3-5 paragraphs):"""

        response = self._call_llm(
            messages=[{"role": "user", "content": prompt}],
            step_type="research",
        )

        content = cast("str", response["choices"][0]["message"]["content"])

        # Record step
        usage = response.get("usage", {})
        step = ResearchStep(
            step_type="research",
            input_text=subquestion,
            output_text=content,
            tokens_used=usage.get("total_tokens", 0),
            cost_usd=response.get("_hidden_params", {}).get("response_cost", 0),
            reasoning_tokens=self._extract_reasoning_tokens(usage),
            text_tokens=self._extract_text_tokens(usage),
        )
        result.add_step(step)

        return content

    def _synthesize(
        self,
        question: str,
        subquestions: list[str],
        subanswers: list[str],
        result: ResearchResult,
    ) -> str:
        """Synthesize sub-answers into final answer.

        Args:
            question: Original research question
            subquestions: List of sub-questions
            subanswers: Answers to each sub-question
            result: ResearchResult to update

        Returns:
            Final synthesized answer
        """
        # Build context from sub-answers
        context = ""
        for i, (subq, ans) in enumerate(zip(subquestions, subanswers, strict=True), 1):
            context += f"\n\nSub-question {i}: {subq}\nAnswer: {ans}"

        prompt = f"""You are a research assistant. You have researched several sub-questions related to a main question. Now synthesize these findings into a comprehensive, coherent answer to the main question.

Main Question: {question}

Research Findings:{context}

Provide a well-structured final answer that:
- Addresses the main question directly
- Integrates insights from all sub-answers
- Is coherent and logically organized
- Highlights key tradeoffs and conclusions

Final Answer:"""

        response = self._call_llm(
            messages=[{"role": "user", "content": prompt}],
            step_type="synthesize",
        )

        content = cast("str", response["choices"][0]["message"]["content"])

        # Record step
        usage = response.get("usage", {})
        step = ResearchStep(
            step_type="synthesize",
            input_text=f"{question}\n\nContext: {len(context)} chars",
            output_text=content,
            tokens_used=usage.get("total_tokens", 0),
            cost_usd=response.get("_hidden_params", {}).get("response_cost", 0),
            reasoning_tokens=self._extract_reasoning_tokens(usage),
            text_tokens=self._extract_text_tokens(usage),
        )
        result.add_step(step)

        return content

    def _call_llm(self, messages: list[dict[str, str]], step_type: str) -> Any:
        """Call LLM (to be overridden by subclasses).

        Args:
            messages: Messages to send to LLM
            step_type: Type of step (decompose, research, synthesize)

        Returns:
            LLM response
        """
        # Default implementation: direct litellm call with temperature=0 for reproducibility
        return completion(model=self.model, messages=messages, temperature=0)

    def _extract_reasoning_tokens(self, usage: dict[str, Any]) -> int:
        """Extract reasoning tokens from usage data.

        Args:
            usage: Usage data from LLM response

        Returns:
            Number of reasoning tokens
        """
        details = usage.get("completion_tokens_details")
        if not details:
            return 0

        if isinstance(details, dict):
            return cast("int", details.get("reasoning_tokens", 0))
        else:
            return cast("int", getattr(details, "reasoning_tokens", 0) or 0)

    def _extract_text_tokens(self, usage: dict[str, Any]) -> int:
        """Extract text output tokens from usage data.

        Args:
            usage: Usage data from LLM response

        Returns:
            Number of text output tokens
        """
        details = usage.get("completion_tokens_details")
        if not details:
            # Fallback: all completion tokens are text
            return cast("int", usage.get("completion_tokens", 0))

        if isinstance(details, dict):
            text = details.get("text_tokens", 0)
            # If no breakdown, use all completion tokens
            if text == 0:
                return cast("int", usage.get("completion_tokens", 0))
            return cast("int", text)
        else:
            text = getattr(details, "text_tokens", 0) or 0
            if text == 0:
                return cast("int", usage.get("completion_tokens", 0))
            return cast("int", text)
