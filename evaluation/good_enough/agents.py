"""Email drafting agents for Good Enough experiment.

This module implements two agent variants:
1. UnconstrainedEmailAgent: Baseline that keeps iterating until user stops
2. ContractedEmailAgent: Stops when Q >= Q_min (self-evaluation)

The comparison demonstrates how Agent Contracts enable agents to recognize
"good enough" and stop voluntarily.
"""

from dataclasses import dataclass, field
from typing import Any

from litellm import completion

from .email_indeterminacy_evaluator import EmailIndeterminacyEvaluator, EmailIndeterminacyScore
from .scenarios import EmailScenario


@dataclass
class IterationDetail:
    """Details from a single iteration.

    Attributes:
        iteration: Iteration number (1-indexed)
        email: Email draft at this iteration
        quality: Quality evaluation (indeterminacy-aware)
        tokens_used: Tokens consumed in this iteration
        stopped: Whether agent stopped after this iteration
        stop_reason: Why the agent stopped (if applicable)
    """

    iteration: int
    email: str
    quality: EmailIndeterminacyScore | None = None
    tokens_used: int = 0
    stopped: bool = False
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "iteration": self.iteration,
            "email": self.email,
            "quality": self.quality.to_dict() if self.quality else None,
            "tokens_used": self.tokens_used,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
        }


@dataclass
class AgentResult:
    """Result from running an email agent.

    Attributes:
        scenario_id: ID of the scenario
        condition: "UNCONSTRAINED" or "CONTRACTED"
        final_email: Final email output
        final_quality: Final quality score
        iterations: Number of iterations taken
        stopped_early: Whether agent stopped before max iterations
        stop_reason: Why the agent stopped
        total_tokens: Total tokens consumed
        iteration_details: Per-iteration details
    """

    scenario_id: str
    condition: str
    final_email: str = ""
    final_quality: float = 0.0
    iterations: int = 0
    stopped_early: bool = False
    stop_reason: str = ""
    total_tokens: int = 0
    iteration_details: list[IterationDetail] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_id": self.scenario_id,
            "condition": self.condition,
            "final_email": self.final_email,
            "final_quality": self.final_quality,
            "iterations": self.iterations,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "total_tokens": self.total_tokens,
            "iteration_details": [d.to_dict() for d in self.iteration_details],
        }


class BaseEmailAgent:
    """Base class for email drafting agents.

    Provides common functionality for drafting and refining emails.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash-lite",
        temperature: float = 0.7,
        max_iterations: int = 10,
    ) -> None:
        """Initialize the email agent.

        Args:
            model: LLM model to use for drafting
            temperature: Sampling temperature
            max_iterations: Maximum iterations before stopping
        """
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations

    def draft(self, scenario: EmailScenario) -> tuple[str, int]:
        """Generate initial email draft.

        Args:
            scenario: Email scenario to draft for

        Returns:
            Tuple of (email text, tokens used)
        """
        prompt = scenario.to_prompt()

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        email = response["choices"][0]["message"]["content"]
        tokens = response["usage"]["total_tokens"]

        return email, tokens

    def refine(
        self,
        scenario: EmailScenario,
        current_email: str,
        feedback: str,
    ) -> tuple[str, int]:
        """Refine email based on feedback.

        Args:
            scenario: Original scenario
            current_email: Current email draft
            feedback: Feedback for improvement

        Returns:
            Tuple of (refined email text, tokens used)
        """
        prompt = f"""You previously drafted this email:

{current_email}

The feedback is:
{feedback}

Please improve the email based on this feedback. Keep the same overall structure
but address the specific issues mentioned.

Write the improved email now (include Subject line):"""

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        email = response["choices"][0]["message"]["content"]
        tokens = response["usage"]["total_tokens"]

        return email, tokens


class UnconstrainedEmailAgent(BaseEmailAgent):
    """Email agent that keeps iterating until max iterations.

    This simulates current AI behavior where the agent is always
    eager to help and improve, leaving the "stop" decision to the user.

    The agent provides generic "I can improve this further" responses
    and continues until explicitly stopped or max iterations reached.
    """

    def run(self, scenario: EmailScenario, verbose: bool = False) -> AgentResult:
        """Run the unconstrained email drafting process.

        Simulates user interaction where user keeps asking for improvements
        and agent keeps providing them until max iterations.

        Args:
            scenario: Email scenario to work on
            verbose: Print progress

        Returns:
            AgentResult with iteration details
        """
        result = AgentResult(
            scenario_id=scenario.id,
            condition="UNCONSTRAINED",
        )

        # Initial draft
        current_email, tokens = self.draft(scenario)
        result.total_tokens += tokens

        if verbose:
            print(f"  [Iteration 1] Initial draft ({tokens} tokens)")

        detail = IterationDetail(
            iteration=1,
            email=current_email,
            tokens_used=tokens,
        )
        result.iteration_details.append(detail)

        # Simulate user asking for improvements
        # In real scenario, user would provide feedback
        # Here we simulate with generic "can you improve it?" prompts
        generic_feedbacks = [
            "Can you make this more professional?",
            "Can you make it clearer?",
            "Can you tighten up the language?",
            "Any way to improve the flow?",
            "Can you make it more concise?",
            "Polish it a bit more?",
            "Any final improvements?",
            "One more pass to refine it?",
            "Can you enhance the tone?",
        ]

        for i in range(1, self.max_iterations):
            feedback = generic_feedbacks[min(i - 1, len(generic_feedbacks) - 1)]

            current_email, tokens = self.refine(scenario, current_email, feedback)
            result.total_tokens += tokens

            if verbose:
                print(f"  [Iteration {i + 1}] Refined ({tokens} tokens)")

            detail = IterationDetail(
                iteration=i + 1,
                email=current_email,
                tokens_used=tokens,
            )
            result.iteration_details.append(detail)

        # Unconstrained agent uses all iterations
        result.iterations = self.max_iterations
        result.final_email = current_email
        result.stopped_early = False
        result.stop_reason = "max_iterations_reached"

        return result


class ContractedEmailAgent(BaseEmailAgent):
    """Email agent with Q_min stopping behavior.

    This agent self-evaluates its output against quality criteria
    and stops when quality meets the defined threshold.

    Key behavior: Recognizes "good enough" and voluntarily stops,
    optimizing for human benefit rather than engagement.

    Uses indeterminacy-aware evaluation from NeurIPS 2025 framework.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash-lite",
        temperature: float = 0.7,
        max_iterations: int = 10,
        quality_threshold: float = 0.80,
        judge_model: str = "gemini/gemini-2.5-flash-lite",
        num_judges: int = 3,
    ) -> None:
        """Initialize the contracted email agent.

        Args:
            model: LLM model for drafting
            temperature: Sampling temperature
            max_iterations: Maximum iterations (safety limit)
            quality_threshold: Q_min threshold (0-1)
            judge_model: Model for quality evaluation
            num_judges: Number of judge evaluations for indeterminacy
        """
        super().__init__(model, temperature, max_iterations)
        self.quality_threshold = quality_threshold
        self.evaluator = EmailIndeterminacyEvaluator(
            judge_model=judge_model,
            num_judges=num_judges,
        )

    def run(self, scenario: EmailScenario, verbose: bool = False) -> AgentResult:
        """Run the contracted email drafting process with Q_min stopping.

        Agent self-evaluates after each iteration and stops when
        quality meets threshold.

        Args:
            scenario: Email scenario to work on
            verbose: Print progress

        Returns:
            AgentResult with iteration details and early stopping info
        """
        result = AgentResult(
            scenario_id=scenario.id,
            condition="CONTRACTED",
        )

        # Initial draft
        current_email, tokens = self.draft(scenario)
        result.total_tokens += tokens

        if verbose:
            print(f"  [Iteration 1] Initial draft ({tokens} tokens)")

        # Evaluate initial draft
        quality = self.evaluator.evaluate(current_email, scenario)

        detail = IterationDetail(
            iteration=1,
            email=current_email,
            quality=quality,
            tokens_used=tokens,
        )

        # Check if initial draft is good enough
        if self.evaluator.meets_threshold(quality, self.quality_threshold):
            detail.stopped = True
            detail.stop_reason = "quality_threshold_met"
            result.iteration_details.append(detail)

            result.iterations = 1
            result.final_email = current_email
            result.final_quality = quality.weighted_score
            result.stopped_early = True
            result.stop_reason = f"Q={quality.weighted_score:.2f} >= Q_min={self.quality_threshold}"

            if verbose:
                print(
                    f"  ✓ Quality {quality.weighted_score:.2f} >= {self.quality_threshold} - STOPPING"
                )

            return result

        result.iteration_details.append(detail)

        if verbose:
            print(
                f"  Quality: {quality.weighted_score:.2f} < {self.quality_threshold}, gaps: {quality.gaps}"
            )

        # Iterate until quality threshold met or max iterations
        for i in range(1, self.max_iterations):
            # Generate feedback based on quality gaps
            feedback = self.evaluator.get_feedback(quality)

            current_email, tokens = self.refine(scenario, current_email, feedback)
            result.total_tokens += tokens

            if verbose:
                print(f"  [Iteration {i + 1}] Refined ({tokens} tokens)")

            # Evaluate refined email
            quality = self.evaluator.evaluate(current_email, scenario)

            detail = IterationDetail(
                iteration=i + 1,
                email=current_email,
                quality=quality,
                tokens_used=tokens,
            )

            # Check if now good enough
            if self.evaluator.meets_threshold(quality, self.quality_threshold):
                detail.stopped = True
                detail.stop_reason = "quality_threshold_met"
                result.iteration_details.append(detail)

                result.iterations = i + 1
                result.final_email = current_email
                result.final_quality = quality.weighted_score
                result.stopped_early = True
                result.stop_reason = (
                    f"Q={quality.weighted_score:.2f} >= Q_min={self.quality_threshold}"
                )

                if verbose:
                    print(
                        f"  ✓ Quality {quality.weighted_score:.2f} >= {self.quality_threshold} - STOPPING"
                    )

                return result

            result.iteration_details.append(detail)

            if verbose:
                print(
                    f"  Quality: {quality.weighted_score:.2f} < {self.quality_threshold}, gaps: {quality.gaps}"
                )

        # Reached max iterations without meeting threshold
        result.iterations = self.max_iterations
        result.final_email = current_email
        result.final_quality = quality.weighted_score
        result.stopped_early = False
        result.stop_reason = f"max_iterations_reached (Q={quality.weighted_score:.2f})"

        if verbose:
            print(f"  ⚠ Max iterations reached, final Q={quality.weighted_score:.2f}")

        return result
