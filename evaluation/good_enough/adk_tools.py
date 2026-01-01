"""Email drafting tools for Google ADK agents.

These tools enable the agent to:
1. Evaluate email quality against criteria
2. Submit the final email (signal completion)

NOTE: The agent drafts the email DIRECTLY in its response using its own
LLM capabilities. We don't need a "draft_email" tool - the agent IS the
drafter. Tools are only for evaluation and submission.
"""

from typing import Any

from .email_indeterminacy_evaluator import EmailIndeterminacyEvaluator
from .scenarios import EmailScenario

# Global state for the current scenario (set before agent runs)
_current_scenario: EmailScenario | None = None
_submitted_email: str = ""
_evaluation_count: int = 0
_evaluator: EmailIndeterminacyEvaluator | None = None
_total_tokens_estimate: int = 0  # Rough estimate based on text processed


def set_scenario(scenario: EmailScenario, evaluator: EmailIndeterminacyEvaluator) -> None:
    """Set the current scenario for tool execution.

    Must be called before running the agent.

    Args:
        scenario: The email scenario to work on
        evaluator: The quality evaluator to use
    """
    global \
        _current_scenario, \
        _submitted_email, \
        _evaluation_count, \
        _evaluator, \
        _total_tokens_estimate
    _current_scenario = scenario
    _submitted_email = ""
    _evaluation_count = 0
    _evaluator = evaluator
    _total_tokens_estimate = 0


def get_execution_stats() -> dict[str, Any]:
    """Get execution statistics after agent completes.

    Returns:
        Dictionary with iterations, tokens, and final email
    """
    return {
        "iterations": _evaluation_count,  # Each evaluation = one iteration
        "total_tokens": _total_tokens_estimate,
        "final_email": _submitted_email,
    }


def evaluate_quality(email_text: str) -> dict[str, Any]:
    """Evaluate the quality of an email draft.

    Uses indeterminacy-aware evaluation to assess the email against
    quality criteria: purpose clarity, professional tone, completeness,
    and actionability.

    This tool returns NEUTRAL quality metrics. The agent must interpret
    the score based on its own contract/criteria to decide next steps.

    Args:
        email_text: The email text to evaluate. Include the full email
                   with subject line and body.

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - quality_score: Overall quality (0-1 scale)
        - gaps: List of criteria that need improvement
        - feedback: Specific improvement suggestions
        - details: Breakdown by dimension
    """
    global _evaluation_count, _total_tokens_estimate

    if _current_scenario is None:
        return {
            "status": "error",
            "error_message": "No scenario set. Cannot evaluate.",
        }

    if not email_text or len(email_text.strip()) < 20:
        return {
            "status": "error",
            "error_message": "Email text is too short or empty. Please provide a complete email.",
        }

    if _evaluator is None:
        return {
            "status": "error",
            "error_message": "No evaluator configured.",
        }

    _evaluation_count += 1
    # Rough token estimate: ~4 chars per token for English text
    _total_tokens_estimate += len(email_text) // 4

    try:
        # Evaluate the email
        score = _evaluator.evaluate(email_text, _current_scenario)

        # Get feedback for improvement
        feedback = _evaluator.get_feedback(score)

        # Return NEUTRAL metrics - no threshold judgment
        # The agent interprets the score based on its contract
        return {
            "status": "success",
            "quality_score": round(float(score.weighted_score), 3),
            "gaps": score.gaps,
            "feedback": feedback,
            "details": {
                "purpose_clarity": round(float(score.purpose_clarity.point_estimate), 1),
                "professional_tone": round(float(score.professional_tone.point_estimate), 1),
                "completeness": round(float(score.completeness.point_estimate), 1),
                "actionability": round(float(score.actionability.point_estimate), 1),
                "length_score": round(float(score.length_score), 2),
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to evaluate email: {e}",
        }


def submit_email(final_email: str) -> dict[str, Any]:
    """Submit the final email, signaling that the task is complete.

    Call this tool when you have determined the email is ready to send.
    This signals that you are done with the drafting process.

    IMPORTANT: Once you call this, the task is complete. Make sure the
    email meets the quality requirements before submitting.

    Args:
        final_email: The complete email text to submit, including subject
                    line and body.

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - message: Completion message
    """
    global _submitted_email

    if not final_email or len(final_email.strip()) < 20:
        return {
            "status": "error",
            "error_message": "Email text is too short or empty. Please provide a complete email.",
        }

    _submitted_email = final_email

    return {
        "status": "success",
        "message": "Email submitted successfully. Task complete.",
        "email_length": len(final_email),
    }


# Export tools as simple functions (ADK will wrap them automatically)
EMAIL_TOOLS = [evaluate_quality, submit_email]
