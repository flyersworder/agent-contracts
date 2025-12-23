"""Agent definitions for the code review pipeline.

This module defines the Coder and Reviewer agents, along with their
prompts and configurations.

The agents are designed to work in an iterative loop:
    Coder → Reviewer → Coder → Reviewer → ... → Approval or Max Iterations
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for an agent.

    Attributes:
        name: Agent name
        model: Model identifier (e.g., "gemini-2.0-flash")
        tokens: Token budget for this agent
        iterations: Maximum iterations (LLM calls)
        temperature: Sampling temperature
    """

    name: str
    model: str = "gemini-2.0-flash"
    tokens: int = 30000
    iterations: int = 10
    temperature: float = 0.2


# Default configurations by difficulty
CODER_CONFIGS = {
    "easy": AgentConfig(
        name="Coder",
        tokens=20000,
        iterations=8,
    ),
    "medium": AgentConfig(
        name="Coder",
        tokens=30000,
        iterations=10,
    ),
    "hard": AgentConfig(
        name="Coder",
        tokens=40000,
        iterations=12,
    ),
}

REVIEWER_CONFIGS = {
    "easy": AgentConfig(
        name="Reviewer",
        tokens=5000,
        iterations=3,
    ),
    "medium": AgentConfig(
        name="Reviewer",
        tokens=8000,
        iterations=4,
    ),
    "hard": AgentConfig(
        name="Reviewer",
        tokens=10000,
        iterations=5,
    ),
}


def get_coder_instruction(problem_description: str, budget_aware: bool = False) -> str:
    """Generate instruction for the Coder agent.

    Args:
        problem_description: The coding problem to solve
        budget_aware: Whether to include budget awareness in prompt

    Returns:
        Formatted instruction string
    """
    base_instruction = f"""You are an expert Python programmer solving competitive programming problems.

## Problem

{problem_description}

## Your Task

Write Python code that solves this problem. Your code should:
1. Read input from stdin using input()
2. Process the input according to the problem requirements
3. Print the output to stdout using print()

## Important Guidelines

- Write complete, runnable Python code
- Handle all edge cases mentioned in the problem
- Optimize for correctness first, then efficiency
- Do NOT include any explanation - return ONLY the Python code
- Do NOT wrap the code in markdown code blocks

## If You Receive Feedback

If the Reviewer provides feedback about test failures:
1. Carefully analyze the failing test case
2. Identify the bug or edge case you missed
3. Fix your code and return the corrected version
4. Return ONLY the complete, fixed Python code
"""

    if budget_aware:
        budget_section = """
## Resource Awareness

You have limited computational resources. Be efficient:
- Think carefully before writing to minimize iterations
- Aim to solve the problem correctly in as few attempts as possible
- Each revision costs resources - make each one count
"""
        return base_instruction + budget_section

    return base_instruction


def get_reviewer_instruction(budget_aware: bool = False) -> str:
    """Generate instruction for the Reviewer agent.

    Args:
        budget_aware: Whether to include budget awareness in prompt

    Returns:
        Formatted instruction string
    """
    base_instruction = """You are a code reviewer that tests Python solutions against test cases.

## Your Role

1. Execute the provided code against test cases using the test_code tool
2. Analyze the results
3. Either APPROVE the code or provide feedback for revision

## Decision Process

Call the test_code tool first, then:

**If ALL tests pass:**
- Respond with: "APPROVE: All tests passed. The solution is correct."

**If ANY test fails:**
- Respond with: "REVISE: [Brief description of the issue]"
- Include the failing test case details
- Suggest what the Coder should fix

## Important

- Always call test_code BEFORE making your decision
- Be concise - the Coder needs clear, actionable feedback
- Focus on the specific failure, not general improvements
"""

    if budget_aware:
        budget_section = """
## Resource Awareness

The team has limited resources. Be efficient:
- Provide specific, actionable feedback to minimize iterations
- If multiple tests fail, focus on the most informative failure
- Help the Coder fix issues quickly to conserve resources
"""
        return base_instruction + budget_section

    return base_instruction


def create_test_tool(execute_callback: Any) -> dict[str, Any]:
    """Create a test_code tool definition for the Reviewer.

    The tool executes the current code from session state against test cases.

    Args:
        execute_callback: Function to call for code execution

    Returns:
        Tool definition dictionary
    """
    return {
        "name": "test_code",
        "description": "Execute the current code against the problem's test cases. Returns PASS if all tests pass, or details about failing tests.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "callback": execute_callback,
    }


@dataclass
class PipelineConfig:
    """Configuration for the code review pipeline.

    Attributes:
        parent_tokens: Total token budget for the pipeline
        parent_iterations: Maximum total iterations across all agents
        coder_tokens: Token budget for Coder agent
        coder_iterations: Max iterations for Coder
        reviewer_tokens: Token budget for Reviewer agent
        reviewer_iterations: Max iterations for Reviewer
        max_review_rounds: Maximum Coder→Reviewer rounds
        model: Model to use for all agents
        reserve_ratio: Fraction of budget to reserve for coordination
    """

    parent_tokens: int = 50000
    parent_iterations: int = 20
    coder_tokens: int = 30000
    coder_iterations: int = 10
    reviewer_tokens: int = 10000
    reviewer_iterations: int = 5
    max_review_rounds: int = 5
    model: str = "gemini-2.0-flash"
    reserve_ratio: float = 0.1

    @classmethod
    def for_difficulty(cls, difficulty: str) -> "PipelineConfig":
        """Create a config appropriate for the given difficulty.

        Args:
            difficulty: "easy", "medium", or "hard"

        Returns:
            PipelineConfig with appropriate budgets
        """
        configs = {
            "easy": cls(
                parent_tokens=35000,
                parent_iterations=15,
                coder_tokens=20000,
                coder_iterations=8,
                reviewer_tokens=5000,
                reviewer_iterations=3,
                max_review_rounds=4,
            ),
            "medium": cls(
                parent_tokens=50000,
                parent_iterations=20,
                coder_tokens=30000,
                coder_iterations=10,
                reviewer_tokens=10000,
                reviewer_iterations=5,
                max_review_rounds=5,
            ),
            "hard": cls(
                parent_tokens=70000,
                parent_iterations=25,
                coder_tokens=40000,
                coder_iterations=12,
                reviewer_tokens=15000,
                reviewer_iterations=6,
                max_review_rounds=6,
            ),
        }
        return configs.get(difficulty, configs["medium"])

    def validate_conservation(self) -> bool:
        """Check if child budgets respect conservation laws.

        Returns:
            True if coder + reviewer tokens <= parent tokens (minus reserve)
        """
        available = self.parent_tokens * (1 - self.reserve_ratio)
        return self.coder_tokens + self.reviewer_tokens <= available
