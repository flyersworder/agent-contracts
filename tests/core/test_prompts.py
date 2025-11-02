"""Unit tests for budget-aware prompt generation."""

from datetime import datetime, timedelta

from agent_contracts.core import (
    Contract,
    ContractMode,
    ResourceConstraints,
    ResourceUsage,
    TemporalConstraints,
)
from agent_contracts.core.prompts import (
    estimate_prompt_tokens,
    generate_adaptive_instruction,
    generate_budget_prompt,
)


class TestGenerateBudgetPrompt:
    """Tests for generate_budget_prompt function."""

    def test_economical_mode_prompt(self) -> None:
        """Test prompt generation for ECONOMICAL mode."""
        contract = Contract(
            id="eco-test",
            name="Economical Task",
            mode=ContractMode.ECONOMICAL,
            resources=ResourceConstraints(tokens=5000, web_searches=3),
        )

        prompt = generate_budget_prompt(contract, "Research electric vehicles")

        # Check mode-specific content
        assert "ECONOMICAL" in prompt
        assert "💰" in prompt
        assert "cost" in prompt.lower() or "resource efficiency" in prompt.lower()
        assert "Research electric vehicles" in prompt

        # Check resource budget section
        assert "5,000" in prompt
        assert "3" in prompt

        # Check strategic guidance
        assert "Minimize" in prompt or "minimize" in prompt

    def test_urgent_mode_prompt(self) -> None:
        """Test prompt generation for URGENT mode."""
        contract = Contract(
            id="urgent-test",
            name="Urgent Task",
            mode=ContractMode.URGENT,
            resources=ResourceConstraints(tokens=10000, api_calls=20),
        )

        prompt = generate_budget_prompt(contract, "Analyze data quickly")

        # Check mode-specific content
        assert "URGENT" in prompt
        assert "⚡" in prompt
        assert "speed" in prompt.lower()
        assert "Analyze data quickly" in prompt

        # Check strategic guidance for urgent mode
        assert "parallel" in prompt.lower() or "speed" in prompt.lower()

    def test_balanced_mode_prompt(self) -> None:
        """Test prompt generation for BALANCED mode (default)."""
        contract = Contract(
            id="balanced-test",
            name="Balanced Task",
            # mode defaults to BALANCED
            resources=ResourceConstraints(tokens=8000),
        )

        prompt = generate_budget_prompt(contract, "Standard analysis")

        # Check mode-specific content
        assert "BALANCED" in prompt
        assert "⚖️" in prompt
        assert "balance" in prompt.lower()

    def test_prompt_with_current_usage(self) -> None:
        """Test prompt generation with current resource usage."""
        contract = Contract(
            id="usage-test",
            name="Task with Usage",
            mode=ContractMode.ECONOMICAL,
            resources=ResourceConstraints(tokens=10000, api_calls=10),
        )

        usage = ResourceUsage(tokens=7000, api_calls=7)
        prompt = generate_budget_prompt(contract, "Continue task", usage)

        # Check that remaining resources are shown
        assert "3,000 remaining" in prompt
        assert "70% used" in prompt
        assert "3 remaining" in prompt

        # Check adaptive guidance for high utilization
        assert "conservation" in prompt.lower() or "low budget" in prompt.lower()

    def test_prompt_with_temporal_constraints(self) -> None:
        """Test prompt generation with temporal constraints."""
        deadline = datetime(2025, 12, 31, 23, 59, 59)
        contract = Contract(
            id="temporal-test",
            name="Task with Deadline",
            resources=ResourceConstraints(tokens=5000),
            temporal=TemporalConstraints(deadline=deadline, max_duration=timedelta(hours=2)),
        )

        prompt = generate_budget_prompt(contract, "Time-sensitive task")

        # Check temporal section exists
        assert "Time Constraints" in prompt or "Deadline" in prompt
        assert "2025-12-31" in prompt
        assert "2.0 hours" in prompt

    def test_prompt_with_cost_constraint(self) -> None:
        """Test prompt generation with cost constraints."""
        contract = Contract(
            id="cost-test",
            name="Cost-Constrained Task",
            resources=ResourceConstraints(cost_usd=0.05, tokens=5000),
        )

        prompt = generate_budget_prompt(contract, "Budget-limited task")

        # Check cost is displayed
        assert "$0.0500" in prompt or "$0.05" in prompt
        assert "Cost:" in prompt

    def test_prompt_without_constraints(self) -> None:
        """Test prompt generation with minimal constraints."""
        contract = Contract(
            id="minimal-test",
            name="Minimal Task",
            # No resources or temporal constraints
        )

        prompt = generate_budget_prompt(contract, "Unconstrained task")

        # Should still have basic structure
        assert "Unconstrained task" in prompt
        assert "BALANCED" in prompt  # Default mode
        assert "operating under a formal Agent Contract" in prompt


class TestGenerateAdaptiveInstruction:
    """Tests for generate_adaptive_instruction function."""

    def test_low_utilization_balanced(self) -> None:
        """Test adaptive instruction at low utilization (< 30%) in BALANCED mode."""
        instruction = generate_adaptive_instruction(0.2, ContractMode.BALANCED)

        assert "ample" in instruction.lower() or "quality" in instruction.lower()
        assert "thorough" in instruction.lower() or "prioritize quality" in instruction.lower()

    def test_low_utilization_economical(self) -> None:
        """Test adaptive instruction at low utilization in ECONOMICAL mode."""
        instruction = generate_adaptive_instruction(0.15, ContractMode.ECONOMICAL)

        # Even with low utilization, ECONOMICAL mode still emphasizes efficiency
        assert "efficiency" in instruction.lower() or "prioritize" in instruction.lower()

    def test_moderate_utilization(self) -> None:
        """Test adaptive instruction at moderate utilization (30-70%)."""
        instruction = generate_adaptive_instruction(0.5, ContractMode.BALANCED)

        assert "moderate" in instruction.lower() or "balance" in instruction.lower()

    def test_high_utilization_balanced(self) -> None:
        """Test adaptive instruction at high utilization (> 70%) in BALANCED mode."""
        instruction = generate_adaptive_instruction(0.85, ContractMode.BALANCED)

        assert "conservation" in instruction.lower() or "minimize" in instruction.lower()
        assert "partial" in instruction.lower() or "parametric" in instruction.lower()

    def test_high_utilization_urgent(self) -> None:
        """Test adaptive instruction at high utilization in URGENT mode."""
        instruction = generate_adaptive_instruction(0.90, ContractMode.URGENT)

        # URGENT mode prioritizes speed even when budget is low
        assert "speed" in instruction.lower() or "core task" in instruction.lower()

    def test_edge_case_zero_utilization(self) -> None:
        """Test adaptive instruction at 0% utilization."""
        instruction = generate_adaptive_instruction(0.0, ContractMode.BALANCED)

        # Should give guidance for ample budget
        assert len(instruction) > 0
        assert "ample" in instruction.lower() or "quality" in instruction.lower()

    def test_edge_case_full_utilization(self) -> None:
        """Test adaptive instruction at 100% utilization."""
        instruction = generate_adaptive_instruction(1.0, ContractMode.ECONOMICAL)

        # Should give conservation guidance
        assert "conservation" in instruction.lower() or "minimize" in instruction.lower()


class TestEstimatePromptTokens:
    """Tests for estimate_prompt_tokens function."""

    def test_empty_string(self) -> None:
        """Test token estimation for empty string."""
        tokens = estimate_prompt_tokens("")
        assert tokens == 0

    def test_short_string(self) -> None:
        """Test token estimation for short string."""
        tokens = estimate_prompt_tokens("Hello world!")
        # ~4 chars per token, so 12 chars = ~3 tokens
        assert tokens == 3

    def test_long_string(self) -> None:
        """Test token estimation for longer string."""
        text = "This is a longer test string with multiple words to estimate tokens."
        tokens = estimate_prompt_tokens(text)

        # Should be roughly len(text) / 4
        expected = len(text) // 4
        assert tokens == expected

    def test_multiline_string(self) -> None:
        """Test token estimation for multi-line string."""
        text = """This is a multi-line string.
        It has several lines.
        Each line should be counted."""

        tokens = estimate_prompt_tokens(text)
        assert tokens > 0
        assert tokens == len(text) // 4


class TestIntegration:
    """Integration tests for prompt generation workflow."""

    def test_full_workflow_economical(self) -> None:
        """Test complete workflow for ECONOMICAL mode with budget tracking."""
        # Create contract
        contract = Contract(
            id="research-eco",
            name="Research Report",
            mode=ContractMode.ECONOMICAL,
            resources=ResourceConstraints(tokens=5000, web_searches=3, cost_usd=0.02),
            temporal=TemporalConstraints(max_duration=timedelta(hours=1)),
        )

        # Initial prompt (no usage)
        initial_prompt = generate_budget_prompt(contract, "Research AI trends")
        assert "5,000 total (5,000 remaining, 0% used)" in initial_prompt
        assert "ECONOMICAL" in initial_prompt

        # Simulate some usage
        usage = ResourceUsage(tokens=2000, web_searches=1, cost_usd=0.008)

        # Updated prompt with usage
        updated_prompt = generate_budget_prompt(contract, "Research AI trends", usage)
        assert "3,000 remaining" in updated_prompt
        assert "40% used" in updated_prompt

        # Adaptive instruction at 40% usage
        instruction = generate_adaptive_instruction(0.4, ContractMode.ECONOMICAL)
        assert len(instruction) > 0

    def test_full_workflow_urgent_high_usage(self) -> None:
        """Test complete workflow for URGENT mode under budget pressure."""
        contract = Contract(
            id="urgent-analysis",
            name="Urgent Analysis",
            mode=ContractMode.URGENT,
            resources=ResourceConstraints(tokens=10000, api_calls=20),
        )

        # Simulate high usage (80%)
        usage = ResourceUsage(tokens=8000, api_calls=16)
        prompt = generate_budget_prompt(contract, "Complete analysis", usage)

        # Should show high utilization
        assert "2,000 remaining" in prompt
        assert "80% used" in prompt

        # Should have conservation guidance
        assert "conservation" in prompt.lower() or "low budget" in prompt.lower()

        # Adaptive instruction should prioritize completion
        instruction = generate_adaptive_instruction(0.8, ContractMode.URGENT)
        assert "speed" in instruction.lower() or "core" in instruction.lower()
