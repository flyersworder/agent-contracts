"""Tests for fine-grained token control and reasoning effort."""

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.monitor import ResourceMonitor


class TestTokenModes:
    """Test token budget mode detection and validation."""

    def test_lumpsum_mode(self) -> None:
        """Test lumpsum mode detection."""
        constraints = ResourceConstraints(tokens=10000)
        assert constraints.token_mode == "lumpsum"

    def test_fine_grained_mode_both(self) -> None:
        """Test fine-grained mode with both reasoning and text tokens."""
        constraints = ResourceConstraints(reasoning_tokens=5000, text_tokens=2000)
        assert constraints.token_mode == "fine_grained"

    def test_fine_grained_mode_reasoning_only(self) -> None:
        """Test fine-grained mode with only reasoning tokens."""
        constraints = ResourceConstraints(reasoning_tokens=5000)
        assert constraints.token_mode == "fine_grained"

    def test_fine_grained_mode_text_only(self) -> None:
        """Test fine-grained mode with only text tokens."""
        constraints = ResourceConstraints(text_tokens=2000)
        assert constraints.token_mode == "fine_grained"

    def test_no_token_mode(self) -> None:
        """Test no token constraints."""
        constraints = ResourceConstraints(api_calls=10)
        assert constraints.token_mode == "none"

    def test_cannot_mix_modes(self) -> None:
        """Test that lumpsum and fine-grained modes cannot be mixed."""
        with pytest.raises(ValueError, match="Cannot mix token budget modes"):
            ResourceConstraints(tokens=10000, reasoning_tokens=5000)

        with pytest.raises(ValueError, match="Cannot mix token budget modes"):
            ResourceConstraints(tokens=10000, text_tokens=2000)

        with pytest.raises(ValueError, match="Cannot mix token budget modes"):
            ResourceConstraints(tokens=10000, reasoning_tokens=5000, text_tokens=2000)


class TestReasoningEffort:
    """Test reasoning effort validation and auto-selection."""

    def test_valid_reasoning_efforts(self) -> None:
        """Test that valid reasoning efforts are accepted."""
        for effort in ["low", "medium", "high"]:
            constraints = ResourceConstraints(reasoning_tokens=5000, reasoning_effort=effort)
            assert constraints.reasoning_effort == effort

    def test_invalid_reasoning_effort(self) -> None:
        """Test that invalid reasoning efforts are rejected."""
        with pytest.raises(ValueError, match="reasoning_effort must be one of"):
            ResourceConstraints(reasoning_tokens=5000, reasoning_effort="extreme")

        with pytest.raises(ValueError, match="reasoning_effort must be one of"):
            ResourceConstraints(reasoning_tokens=5000, reasoning_effort="MEDIUM")

    def test_reasoning_effort_without_budget(self) -> None:
        """Test that reasoning_effort can be specified without reasoning_tokens."""
        # This should work - effort is specified but budget is unlimited
        constraints = ResourceConstraints(text_tokens=1000, reasoning_effort="high")
        assert constraints.reasoning_effort == "high"


class TestReasoningCompatibility:
    """Test reasoning effort and budget compatibility validation."""

    def test_high_effort_requires_sufficient_tokens(self) -> None:
        """Test that high effort requires ≥2000 tokens."""
        # Valid: high effort with sufficient budget
        constraints = ResourceConstraints(reasoning_tokens=2000, reasoning_effort="high")
        assert constraints.reasoning_effort == "high"

        constraints = ResourceConstraints(reasoning_tokens=5000, reasoning_effort="high")
        assert constraints.reasoning_effort == "high"

        # Invalid: high effort with insufficient budget
        with pytest.raises(ValueError, match="reasoning_effort='high' typically requires"):
            ResourceConstraints(reasoning_tokens=1999, reasoning_effort="high")

        with pytest.raises(ValueError, match="reasoning_effort='high' typically requires"):
            ResourceConstraints(reasoning_tokens=100, reasoning_effort="high")

    def test_medium_effort_requires_sufficient_tokens(self) -> None:
        """Test that medium effort requires ≥500 tokens."""
        # Valid: medium effort with sufficient budget
        constraints = ResourceConstraints(reasoning_tokens=500, reasoning_effort="medium")
        assert constraints.reasoning_effort == "medium"

        constraints = ResourceConstraints(reasoning_tokens=1000, reasoning_effort="medium")
        assert constraints.reasoning_effort == "medium"

        # Invalid: medium effort with insufficient budget
        with pytest.raises(ValueError, match="reasoning_effort='medium' typically requires"):
            ResourceConstraints(reasoning_tokens=499, reasoning_effort="medium")

        with pytest.raises(ValueError, match="reasoning_effort='medium' typically requires"):
            ResourceConstraints(reasoning_tokens=100, reasoning_effort="medium")

    def test_low_effort_works_with_any_budget(self) -> None:
        """Test that low effort works with any budget."""
        # Low effort should work even with very small budgets
        constraints = ResourceConstraints(reasoning_tokens=100, reasoning_effort="low")
        assert constraints.reasoning_effort == "low"

        constraints = ResourceConstraints(reasoning_tokens=10, reasoning_effort="low")
        assert constraints.reasoning_effort == "low"


class TestRecommendedEffort:
    """Test auto-selection of reasoning effort based on budget."""

    def test_high_effort_recommended_for_large_budget(self) -> None:
        """Test that high effort is recommended for ≥2000 tokens."""
        constraints = ResourceConstraints(reasoning_tokens=2000)
        assert constraints.recommended_reasoning_effort == "high"

        constraints = ResourceConstraints(reasoning_tokens=5000)
        assert constraints.recommended_reasoning_effort == "high"

    def test_medium_effort_recommended_for_moderate_budget(self) -> None:
        """Test that medium effort is recommended for 500-1999 tokens."""
        constraints = ResourceConstraints(reasoning_tokens=500)
        assert constraints.recommended_reasoning_effort == "medium"

        constraints = ResourceConstraints(reasoning_tokens=1000)
        assert constraints.recommended_reasoning_effort == "medium"

        constraints = ResourceConstraints(reasoning_tokens=1999)
        assert constraints.recommended_reasoning_effort == "medium"

    def test_low_effort_recommended_for_small_budget(self) -> None:
        """Test that low effort is recommended for <500 tokens."""
        constraints = ResourceConstraints(reasoning_tokens=499)
        assert constraints.recommended_reasoning_effort == "low"

        constraints = ResourceConstraints(reasoning_tokens=100)
        assert constraints.recommended_reasoning_effort == "low"

        constraints = ResourceConstraints(reasoning_tokens=10)
        assert constraints.recommended_reasoning_effort == "low"

    def test_no_recommendation_without_budget(self) -> None:
        """Test that no effort is recommended without reasoning_tokens."""
        constraints = ResourceConstraints(tokens=10000)
        assert constraints.recommended_reasoning_effort is None

        constraints = ResourceConstraints(text_tokens=1000)
        assert constraints.recommended_reasoning_effort is None

        constraints = ResourceConstraints(api_calls=10)
        assert constraints.recommended_reasoning_effort is None


class TestModeAwareMonitoring:
    """Test that ResourceMonitor validates based on active mode."""

    def test_lumpsum_mode_validation(self) -> None:
        """Test that lumpsum mode checks total tokens only."""
        constraints = ResourceConstraints(tokens=1000)
        monitor = ResourceMonitor(constraints)

        # Use 500 tokens - within budget
        monitor.usage.add_tokens(count=500)
        violations = monitor.check_constraints()
        assert len(violations) == 0

        # Use 600 more tokens (total 1100) - exceeds budget
        monitor.usage.add_tokens(count=600)
        violations = monitor.check_constraints()
        assert len(violations) == 1
        assert violations[0].resource == "tokens"
        assert violations[0].limit == 1000
        assert violations[0].actual == 1100

    def test_fine_grained_mode_validation_both(self) -> None:
        """Test that fine-grained mode checks reasoning and text separately."""
        constraints = ResourceConstraints(reasoning_tokens=500, text_tokens=200)
        monitor = ResourceMonitor(constraints)

        # Within both budgets
        monitor.usage.add_tokens(count=0, reasoning=400, text=150)
        violations = monitor.check_constraints()
        assert len(violations) == 0

        # Exceed reasoning budget only
        monitor.usage.add_tokens(count=0, reasoning=200, text=0)  # Total reasoning: 600
        violations = monitor.check_constraints()
        assert len(violations) == 1
        assert violations[0].resource == "reasoning_tokens"
        assert violations[0].limit == 500
        assert violations[0].actual == 600

        # Also exceed text budget
        monitor.usage.add_tokens(count=0, reasoning=0, text=100)  # Total text: 250
        violations = monitor.check_constraints()
        assert len(violations) == 2
        assert any(v.resource == "reasoning_tokens" for v in violations)
        assert any(v.resource == "text_tokens" for v in violations)

    def test_fine_grained_mode_reasoning_only(self) -> None:
        """Test fine-grained mode with only reasoning_tokens constraint."""
        constraints = ResourceConstraints(reasoning_tokens=500)  # text_tokens unlimited
        monitor = ResourceMonitor(constraints)

        # Within reasoning budget, unlimited text
        monitor.usage.add_tokens(count=0, reasoning=400, text=10000)
        violations = monitor.check_constraints()
        assert len(violations) == 0  # Only reasoning budget enforced

        # Exceed reasoning budget
        monitor.usage.add_tokens(count=0, reasoning=200, text=0)  # Total reasoning: 600
        violations = monitor.check_constraints()
        assert len(violations) == 1
        assert violations[0].resource == "reasoning_tokens"

    def test_fine_grained_mode_text_only(self) -> None:
        """Test fine-grained mode with only text_tokens constraint."""
        constraints = ResourceConstraints(text_tokens=200)  # reasoning_tokens unlimited
        monitor = ResourceMonitor(constraints)

        # Unlimited reasoning, within text budget
        monitor.usage.add_tokens(count=0, reasoning=10000, text=150)
        violations = monitor.check_constraints()
        assert len(violations) == 0  # Only text budget enforced

        # Exceed text budget
        monitor.usage.add_tokens(count=0, reasoning=0, text=100)  # Total text: 250
        violations = monitor.check_constraints()
        assert len(violations) == 1
        assert violations[0].resource == "text_tokens"

    def test_no_token_mode_skips_checks(self) -> None:
        """Test that no token mode skips all token checks."""
        constraints = ResourceConstraints(api_calls=10)  # No token constraints
        monitor = ResourceMonitor(constraints)

        # Use any amount of tokens - no violations
        monitor.usage.add_tokens(count=1000000)
        monitor.usage.add_tokens(count=0, reasoning=500000, text=500000)
        violations = monitor.check_constraints()
        assert len(violations) == 0  # No token constraints to violate


class TestModeAwareUsagePercentage:
    """Test that usage percentages are mode-aware."""

    def test_lumpsum_mode_shows_total_only(self) -> None:
        """Test that lumpsum mode shows only total token percentage."""
        constraints = ResourceConstraints(tokens=1000, api_calls=10)
        monitor = ResourceMonitor(constraints)
        monitor.usage.add_tokens(count=500)
        monitor.usage.add_api_call()

        percentages = monitor.get_usage_percentage()
        assert "tokens" in percentages
        assert percentages["tokens"] == 50.0
        assert "reasoning_tokens" not in percentages
        assert "text_tokens" not in percentages
        assert "api_calls" in percentages

    def test_fine_grained_mode_shows_breakdown(self) -> None:
        """Test that fine-grained mode shows reasoning and text percentages."""
        constraints = ResourceConstraints(reasoning_tokens=500, text_tokens=200)
        monitor = ResourceMonitor(constraints)
        monitor.usage.add_tokens(count=0, reasoning=250, text=100)

        percentages = monitor.get_usage_percentage()
        assert "tokens" not in percentages  # Not shown in fine-grained mode
        assert "reasoning_tokens" in percentages
        assert percentages["reasoning_tokens"] == 50.0
        assert "text_tokens" in percentages
        assert percentages["text_tokens"] == 50.0

    def test_fine_grained_partial_shows_relevant_only(self) -> None:
        """Test that partial fine-grained mode shows only relevant percentages."""
        # Only reasoning tokens constrained
        constraints = ResourceConstraints(reasoning_tokens=500)
        monitor = ResourceMonitor(constraints)
        monitor.usage.add_tokens(count=0, reasoning=250, text=1000)

        percentages = monitor.get_usage_percentage()
        assert "reasoning_tokens" in percentages
        assert percentages["reasoning_tokens"] == 50.0
        assert "text_tokens" not in percentages  # Not constrained
        assert "tokens" not in percentages

    def test_no_token_mode_shows_no_token_percentages(self) -> None:
        """Test that no token mode shows no token percentages."""
        constraints = ResourceConstraints(api_calls=10)
        monitor = ResourceMonitor(constraints)
        monitor.usage.add_tokens(count=1000)
        monitor.usage.add_api_call()

        percentages = monitor.get_usage_percentage()
        assert "tokens" not in percentages
        assert "reasoning_tokens" not in percentages
        assert "text_tokens" not in percentages
        assert "api_calls" in percentages


class TestIntegrationWithContract:
    """Test integration of fine-grained tokens with full contracts."""

    def test_contract_with_fine_grained_tokens(self) -> None:
        """Test creating a contract with fine-grained token control."""
        contract = Contract(
            id="test",
            name="Test Contract",
            resources=ResourceConstraints(
                reasoning_tokens=5000,
                text_tokens=2000,
                reasoning_effort="high",
                api_calls=10,
            ),
        )

        assert contract.resources.token_mode == "fine_grained"
        assert contract.resources.reasoning_tokens == 5000
        assert contract.resources.text_tokens == 2000
        assert contract.resources.reasoning_effort == "high"

    def test_contract_with_auto_effort(self) -> None:
        """Test contract with auto-selected reasoning effort."""
        contract = Contract(
            id="test",
            name="Test Contract",
            resources=ResourceConstraints(
                reasoning_tokens=1000,  # Should auto-select "medium"
                text_tokens=500,
            ),
        )

        assert contract.resources.recommended_reasoning_effort == "medium"

    def test_contract_validation_catches_incompatibility(self) -> None:
        """Test that contract creation fails on effort/budget incompatibility."""
        with pytest.raises(ValueError, match="reasoning_effort='high' typically requires"):
            Contract(
                id="test",
                name="Test Contract",
                resources=ResourceConstraints(
                    reasoning_tokens=100,  # Too small for high effort
                    reasoning_effort="high",
                ),
            )
