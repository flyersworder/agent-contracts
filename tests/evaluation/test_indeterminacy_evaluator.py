"""Tests for indeterminacy-aware LLM-as-judge evaluator.

Tests the implementation of the rating indeterminacy framework from
NeurIPS 2025 paper "Validating LLM-as-a-Judge Systems under Rating Indeterminacy".
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from evaluation.indeterminacy_evaluator import (  # noqa: E402
    ACCURACY_RANGES,
    COHERENCE_RANGES,
    COMPLETENESS_RANGES,
    IndeterminacyAwareEvaluator,
    IndeterminacyAwareScore,
    MultiLabelScore,
    ResponseSet,
    decision_consistency,
    mse_srs_srs,
    prevalence_bias,
)


class TestResponseSet:
    """Test ResponseSet dataclass."""

    def test_create_response_set(self) -> None:
        """Test creating a ResponseSet."""
        rs = ResponseSet(selected_options={"C", "D"}, dimension="accuracy")

        assert rs.selected_options == {"C", "D"}
        assert rs.dimension == "accuracy"
        assert rs.confidence == 1.0

    def test_is_determinate_single_option(self) -> None:
        """Test determinacy with single option."""
        rs = ResponseSet({"C"}, "accuracy")
        assert rs.is_determinate is True

    def test_is_determinate_multiple_options(self) -> None:
        """Test determinacy with multiple options."""
        rs = ResponseSet({"C", "D"}, "accuracy")
        assert rs.is_determinate is False

    def test_indeterminacy_level(self) -> None:
        """Test indeterminacy level calculation."""
        rs1 = ResponseSet({"C"}, "accuracy")
        rs2 = ResponseSet({"C", "D"}, "accuracy")
        rs3 = ResponseSet({"B", "C", "D"}, "accuracy")

        assert rs1.indeterminacy_level == 1.0
        assert rs2.indeterminacy_level == 2.0
        assert rs3.indeterminacy_level == 3.0

    def test_to_multilabel_vector(self) -> None:
        """Test conversion to multi-label vector."""
        rs = ResponseSet({"B", "C"}, "accuracy")
        options = ["A", "B", "C", "D"]

        vector = rs.to_multilabel_vector(options)

        expected = np.array([0.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(vector, expected)

    def test_to_multilabel_vector_all_selected(self) -> None:
        """Test multi-label vector with all options selected."""
        rs = ResponseSet({"A", "B", "C", "D"}, "accuracy")
        options = ["A", "B", "C", "D"]

        vector = rs.to_multilabel_vector(options)

        expected = np.array([1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(vector, expected)


class TestMultiLabelScore:
    """Test MultiLabelScore dataclass."""

    def test_create_multilabel_score(self) -> None:
        """Test creating a MultiLabelScore."""
        omega = np.array([0.0, 0.33, 0.67, 1.0])
        score = MultiLabelScore(
            dimension="accuracy",
            omega=omega,
            point_estimate=7.5,
            indeterminacy=1.5,
        )

        assert score.dimension == "accuracy"
        np.testing.assert_array_equal(score.omega, omega)
        assert score.point_estimate == 7.5
        assert score.indeterminacy == 1.5

    def test_is_ambiguous_high(self) -> None:
        """Test ambiguity detection with high indeterminacy."""
        score = MultiLabelScore(
            dimension="accuracy",
            omega=np.array([0.0, 0.5, 0.5, 0.5]),
            point_estimate=7.0,
            indeterminacy=1.5,  # > 1.2
        )
        assert score.is_ambiguous is True

    def test_is_ambiguous_low(self) -> None:
        """Test ambiguity detection with low indeterminacy."""
        score = MultiLabelScore(
            dimension="accuracy",
            omega=np.array([0.0, 0.0, 0.0, 1.0]),
            point_estimate=9.5,
            indeterminacy=1.0,  # < 1.2
        )
        assert score.is_ambiguous is False

    def test_to_hard_multilabel_default_tau(self) -> None:
        """Test hard multi-label conversion with default tau=0.5."""
        score = MultiLabelScore(
            dimension="accuracy",
            omega=np.array([0.0, 0.33, 0.67, 1.0]),
            point_estimate=7.5,
            indeterminacy=1.5,
        )

        hard = score.to_hard_multilabel()

        expected = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(hard, expected)

    def test_to_hard_multilabel_custom_tau(self) -> None:
        """Test hard multi-label with custom threshold."""
        score = MultiLabelScore(
            dimension="accuracy",
            omega=np.array([0.0, 0.33, 0.67, 1.0]),
            point_estimate=7.5,
            indeterminacy=1.5,
        )

        # Lower threshold includes more options
        hard = score.to_hard_multilabel(tau=0.3)

        expected = np.array([0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(hard, expected)


class TestIndeterminacyAwareScore:
    """Test IndeterminacyAwareScore dataclass."""

    def _create_sample_score(self) -> IndeterminacyAwareScore:
        """Create a sample score for testing."""
        accuracy = MultiLabelScore(
            dimension="accuracy",
            omega=np.array([0.0, 0.0, 0.33, 0.67]),
            point_estimate=8.5,
            indeterminacy=1.5,
        )
        completeness = MultiLabelScore(
            dimension="completeness",
            omega=np.array([0.0, 0.0, 0.67, 0.33]),
            point_estimate=7.5,
            indeterminacy=1.2,
        )
        coherence = MultiLabelScore(
            dimension="coherence",
            omega=np.array([0.0, 0.0, 0.0, 1.0]),
            point_estimate=9.5,
            indeterminacy=1.0,
        )
        return IndeterminacyAwareScore(
            accuracy=accuracy,
            completeness=completeness,
            coherence=coherence,
            total=85.0,
            explanation="Test explanation",
            judge_agreement=0.8,
        )

    def test_point_estimates(self) -> None:
        """Test getting point estimates."""
        score = self._create_sample_score()
        estimates = score.point_estimates

        assert estimates["accuracy"] == 8.5
        assert estimates["completeness"] == 7.5
        assert estimates["coherence"] == 9.5
        assert estimates["total"] == 85.0

    def test_indeterminacy_summary(self) -> None:
        """Test indeterminacy summary."""
        score = self._create_sample_score()
        summary = score.indeterminacy_summary

        assert summary["accuracy"] == 1.5
        assert summary["completeness"] == 1.2
        assert summary["coherence"] == 1.0
        assert summary["overall"] == pytest.approx((1.5 + 1.2 + 1.0) / 3)

    def test_omega_vectors(self) -> None:
        """Test getting omega vectors."""
        score = self._create_sample_score()
        omegas = score.omega_vectors

        np.testing.assert_array_equal(omegas["accuracy"], np.array([0.0, 0.0, 0.33, 0.67]))
        np.testing.assert_array_equal(omegas["completeness"], np.array([0.0, 0.0, 0.67, 0.33]))
        np.testing.assert_array_equal(omegas["coherence"], np.array([0.0, 0.0, 0.0, 1.0]))

    def test_is_dimension_ambiguous(self) -> None:
        """Test dimension-specific ambiguity check."""
        score = self._create_sample_score()

        assert score.is_dimension_ambiguous("accuracy") is True  # 1.5 > 1.2
        assert score.is_dimension_ambiguous("completeness") is False  # 1.2 not > 1.2
        assert score.is_dimension_ambiguous("coherence") is False  # 1.0 < 1.2


class TestRatingRanges:
    """Test rating range configurations."""

    def test_accuracy_ranges(self) -> None:
        """Test accuracy ranges are properly defined."""
        assert len(ACCURACY_RANGES) == 4
        assert "A" in ACCURACY_RANGES
        assert ACCURACY_RANGES["A"][0] == 0
        assert ACCURACY_RANGES["A"][1] == 3
        assert ACCURACY_RANGES["D"][0] == 9
        assert ACCURACY_RANGES["D"][1] == 10

    def test_completeness_ranges(self) -> None:
        """Test completeness ranges are properly defined."""
        assert len(COMPLETENESS_RANGES) == 4
        assert COMPLETENESS_RANGES["A"][0] == 0
        assert COMPLETENESS_RANGES["D"][1] == 10

    def test_coherence_ranges(self) -> None:
        """Test coherence ranges are properly defined."""
        assert len(COHERENCE_RANGES) == 4
        assert COHERENCE_RANGES["A"][0] == 0
        assert COHERENCE_RANGES["D"][1] == 10


class TestIndeterminacyAwareEvaluator:
    """Test IndeterminacyAwareEvaluator class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        evaluator = IndeterminacyAwareEvaluator()

        assert evaluator.judge_model == "gemini/gemini-2.0-flash"
        assert evaluator.num_judges == 3
        assert evaluator.use_hybrid_scoring is True
        assert evaluator.tau == 0.5

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        evaluator = IndeterminacyAwareEvaluator(
            judge_model="gpt-4",
            num_judges=5,
            use_hybrid_scoring=False,
            tau=0.7,
        )

        assert evaluator.judge_model == "gpt-4"
        assert evaluator.num_judges == 5
        assert evaluator.use_hybrid_scoring is False
        assert evaluator.tau == 0.7

    def test_extract_letters(self) -> None:
        """Test letter extraction from text."""
        evaluator = IndeterminacyAwareEvaluator()

        assert evaluator._extract_letters("CD") == {"C", "D"}
        assert evaluator._extract_letters("C, D") == {"C", "D"}
        assert evaluator._extract_letters("C D") == {"C", "D"}  # Space separated
        assert evaluator._extract_letters("BCD") == {"B", "C", "D"}
        assert evaluator._extract_letters("A") == {"A"}
        assert evaluator._extract_letters("") == {"C"}  # Default
        assert evaluator._extract_letters("XYZ") == {"C"}  # No valid letters
        # Note: "[C] and [D]" would match 'A' from 'and' - this is expected
        # since the LLM should use the strict format "CD" not prose

    def test_aggregate_response_sets(self) -> None:
        """Test response set aggregation."""
        evaluator = IndeterminacyAwareEvaluator()

        response_sets = [
            ResponseSet({"C", "D"}, "accuracy"),
            ResponseSet({"D"}, "accuracy"),
            ResponseSet({"C", "D"}, "accuracy"),
        ]

        score = evaluator._aggregate_response_sets(response_sets, "accuracy")

        # C selected 2/3 times, D selected 3/3 times
        assert score.omega[2] == pytest.approx(2 / 3)  # C
        assert score.omega[3] == pytest.approx(1.0)  # D
        assert score.indeterminacy == pytest.approx((2 + 1 + 2) / 3)

    def test_calculate_judge_agreement_perfect(self) -> None:
        """Test judge agreement calculation with perfect agreement."""
        evaluator = IndeterminacyAwareEvaluator()

        all_response_sets = {
            "accuracy": [
                ResponseSet({"C", "D"}, "accuracy"),
                ResponseSet({"C", "D"}, "accuracy"),
                ResponseSet({"C", "D"}, "accuracy"),
            ],
            "completeness": [
                ResponseSet({"D"}, "completeness"),
                ResponseSet({"D"}, "completeness"),
                ResponseSet({"D"}, "completeness"),
            ],
            "coherence": [
                ResponseSet({"C"}, "coherence"),
                ResponseSet({"C"}, "coherence"),
                ResponseSet({"C"}, "coherence"),
            ],
        }

        agreement = evaluator._calculate_judge_agreement(all_response_sets)
        assert agreement == 1.0

    def test_calculate_judge_agreement_partial(self) -> None:
        """Test judge agreement with partial agreement."""
        evaluator = IndeterminacyAwareEvaluator()

        all_response_sets = {
            "accuracy": [
                ResponseSet({"C", "D"}, "accuracy"),  # CD
                ResponseSet({"C"}, "accuracy"),  # C only
                ResponseSet({"D"}, "accuracy"),  # D only
            ],
            "completeness": [
                ResponseSet({"D"}, "completeness"),
                ResponseSet({"D"}, "completeness"),
                ResponseSet({"D"}, "completeness"),
            ],
            "coherence": [
                ResponseSet({"C"}, "coherence"),
                ResponseSet({"C"}, "coherence"),
                ResponseSet({"C"}, "coherence"),
            ],
        }

        agreement = evaluator._calculate_judge_agreement(all_response_sets)

        # Accuracy has imperfect agreement, others are perfect
        # Should be less than 1.0
        assert 0 < agreement < 1.0

    def test_parse_response_set_evaluation(self) -> None:
        """Test parsing LLM response."""
        evaluator = IndeterminacyAwareEvaluator()

        content = """Accuracy: CD
Completeness: D
Coherence: BCD

Explanation:
The answer is mostly accurate with some minor issues.
Multiple coherence levels could apply."""

        result = evaluator._parse_response_set_evaluation(content)

        assert result["accuracy"].selected_options == {"C", "D"}
        assert result["completeness"].selected_options == {"D"}
        assert result["coherence"].selected_options == {"B", "C", "D"}
        assert "mostly accurate" in result["explanation"]

    def test_parse_response_set_evaluation_defaults(self) -> None:
        """Test parsing with missing data defaults to C."""
        evaluator = IndeterminacyAwareEvaluator()

        content = """Accuracy:
Completeness: D
Coherence: """

        result = evaluator._parse_response_set_evaluation(content)

        # Empty responses default to {"C"}
        assert result["accuracy"].selected_options == {"C"}
        assert result["completeness"].selected_options == {"D"}
        assert result["coherence"].selected_options == {"C"}

    def test_rule_based_scores(self) -> None:
        """Test rule-based scoring."""
        evaluator = IndeterminacyAwareEvaluator()

        answer = """
        ## Technical Analysis

        The algorithm uses a novel approach with 3 key optimizations.
        Performance improved by 45% compared to the baseline method.

        1. First optimization reduces latency
        2. Second optimization improves throughput
        3. Third optimization handles edge cases

        This framework addresses the main challenge of scaling.
        The implementation uses standard architectural patterns.
        """

        scores = evaluator._calculate_rule_based_scores("Test question", answer)

        assert "accuracy" in scores
        assert "completeness" in scores
        assert "coherence" in scores
        assert all(0 <= s <= 10 for s in scores.values())


class TestMetrics:
    """Test metrics functions following CMU paper."""

    def _create_score(
        self,
        accuracy_omega: list[float],
        completeness_omega: list[float],
        coherence_omega: list[float],
    ) -> IndeterminacyAwareScore:
        """Helper to create scores for testing."""
        return IndeterminacyAwareScore(
            accuracy=MultiLabelScore("accuracy", np.array(accuracy_omega), 7.0, 1.0),
            completeness=MultiLabelScore("completeness", np.array(completeness_omega), 7.0, 1.0),
            coherence=MultiLabelScore("coherence", np.array(coherence_omega), 7.0, 1.0),
            total=70.0,
            explanation="Test",
        )

    def test_mse_srs_srs_identical(self) -> None:
        """Test MSE is 0 for identical scores."""
        score1 = self._create_score(
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        score2 = self._create_score(
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )

        mse = mse_srs_srs(score1, score2)
        assert mse == 0.0

    def test_mse_srs_srs_different(self) -> None:
        """Test MSE is positive for different scores."""
        score1 = self._create_score(
            [0.0, 0.0, 0.0, 1.0],  # Only D
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        score2 = self._create_score(
            [1.0, 0.0, 0.0, 0.0],  # Only A (opposite)
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )

        mse = mse_srs_srs(score1, score2)
        assert mse > 0.0

    def test_decision_consistency_perfect(self) -> None:
        """Test decision consistency with matching decisions."""
        score1 = self._create_score(
            [0.0, 0.0, 0.3, 0.7],  # C+D > 0.5
            [0.0, 0.0, 0.4, 0.6],
            [0.0, 0.0, 0.5, 0.5],
        )
        score2 = self._create_score(
            [0.0, 0.0, 0.2, 0.8],  # C+D > 0.5
            [0.0, 0.0, 0.6, 0.4],
            [0.0, 0.0, 0.6, 0.4],
        )

        consistency = decision_consistency(score1, score2)
        assert consistency == 1.0

    def test_decision_consistency_partial(self) -> None:
        """Test decision consistency with some disagreement."""
        score1 = self._create_score(
            [0.0, 0.0, 0.3, 0.7],  # C+D = 1.0 > 0.5 -> positive
            [0.0, 0.0, 0.4, 0.6],  # C+D = 1.0 > 0.5 -> positive
            [0.5, 0.5, 0.0, 0.0],  # C+D = 0.0 < 0.5 -> negative
        )
        score2 = self._create_score(
            [0.0, 0.0, 0.2, 0.8],  # C+D = 1.0 > 0.5 -> positive
            [0.6, 0.4, 0.0, 0.0],  # C+D = 0.0 < 0.5 -> negative (disagrees)
            [0.6, 0.4, 0.0, 0.0],  # C+D = 0.0 < 0.5 -> negative
        )

        consistency = decision_consistency(score1, score2)
        assert consistency == pytest.approx(2 / 3)  # 2 out of 3 match

    def test_prevalence_bias_no_bias(self) -> None:
        """Test prevalence bias with no systematic bias."""
        scores = [self._create_score([0, 0, 0.5, 0.5], [0, 0, 1, 0], [0, 0, 0, 1])]
        refs = [self._create_score([0, 0, 0.5, 0.5], [0, 0, 1, 0], [0, 0, 0, 1])]

        bias = prevalence_bias(scores, refs)
        assert bias == 0.0

    def test_prevalence_bias_positive(self) -> None:
        """Test prevalence bias when judge overestimates quality."""
        scores = [self._create_score([0, 0, 0.6, 0.4], [0, 0, 0.8, 0.2], [0, 0, 0.7, 0.3])]
        refs = [self._create_score([0.6, 0.4, 0, 0], [0.8, 0.2, 0, 0], [0.7, 0.3, 0, 0])]

        bias = prevalence_bias(scores, refs)
        assert bias > 0.0  # Judge sees more positives than reference


class TestIntegration:
    """Integration tests (require LLM API - skip in CI)."""

    @pytest.mark.skip(reason="Requires LLM API - run manually")
    def test_full_evaluation(self) -> None:
        """Test full evaluation flow with real LLM."""
        evaluator = IndeterminacyAwareEvaluator(
            judge_model="gemini/gemini-2.0-flash",
            num_judges=3,
        )

        question = "What are the benefits of using contracts for AI agents?"
        answer = """
        Agent contracts provide several key benefits:

        1. **Resource Governance**: Explicit limits on tokens, API calls, and costs
           prevent runaway resource consumption.

        2. **Temporal Boundaries**: Maximum duration constraints ensure agents
           complete tasks within acceptable timeframes.

        3. **Predictability**: Pre-defined constraints make agent behavior more
           predictable and auditable.

        4. **Composability**: Contracts enable safe multi-agent coordination
           through hierarchical resource sharing.

        These benefits are particularly valuable for enterprise deployments
        where compliance and cost control are critical.
        """

        score = evaluator.evaluate(question, answer)

        assert isinstance(score, IndeterminacyAwareScore)
        assert 0 <= score.total <= 100
        assert all(0 <= v <= 1 for v in score.accuracy.omega)
        assert score.indeterminacy_summary["overall"] > 0
