"""Tests for contract templates (Phase 2B)."""

from datetime import timedelta

import pytest

from agent_contracts.core.contract import Contract, ContractMode, ResourceConstraints
from agent_contracts.templates import (
    CodeReviewContract,
    CustomerSupportContract,
    DataAnalysisContract,
    ResearchContract,
    get_template,
)


class TestResearchContract:
    """Test ResearchContract template."""

    def test_create_default_research_contract(self) -> None:
        """Test creating research contract with defaults."""
        contract = ResearchContract.create(topic="AI Ethics")

        assert isinstance(contract, Contract)
        assert contract.name == "Research: AI Ethics"
        assert contract.mode == ContractMode.BALANCED
        assert contract.resources.tokens == 150_000
        assert contract.resources.web_searches == 20
        assert contract.resources.api_calls == 40
        assert contract.resources.cost_usd == 3.0

    def test_research_contract_urgent_mode(self) -> None:
        """Test research contract in URGENT mode."""
        contract = ResearchContract.create(
            topic="Quantum Computing", mode=ContractMode.URGENT
        )

        assert contract.mode == ContractMode.URGENT
        assert contract.resources.tokens == 200_000
        assert contract.resources.web_searches == 30
        assert contract.resources.api_calls == 60
        assert contract.resources.cost_usd == 4.0
        assert contract.temporal.max_duration == timedelta(minutes=30)

    def test_research_contract_economical_mode(self) -> None:
        """Test research contract in ECONOMICAL mode."""
        contract = ResearchContract.create(topic="ML", mode=ContractMode.ECONOMICAL)

        assert contract.mode == ContractMode.ECONOMICAL
        assert contract.resources.tokens == 80_000
        assert contract.resources.web_searches == 10
        assert contract.resources.api_calls == 25
        assert contract.resources.cost_usd == 1.5
        assert contract.temporal.max_duration == timedelta(hours=4)

    def test_research_contract_balanced_mode(self) -> None:
        """Test research contract in BALANCED mode."""
        contract = ResearchContract.create(topic="Test", mode=ContractMode.BALANCED)

        assert contract.mode == ContractMode.BALANCED
        assert contract.resources.tokens == 150_000
        assert contract.temporal.max_duration == timedelta(hours=2)

    def test_research_contract_custom_resources(self) -> None:
        """Test research contract with custom resource overrides."""
        contract = ResearchContract.create(
            topic="Custom Research",
            resources={"tokens": 50000, "cost_usd": 1.0, "web_searches": 15},
        )

        assert contract.resources.tokens == 50000
        assert contract.resources.cost_usd == 1.0
        assert contract.resources.web_searches == 15

    def test_research_contract_custom_id(self) -> None:
        """Test research contract with custom ID."""
        contract = ResearchContract.create(topic="Test", contract_id="my-research-123")

        assert contract.id == "my-research-123"

    def test_research_contract_auto_id(self) -> None:
        """Test research contract with auto-generated ID."""
        contract = ResearchContract.create(topic="AI Safety Research")

        assert "ai-safety-research" in contract.id.lower()

    def test_research_contract_depth_parameter(self) -> None:
        """Test research contract with different depth levels."""
        for depth in ["overview", "standard", "comprehensive"]:
            contract = ResearchContract.create(topic="Test", depth=depth)
            assert contract.metadata["depth"] == depth

    def test_research_contract_metadata(self) -> None:
        """Test research contract metadata."""
        contract = ResearchContract.create(topic="Test Topic", depth="comprehensive")

        assert contract.metadata["topic"] == "Test Topic"
        assert contract.metadata["depth"] == "comprehensive"
        assert contract.metadata["template"] == "ResearchContract"
        assert "allocation" in contract.metadata
        assert contract.metadata["allocation"]["gathering"] == 0.40
        assert contract.metadata["allocation"]["analysis"] == 0.30
        assert contract.metadata["allocation"]["writing"] == 0.30


class TestCodeReviewContract:
    """Test CodeReviewContract template."""

    def test_create_code_review_small_pr(self) -> None:
        """Test code review for small PR (<5 files)."""
        contract = CodeReviewContract.create(
            repository="org/repo", pr_number=123, files_changed=3
        )

        assert isinstance(contract, Contract)
        assert contract.name == "Code Review: org/repo"
        assert contract.resources.tokens == 30_000
        assert contract.resources.api_calls == 15
        assert contract.resources.cost_usd == 1.50
        assert contract.resources.web_searches == 0  # No web access

    def test_code_review_medium_pr(self) -> None:
        """Test code review for medium PR (5-15 files)."""
        contract = CodeReviewContract.create(
            repository="org/repo", pr_number=456, files_changed=10
        )

        assert contract.resources.tokens == 50_000
        assert contract.resources.api_calls == 25
        assert contract.resources.cost_usd == 2.50

    def test_code_review_large_pr(self) -> None:
        """Test code review for large PR (>15 files)."""
        contract = CodeReviewContract.create(
            repository="org/repo", pr_number=789, files_changed=20
        )

        assert contract.resources.tokens == 80_000
        assert contract.resources.api_calls == 40
        assert contract.resources.cost_usd == 4.00

    def test_code_review_without_pr_number(self) -> None:
        """Test code review without PR number."""
        contract = CodeReviewContract.create(repository="org/repo", files_changed=5)

        assert contract.id == "code-review-org-repo"
        assert contract.metadata["pr_number"] is None

    def test_code_review_with_pr_number_in_id(self) -> None:
        """Test that PR number is included in contract ID."""
        contract = CodeReviewContract.create(
            repository="org/repo", pr_number=123, files_changed=5
        )

        assert "pr123" in contract.id

    def test_code_review_strict_mode(self) -> None:
        """Test code review with strict mode."""
        from agent_contracts.core.contract import DeadlineType

        contract = CodeReviewContract.create(
            repository="org/repo", files_changed=5, strict_mode=True
        )

        assert contract.temporal.deadline_type == DeadlineType.HARD

    def test_code_review_lenient_mode(self) -> None:
        """Test code review with lenient mode."""
        from agent_contracts.core.contract import DeadlineType

        contract = CodeReviewContract.create(
            repository="org/repo", files_changed=5, strict_mode=False
        )

        assert contract.temporal.deadline_type == DeadlineType.SOFT

    def test_code_review_custom_resources(self) -> None:
        """Test code review with custom resources."""
        contract = CodeReviewContract.create(
            repository="org/repo",
            files_changed=5,
            resources={"tokens": 100000, "cost_usd": 5.0},
        )

        assert contract.resources.tokens == 100000
        assert contract.resources.cost_usd == 5.0

    def test_code_review_metadata(self) -> None:
        """Test code review metadata."""
        contract = CodeReviewContract.create(
            repository="org/repo", pr_number=123, files_changed=10
        )

        assert contract.metadata["repository"] == "org/repo"
        assert contract.metadata["pr_number"] == 123
        assert contract.metadata["files_changed"] == 10
        assert contract.metadata["template"] == "CodeReviewContract"
        assert "skills" in contract.metadata


class TestCustomerSupportContract:
    """Test CustomerSupportContract template."""

    def test_create_customer_support_normal_priority(self) -> None:
        """Test customer support with normal priority."""
        contract = CustomerSupportContract.create(ticket_id="TICKET-123")

        assert isinstance(contract, Contract)
        assert contract.name == "Customer Support: TICKET-123"
        assert contract.mode == ContractMode.BALANCED
        assert contract.resources.tokens == 8_000
        assert contract.resources.api_calls == 10

    def test_customer_support_high_priority(self) -> None:
        """Test customer support with high priority."""
        contract = CustomerSupportContract.create(
            ticket_id="TICKET-456", priority="high"
        )

        assert contract.mode == ContractMode.URGENT
        assert contract.resources.tokens == 10_000
        assert contract.resources.cost_usd == 0.50
        assert contract.temporal.max_duration == timedelta(minutes=2)

    def test_customer_support_urgent_priority(self) -> None:
        """Test customer support with urgent priority."""
        contract = CustomerSupportContract.create(
            ticket_id="TICKET-789", priority="urgent"
        )

        assert contract.mode == ContractMode.URGENT

    def test_customer_support_low_priority(self) -> None:
        """Test customer support with low priority."""
        contract = CustomerSupportContract.create(ticket_id="TICKET-999", priority="low")

        assert contract.mode == ContractMode.ECONOMICAL
        assert contract.resources.tokens == 5_000
        assert contract.resources.cost_usd == 0.15
        assert contract.temporal.max_duration == timedelta(minutes=10)

    def test_customer_support_explicit_mode(self) -> None:
        """Test customer support with explicit mode override."""
        contract = CustomerSupportContract.create(
            ticket_id="TICKET-111", priority="high", mode=ContractMode.ECONOMICAL
        )

        # Mode should override priority
        assert contract.mode == ContractMode.ECONOMICAL

    def test_customer_support_custom_resources(self) -> None:
        """Test customer support with custom resources."""
        contract = CustomerSupportContract.create(
            ticket_id="TICKET-222", resources={"tokens": 20000, "cost_usd": 1.0}
        )

        assert contract.resources.tokens == 20000
        assert contract.resources.cost_usd == 1.0

    def test_customer_support_metadata(self) -> None:
        """Test customer support metadata."""
        contract = CustomerSupportContract.create(
            ticket_id="TICKET-333", priority="high"
        )

        assert contract.metadata["ticket_id"] == "TICKET-333"
        assert contract.metadata["priority"] == "high"
        assert contract.metadata["template"] == "CustomerSupportContract"


class TestDataAnalysisContract:
    """Test DataAnalysisContract template."""

    def test_create_data_analysis_small_dataset(self) -> None:
        """Test data analysis for small dataset (<1MB)."""
        contract = DataAnalysisContract.create(
            dataset_name="small.csv", dataset_size_mb=0.5
        )

        assert isinstance(contract, Contract)
        assert contract.name == "Data Analysis: small.csv"
        assert contract.resources.tokens == 20_000
        assert contract.resources.api_calls == 10
        assert contract.resources.cost_usd == 1.00
        assert contract.resources.memory_mb == 512.0

    def test_data_analysis_medium_dataset(self) -> None:
        """Test data analysis for medium dataset (1-10MB)."""
        contract = DataAnalysisContract.create(
            dataset_name="medium.csv", dataset_size_mb=5.0
        )

        assert contract.resources.tokens == 50_000
        assert contract.resources.api_calls == 25
        assert contract.resources.cost_usd == 2.50
        assert contract.resources.memory_mb == 1024.0

    def test_data_analysis_large_dataset(self) -> None:
        """Test data analysis for large dataset (>10MB)."""
        contract = DataAnalysisContract.create(
            dataset_name="large.csv", dataset_size_mb=15.0
        )

        assert contract.resources.tokens == 100_000
        assert contract.resources.api_calls == 50
        assert contract.resources.cost_usd == 5.00
        assert contract.resources.memory_mb == 2048.0

    def test_data_analysis_no_web_access(self) -> None:
        """Test that data analysis has no web access."""
        contract = DataAnalysisContract.create(
            dataset_name="test.csv", dataset_size_mb=1.0
        )

        assert contract.resources.web_searches == 0

    def test_data_analysis_types(self) -> None:
        """Test different analysis types."""
        for analysis_type in ["exploratory", "statistical", "predictive"]:
            contract = DataAnalysisContract.create(
                dataset_name="test.csv", dataset_size_mb=1.0, analysis_type=analysis_type
            )
            assert contract.metadata["analysis_type"] == analysis_type

    def test_data_analysis_custom_resources(self) -> None:
        """Test data analysis with custom resources."""
        contract = DataAnalysisContract.create(
            dataset_name="test.csv",
            dataset_size_mb=1.0,
            resources={"tokens": 30000, "memory_mb": 4096.0},
        )

        assert contract.resources.tokens == 30000
        assert contract.resources.memory_mb == 4096.0

    def test_data_analysis_metadata(self) -> None:
        """Test data analysis metadata."""
        contract = DataAnalysisContract.create(
            dataset_name="sales.csv", dataset_size_mb=5.0, analysis_type="statistical"
        )

        assert contract.metadata["dataset_name"] == "sales.csv"
        assert contract.metadata["dataset_size_mb"] == 5.0
        assert contract.metadata["analysis_type"] == "statistical"
        assert contract.metadata["template"] == "DataAnalysisContract"


class TestGetTemplate:
    """Test get_template utility function."""

    def test_get_research_template(self) -> None:
        """Test getting research template by name."""
        template = get_template("research")
        assert template == ResearchContract

    def test_get_code_review_template(self) -> None:
        """Test getting code review template by name."""
        template = get_template("code_review")
        assert template == CodeReviewContract

    def test_get_customer_support_template(self) -> None:
        """Test getting customer support template by name."""
        template = get_template("customer_support")
        assert template == CustomerSupportContract

    def test_get_data_analysis_template(self) -> None:
        """Test getting data analysis template by name."""
        template = get_template("data_analysis")
        assert template == DataAnalysisContract

    def test_get_invalid_template(self) -> None:
        """Test getting invalid template raises error."""
        with pytest.raises(KeyError) as exc_info:
            get_template("invalid_template")

        assert "invalid_template" in str(exc_info.value)
        assert "Available templates" in str(exc_info.value)

    def test_use_template_via_get_template(self) -> None:
        """Test using template obtained via get_template."""
        template = get_template("research")
        contract = template.create(topic="Test Topic")

        assert isinstance(contract, Contract)
        assert "Test Topic" in contract.name


class TestTemplateIntegration:
    """Integration tests for contract templates."""

    def test_all_templates_create_valid_contracts(self) -> None:
        """Test that all templates create valid contracts."""
        # Research
        research = ResearchContract.create(topic="Test")
        assert isinstance(research, Contract)
        assert research.resources is not None

        # Code Review
        code_review = CodeReviewContract.create(repository="org/repo", files_changed=5)
        assert isinstance(code_review, Contract)
        assert code_review.resources is not None

        # Customer Support
        support = CustomerSupportContract.create(ticket_id="TEST-123")
        assert isinstance(support, Contract)
        assert support.resources is not None

        # Data Analysis
        analysis = DataAnalysisContract.create(
            dataset_name="test.csv", dataset_size_mb=1.0
        )
        assert isinstance(analysis, Contract)
        assert analysis.resources is not None

    def test_templates_with_contract_agent(self) -> None:
        """Test using templates with ContractAgent."""
        from agent_contracts.core.wrapper import ContractAgent

        contract = ResearchContract.create(topic="AI", mode=ContractMode.ECONOMICAL)

        def agent(x: str) -> str:
            return f"Result: {x}"

        wrapped = ContractAgent(contract=contract, agent=agent)
        result = wrapped.execute("test")

        assert result.success is True
        assert result.contract.mode == ContractMode.ECONOMICAL

    def test_template_resource_scaling(self) -> None:
        """Test that templates scale resources appropriately."""
        # Code review scales by PR size
        small_pr = CodeReviewContract.create(repository="test", files_changed=2)
        medium_pr = CodeReviewContract.create(repository="test", files_changed=10)
        large_pr = CodeReviewContract.create(repository="test", files_changed=25)

        assert small_pr.resources.tokens < medium_pr.resources.tokens < large_pr.resources.tokens

        # Data analysis scales by dataset size
        small_data = DataAnalysisContract.create(
            dataset_name="test", dataset_size_mb=0.5
        )
        medium_data = DataAnalysisContract.create(dataset_name="test", dataset_size_mb=5.0)
        large_data = DataAnalysisContract.create(dataset_name="test", dataset_size_mb=15.0)

        assert (
            small_data.resources.tokens
            < medium_data.resources.tokens
            < large_data.resources.tokens
        )

    def test_template_mode_variations(self) -> None:
        """Test templates with different modes."""
        for mode in [ContractMode.URGENT, ContractMode.BALANCED, ContractMode.ECONOMICAL]:
            contract = ResearchContract.create(topic="Test", mode=mode)
            assert contract.mode == mode

            # Temporal constraints should vary by mode
            if mode == ContractMode.URGENT:
                assert contract.temporal.max_duration == timedelta(minutes=30)
            elif mode == ContractMode.BALANCED:
                assert contract.temporal.max_duration == timedelta(hours=2)
            elif mode == ContractMode.ECONOMICAL:
                assert contract.temporal.max_duration == timedelta(hours=4)
