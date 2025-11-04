"""Pre-configured contract templates for common use cases.

This module provides ready-to-use contract templates based on the whitepaper
examples (Sections 6.1, 6.2) and common production scenarios.

Templates can be used as-is or customized for specific needs.

Example:
    >>> from agent_contracts.templates import ResearchContract
    >>>
    >>> # Use template with custom topic
    >>> contract = ResearchContract.create(
    ...     topic="AI Safety Research",
    ...     depth="comprehensive"
    ... )
    >>>
    >>> # Customize resource limits
    >>> contract = ResearchContract.create(
    ...     topic="Market Analysis",
    ...     resources={"tokens": 50000, "cost_usd": 2.0}
    ... )
"""

from datetime import timedelta
from typing import Any

from agent_contracts.core.contract import (
    Contract,
    ContractMode,
    DeadlineType,
    ResourceConstraints,
    TemporalConstraints,
)


class ResearchContract:
    """Research report generation contract (Whitepaper Section 6.1).

    This template is designed for multi-step research tasks that involve:
    - Information gathering (web search, document retrieval)
    - Analysis and synthesis
    - Report writing

    Resource allocation:
    - 40% for information gathering
    - 30% for analysis
    - 30% for report writing

    Modes:
    - Urgent: 30 min, 200K tokens, 30 searches (85% quality)
    - Normal: 2 hours, 150K tokens, 20 searches (95% quality)
    - Economical: 4 hours, 80K tokens, 10 searches (90% quality)
    """

    @staticmethod
    def create(
        topic: str,
        depth: str = "comprehensive",
        mode: ContractMode = ContractMode.BALANCED,
        resources: dict[str, Any] | None = None,
        contract_id: str | None = None,
    ) -> Contract:
        """Create a research contract.

        Args:
            topic: Research topic (e.g., "Electric Vehicle Market in Europe")
            depth: Depth of research ("overview", "standard", "comprehensive")
            mode: Contract mode (URGENT, BALANCED, ECONOMICAL)
            resources: Optional custom resource limits
            contract_id: Optional contract ID

        Returns:
            Configured Contract instance

        Example:
            >>> contract = ResearchContract.create(
            ...     topic="Quantum Computing Applications",
            ...     depth="comprehensive",
            ...     mode=ContractMode.BALANCED
            ... )
        """
        # Default resource allocations by mode
        mode_configs = {
            ContractMode.URGENT: {
                "tokens": 200_000,
                "web_searches": 30,
                "api_calls": 60,
                "cost_usd": 4.0,
                "max_duration": timedelta(minutes=30),
            },
            ContractMode.BALANCED: {
                "tokens": 150_000,
                "web_searches": 20,
                "api_calls": 40,
                "cost_usd": 3.0,
                "max_duration": timedelta(hours=2),
            },
            ContractMode.ECONOMICAL: {
                "tokens": 80_000,
                "web_searches": 10,
                "api_calls": 25,
                "cost_usd": 1.5,
                "max_duration": timedelta(hours=4),
            },
        }

        # Get config for selected mode
        config = mode_configs[mode]

        # Override with custom resources if provided
        if resources:
            config.update(resources)

        # Create resource constraints
        resource_constraints = ResourceConstraints(
            tokens=config.get("tokens"),
            web_searches=config.get("web_searches"),
            api_calls=config.get("api_calls"),
            cost_usd=config.get("cost_usd"),
        )

        # Create temporal constraints
        temporal_constraints = TemporalConstraints(
            max_duration=config.get("max_duration"),
            deadline_type=DeadlineType.SOFT,
        )

        # Create contract
        return Contract(
            id=contract_id or f"research-{topic.lower().replace(' ', '-')}",
            name=f"Research: {topic}",
            mode=mode,
            resources=resource_constraints,
            temporal=temporal_constraints,
            metadata={
                "topic": topic,
                "depth": depth,
                "template": "ResearchContract",
                "allocation": {
                    "gathering": 0.40,
                    "analysis": 0.30,
                    "writing": 0.30,
                },
            },
        )


class CodeReviewContract:
    """Code review agent contract (Whitepaper Section 6.2).

    This template is designed for automated pull request reviews:
    - Static analysis
    - Security scanning
    - Style checking
    - Complexity analysis

    Scales with PR size (files changed):
    - Small (<5 files): 30K tokens, $1.50
    - Medium (5-15 files): 50K tokens, $2.50
    - Large (>15 files): 80K tokens, $4.00
    """

    @staticmethod
    def create(
        repository: str,
        pr_number: int | None = None,
        files_changed: int = 10,
        strict_mode: bool = True,
        resources: dict[str, Any] | None = None,
        contract_id: str | None = None,
    ) -> Contract:
        """Create a code review contract.

        Args:
            repository: Repository name (e.g., "company/product")
            pr_number: Pull request number (optional)
            files_changed: Number of files changed in PR
            strict_mode: If True, enforce hard limits
            resources: Optional custom resource limits
            contract_id: Optional contract ID

        Returns:
            Configured Contract instance

        Example:
            >>> contract = CodeReviewContract.create(
            ...     repository="myorg/myapp",
            ...     pr_number=1234,
            ...     files_changed=12
            ... )
        """
        # Scale resources based on PR size
        if files_changed < 5:
            # Small PR
            default_config = {
                "tokens": 30_000,
                "api_calls": 15,
                "cost_usd": 1.50,
                "max_duration": timedelta(minutes=3),
            }
        elif files_changed <= 15:
            # Medium PR
            default_config = {
                "tokens": 50_000,
                "api_calls": 25,
                "cost_usd": 2.50,
                "max_duration": timedelta(minutes=5),
            }
        else:
            # Large PR
            default_config = {
                "tokens": 80_000,
                "api_calls": 40,
                "cost_usd": 4.00,
                "max_duration": timedelta(minutes=10),
            }

        # Override with custom resources if provided
        if resources:
            default_config.update(resources)

        # Create resource constraints
        resource_constraints = ResourceConstraints(
            tokens=default_config.get("tokens"),
            api_calls=default_config.get("api_calls"),
            cost_usd=default_config.get("cost_usd"),
            web_searches=0,  # No web access for code review
        )

        # Create temporal constraints
        temporal_constraints = TemporalConstraints(
            max_duration=default_config.get("max_duration"),
            deadline_type=DeadlineType.SOFT if not strict_mode else DeadlineType.HARD,
        )

        # Create contract
        pr_id = f"-pr{pr_number}" if pr_number else ""
        return Contract(
            id=contract_id or f"code-review-{repository.replace('/', '-')}{pr_id}",
            name=f"Code Review: {repository}",
            mode=ContractMode.BALANCED,
            resources=resource_constraints,
            temporal=temporal_constraints,
            metadata={
                "repository": repository,
                "pr_number": pr_number,
                "files_changed": files_changed,
                "template": "CodeReviewContract",
                "skills": [
                    "static_analysis",
                    "security_scanning",
                    "style_checking",
                    "complexity_analysis",
                ],
            },
        )


class CustomerSupportContract:
    """Customer support agent contract.

    This template is designed for handling customer support tickets:
    - Ticket triage and classification
    - Knowledge base search
    - Response generation
    - Escalation handling

    Modes:
    - Urgent: High-priority tickets, faster response (2 min)
    - Normal: Standard tickets, balanced quality (5 min)
    - Economical: Low-priority, cost-optimized (10 min)
    """

    @staticmethod
    def create(
        ticket_id: str,
        priority: str = "normal",
        mode: ContractMode | None = None,
        resources: dict[str, Any] | None = None,
        contract_id: str | None = None,
    ) -> Contract:
        """Create a customer support contract.

        Args:
            ticket_id: Unique ticket identifier
            priority: Ticket priority ("low", "normal", "high", "urgent")
            mode: Contract mode (auto-selected based on priority if not specified)
            resources: Optional custom resource limits
            contract_id: Optional contract ID

        Returns:
            Configured Contract instance

        Example:
            >>> contract = CustomerSupportContract.create(
            ...     ticket_id="TICKET-1234",
            ...     priority="high"
            ... )
        """
        # Auto-select mode based on priority if not specified
        if mode is None:
            mode_mapping = {
                "low": ContractMode.ECONOMICAL,
                "normal": ContractMode.BALANCED,
                "high": ContractMode.URGENT,
                "urgent": ContractMode.URGENT,
            }
            mode = mode_mapping.get(priority.lower(), ContractMode.BALANCED)

        # Resource configurations by mode
        mode_configs = {
            ContractMode.URGENT: {
                "tokens": 10_000,
                "api_calls": 15,
                "web_searches": 5,
                "cost_usd": 0.50,
                "max_duration": timedelta(minutes=2),
            },
            ContractMode.BALANCED: {
                "tokens": 8_000,
                "api_calls": 10,
                "web_searches": 3,
                "cost_usd": 0.30,
                "max_duration": timedelta(minutes=5),
            },
            ContractMode.ECONOMICAL: {
                "tokens": 5_000,
                "api_calls": 5,
                "web_searches": 2,
                "cost_usd": 0.15,
                "max_duration": timedelta(minutes=10),
            },
        }

        # Get config for selected mode
        config = mode_configs[mode]

        # Override with custom resources if provided
        if resources:
            config.update(resources)

        # Create resource constraints
        resource_constraints = ResourceConstraints(
            tokens=config.get("tokens"),
            api_calls=config.get("api_calls"),
            web_searches=config.get("web_searches"),
            cost_usd=config.get("cost_usd"),
        )

        # Create temporal constraints
        temporal_constraints = TemporalConstraints(
            max_duration=config.get("max_duration"),
            deadline_type=DeadlineType.SOFT,
        )

        # Create contract
        return Contract(
            id=contract_id or f"support-{ticket_id}",
            name=f"Customer Support: {ticket_id}",
            mode=mode,
            resources=resource_constraints,
            temporal=temporal_constraints,
            metadata={
                "ticket_id": ticket_id,
                "priority": priority,
                "template": "CustomerSupportContract",
                "skills": ["triage", "knowledge_search", "response_generation"],
            },
        )


class DataAnalysisContract:
    """Data analysis agent contract.

    This template is designed for data analysis tasks:
    - Data loading and validation
    - Statistical analysis
    - Visualization generation
    - Insights reporting

    Scales with dataset size:
    - Small (<1MB): 20K tokens, $1.00
    - Medium (1-10MB): 50K tokens, $2.50
    - Large (>10MB): 100K tokens, $5.00
    """

    @staticmethod
    def create(
        dataset_name: str,
        dataset_size_mb: float = 1.0,
        analysis_type: str = "exploratory",
        resources: dict[str, Any] | None = None,
        contract_id: str | None = None,
    ) -> Contract:
        """Create a data analysis contract.

        Args:
            dataset_name: Name of dataset to analyze
            dataset_size_mb: Dataset size in MB
            analysis_type: Type of analysis ("exploratory", "statistical", "predictive")
            resources: Optional custom resource limits
            contract_id: Optional contract ID

        Returns:
            Configured Contract instance

        Example:
            >>> contract = DataAnalysisContract.create(
            ...     dataset_name="sales_data.csv",
            ...     dataset_size_mb=5.2,
            ...     analysis_type="statistical"
            ... )
        """
        # Scale resources based on dataset size
        if dataset_size_mb < 1.0:
            # Small dataset
            default_config = {
                "tokens": 20_000,
                "api_calls": 10,
                "cost_usd": 1.00,
                "memory_mb": 512.0,
                "max_duration": timedelta(minutes=5),
            }
        elif dataset_size_mb <= 10.0:
            # Medium dataset
            default_config = {
                "tokens": 50_000,
                "api_calls": 25,
                "cost_usd": 2.50,
                "memory_mb": 1024.0,
                "max_duration": timedelta(minutes=15),
            }
        else:
            # Large dataset
            default_config = {
                "tokens": 100_000,
                "api_calls": 50,
                "cost_usd": 5.00,
                "memory_mb": 2048.0,
                "max_duration": timedelta(minutes=30),
            }

        # Override with custom resources if provided
        if resources:
            default_config.update(resources)

        # Create resource constraints
        resource_constraints = ResourceConstraints(
            tokens=default_config.get("tokens"),
            api_calls=default_config.get("api_calls"),
            cost_usd=default_config.get("cost_usd"),
            memory_mb=default_config.get("memory_mb"),
            web_searches=0,  # No web access for data analysis
        )

        # Create temporal constraints
        temporal_constraints = TemporalConstraints(
            max_duration=default_config.get("max_duration"),
            deadline_type=DeadlineType.SOFT,
        )

        # Create contract
        return Contract(
            id=contract_id or f"analysis-{dataset_name.replace('.', '-')}",
            name=f"Data Analysis: {dataset_name}",
            mode=ContractMode.BALANCED,
            resources=resource_constraints,
            temporal=temporal_constraints,
            metadata={
                "dataset_name": dataset_name,
                "dataset_size_mb": dataset_size_mb,
                "analysis_type": analysis_type,
                "template": "DataAnalysisContract",
                "skills": ["data_loading", "statistical_analysis", "visualization"],
            },
        )


# Convenience dict for template lookup
TEMPLATES = {
    "research": ResearchContract,
    "code_review": CodeReviewContract,
    "customer_support": CustomerSupportContract,
    "data_analysis": DataAnalysisContract,
}


def get_template(template_name: str) -> type:
    """Get a contract template by name.

    Args:
        template_name: Name of template ("research", "code_review", etc.)

    Returns:
        Template class

    Raises:
        KeyError: If template name not found

    Example:
        >>> template = get_template("research")
        >>> contract = template.create(topic="AI Ethics")
    """
    if template_name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise KeyError(
            f"Template '{template_name}' not found. Available templates: {available}"
        )
    return TEMPLATES[template_name]
