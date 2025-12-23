"""Research pipeline orchestrator for COINE 2026 evaluation.

This module implements the multi-agent research report generation pipeline
in two modes:
1. UNCONTRACTED: Agents run without budget enforcement
2. CONTRACTED: Agents run with Agent Contracts + conservation laws

The orchestrator coordinates:
- Researcher: Gathers information via web search
- Analyzer: Identifies patterns and insights
- Reporter: Synthesizes into final report

Budget Allocation (from SUBMISSION_PLAN.md):
    Parent Contract: B = 100,000 tokens, $2.00, 15 minutes
    ├── Orchestrator: 10,000 tokens (coordination, validation)
    ├── Researcher: 40,000 tokens (web search, data gathering)
    ├── Analyzer: 25,000 tokens (pattern identification)
    └── Reporter: 25,000 tokens (synthesis, writing)
"""

import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from .agents import (
    create_all_agents,
    create_analyzer_agent,
    create_reporter_agent,
    create_researcher_agent,
)
from .topics import ResearchTopic

# Type checking imports
try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner

    GOOGLE_ADK_AVAILABLE = True
except ImportError:
    GOOGLE_ADK_AVAILABLE = False
    LlmAgent = Any
    InMemoryRunner = Any


@dataclass
class PipelineResult:
    """Result from a research pipeline execution.

    Attributes:
        topic: The research topic
        mode: Execution mode ("CONTRACTED" or "UNCONTRACTED")
        success: Whether the pipeline completed successfully
        report: Final generated report text
        word_count: Number of words in the report
        citation_count: Number of citations found
        total_tokens: Total tokens consumed
        tokens_by_agent: Token breakdown by agent
        execution_time_seconds: Total execution time
        budget_compliant: Whether execution stayed within budget
        conservation_violations: Number of conservation law violations
        error: Error message if failed
        raw_outputs: Raw outputs from each agent
    """

    topic: ResearchTopic
    mode: str
    success: bool = False
    report: str = ""
    word_count: int = 0
    citation_count: int = 0
    total_tokens: int = 0
    tokens_by_agent: dict[str, int] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    budget_compliant: bool = True
    conservation_violations: int = 0
    error: str = ""
    raw_outputs: dict[str, str] = field(default_factory=dict)


@dataclass
class SuccessCriteria:
    """Success criteria for research report (from Section 8).

    Φ(O) = w₁·complete(O) + w₂·length(O) + w₃·citations(O)

    Attributes:
        sections_complete_weight: Weight for all sections complete (0.4)
        word_count_weight: Weight for word count (0.3)
        citation_weight: Weight for citations (0.3)
        min_words: Minimum word count (2000)
        min_citations: Minimum citation count (5)
        threshold: Success threshold θ (0.8)
    """

    sections_complete_weight: float = 0.4
    word_count_weight: float = 0.3
    citation_weight: float = 0.3
    min_words: int = 2000
    min_citations: int = 5
    threshold: float = 0.8

    def evaluate(self, result: PipelineResult) -> tuple[float, bool]:
        """Evaluate if result meets success criteria.

        Args:
            result: Pipeline execution result

        Returns:
            Tuple of (score, success) where score is 0-1 and success is bool
        """
        # Check sections (simplified: check for Introduction, Body, Conclusion)
        sections_score = 0.0
        report_lower = result.report.lower()
        if "introduction" in report_lower:
            sections_score += 0.33
        if "conclusion" in report_lower:
            sections_score += 0.33
        # Body is implied if we have content
        if result.word_count > 500:
            sections_score += 0.34

        # Word count score (linear up to min_words)
        word_score = min(1.0, result.word_count / self.min_words)

        # Citation score (linear up to min_citations)
        citation_score = min(1.0, result.citation_count / self.min_citations)

        # Weighted sum
        total_score = (
            self.sections_complete_weight * sections_score
            + self.word_count_weight * word_score
            + self.citation_weight * citation_score
        )

        return total_score, total_score >= self.threshold


class UncontractedPipeline:
    """Research pipeline WITHOUT contract enforcement.

    This is the baseline condition - agents run without budget limits.
    Used to demonstrate runaway behavior.
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize uncontracted pipeline.

        Args:
            verbose: If True, print progress messages
        """
        if not GOOGLE_ADK_AVAILABLE:
            raise ImportError("google-adk is required. Install with: uv sync --extra google-adk")

        self.verbose = verbose
        self._agents = create_all_agents()

    def run(self, topic: ResearchTopic) -> PipelineResult:
        """Execute research pipeline without contracts.

        Args:
            topic: Research topic to generate report for

        Returns:
            PipelineResult with execution details
        """
        result = PipelineResult(topic=topic, mode="UNCONTRACTED")
        start_time = time.time()

        try:
            # Phase 1: Research
            if self.verbose:
                print(f"  [Researcher] Researching: {topic.title}")

            researcher = self._agents["researcher"]
            runner = InMemoryRunner(agent=researcher, app_name="uncontracted-researcher")

            research_prompt = f"""Research the following topic thoroughly:

Topic: {topic.title}
Description: {topic.description}

Key aspects to cover:
{chr(10).join(f"- {aspect}" for aspect in topic.key_aspects)}

Find current information, facts, statistics, and expert opinions.
Cite your sources with URLs."""

            research_output = self._run_agent(
                runner, research_prompt, "researcher", "uncontracted-researcher"
            )
            result.raw_outputs["researcher"] = research_output["response"]
            result.tokens_by_agent["researcher"] = research_output["tokens"]

            # Phase 2: Analysis
            if self.verbose:
                print("  [Analyzer] Analyzing findings...")

            analyzer = self._agents["analyzer"]
            runner = InMemoryRunner(agent=analyzer, app_name="uncontracted-analyzer")

            analysis_prompt = f"""Analyze the following research findings:

{research_output["response"]}

Identify:
1. Key patterns and trends
2. Important insights
3. Connections between sources
4. Most significant findings

Structure your analysis into clear themes."""

            analysis_output = self._run_agent(
                runner, analysis_prompt, "analyzer", "uncontracted-analyzer"
            )
            result.raw_outputs["analyzer"] = analysis_output["response"]
            result.tokens_by_agent["analyzer"] = analysis_output["tokens"]

            # Phase 3: Report Generation
            if self.verbose:
                print("  [Reporter] Writing report...")

            reporter = self._agents["reporter"]
            runner = InMemoryRunner(agent=reporter, app_name="uncontracted-reporter")

            report_prompt = f"""Write a comprehensive research report on:

Topic: {topic.title}

Based on this analysis:
{analysis_output["response"]}

Requirements:
- At least 2,000 words
- Include Introduction, Main Body (with sections), and Conclusion
- Include at least 5 citations with URLs
- Be professional and well-organized"""

            report_output = self._run_agent(
                runner, report_prompt, "reporter", "uncontracted-reporter"
            )
            result.raw_outputs["reporter"] = report_output["response"]
            result.tokens_by_agent["reporter"] = report_output["tokens"]

            # Compile results
            result.report = report_output["response"]
            result.word_count = len(result.report.split())
            result.citation_count = self._count_citations(result.report)
            result.total_tokens = sum(result.tokens_by_agent.values())
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        result.execution_time_seconds = time.time() - start_time
        return result

    def _run_agent(
        self, runner: "InMemoryRunner", message: str, agent_name: str, app_name: str = ""
    ) -> dict[str, Any]:
        """Run an agent and collect output.

        Args:
            runner: InMemoryRunner for the agent
            message: Message to send
            agent_name: Name of agent (for session ID)
            app_name: App name for session creation

        Returns:
            Dictionary with response and token count
        """
        import asyncio

        from google.genai.types import Content, Part

        # Use provided app_name or construct from agent_name
        session_app_name = app_name or f"uncontracted-{agent_name}"

        # Create session
        async def create_session() -> None:
            await runner.session_service.create_session(
                app_name=session_app_name,
                user_id="eval_user",
                session_id=f"eval_{agent_name}",
            )

        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(create_session(), loop)
            future.result(timeout=30)
        except RuntimeError:
            asyncio.run(create_session())

        # Run agent
        content = Content(parts=[Part(text=message)])
        events = list(
            runner.run(
                user_id="eval_user",
                session_id=f"eval_{agent_name}",
                new_message=content,
            )
        )

        # Extract response and tokens
        response = ""
        total_tokens = 0

        for event in events:
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                total_tokens += getattr(event.usage_metadata, "total_token_count", 0)
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response = part.text

        return {"response": response, "tokens": total_tokens}

    def _count_citations(self, text: str) -> int:
        """Count citations in text (URLs or [n] references)."""
        import re

        # Count URLs
        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, text)

        # Count [n] style references
        ref_pattern = r"\[\d+\]"
        refs = re.findall(ref_pattern, text)

        return len(set(urls)) + len(set(refs))


class ContractedPipeline:
    """Research pipeline WITH contract enforcement and conservation laws.

    This is the experimental condition - demonstrates Agent Contracts value.
    """

    # Budget allocation from SUBMISSION_PLAN.md
    PARENT_TOKENS = 100_000
    PARENT_COST = 2.0
    PARENT_DURATION = timedelta(minutes=15)
    PARENT_ITERATIONS = 50  # Max LLM calls to prevent runaway loops

    ORCHESTRATOR_TOKENS = 10_000
    RESEARCHER_TOKENS = 40_000
    RESEARCHER_ITERATIONS = 15  # Researcher may need multiple search iterations
    ANALYZER_TOKENS = 25_000
    ANALYZER_ITERATIONS = 10
    REPORTER_TOKENS = 25_000
    REPORTER_ITERATIONS = 10

    def __init__(self, verbose: bool = False, strict_mode: bool = True) -> None:
        """Initialize contracted pipeline.

        Args:
            verbose: If True, print progress messages
            strict_mode: If True, stop on budget violations
        """
        if not GOOGLE_ADK_AVAILABLE:
            raise ImportError("google-adk is required. Install with: uv sync --extra google-adk")

        self.verbose = verbose
        self.strict_mode = strict_mode

    def run(self, topic: ResearchTopic) -> PipelineResult:
        """Execute research pipeline with contracts.

        Args:
            topic: Research topic to generate report for

        Returns:
            PipelineResult with execution details
        """
        from agent_contracts import Contract, ResourceConstraints, TemporalConstraints
        from agent_contracts.core.prompts import generate_budget_prompt
        from agent_contracts.integrations.google_adk import DelegatingAdkAgent

        result = PipelineResult(topic=topic, mode="CONTRACTED")
        start_time = time.time()

        try:
            # Create parent contract with iteration limit to prevent runaway loops
            parent_contract = Contract(
                id=f"report-{topic.id}",
                name=f"Research Report: {topic.title}",
                resources=ResourceConstraints(
                    tokens=self.PARENT_TOKENS,
                    cost_usd=self.PARENT_COST,
                    iterations=self.PARENT_ITERATIONS,  # Prevents runaway agent loops
                ),
                temporal=TemporalConstraints(
                    max_duration=self.PARENT_DURATION,
                ),
            )

            # Create orchestrator with delegation capability
            from .agents import create_orchestrator_agent

            orchestrator_agent = create_orchestrator_agent()

            delegating = DelegatingAdkAgent(
                contract=parent_contract,
                agent=orchestrator_agent,
                strict_mode=self.strict_mode,
                reserve_ratio=0.1,  # 10% reserve for coordination
            )

            # Delegate to sub-agents with conservation law enforcement
            # Each agent gets its own iteration limit to prevent runaway loops
            researcher_agent = create_researcher_agent()
            researcher = delegating.delegate(
                name="researcher",
                agent=researcher_agent,
                tokens=self.RESEARCHER_TOKENS,
                iterations=self.RESEARCHER_ITERATIONS,
                description="Research the topic",
            )

            analyzer_agent = create_analyzer_agent()
            analyzer = delegating.delegate(
                name="analyzer",
                agent=analyzer_agent,
                tokens=self.ANALYZER_TOKENS,
                iterations=self.ANALYZER_ITERATIONS,
                description="Analyze findings",
            )

            reporter_agent = create_reporter_agent()
            reporter = delegating.delegate(
                name="reporter",
                agent=reporter_agent,
                tokens=self.REPORTER_TOKENS,
                iterations=self.REPORTER_ITERATIONS,
                description="Write report",
            )

            if self.verbose:
                summary = delegating.get_delegation_summary()
                print(f"  [Contracts] Parent: {summary['parent_budget_tokens']:,} tokens")
                print(f"  [Contracts] Delegated: {summary['total_delegated_tokens']:,} tokens")
                print(f"  [Contracts] Remaining: {summary['remaining_tokens']:,} tokens")

            # Phase 1: Research (with contract)
            if self.verbose:
                print(f"  [Researcher] Researching: {topic.title}")

            # Generate budget-aware prompt with specific token information
            research_task = f"""Research the following topic thoroughly:

Topic: {topic.title}
Description: {topic.description}

Key aspects to cover:
{chr(10).join(f"- {aspect}" for aspect in topic.key_aspects)}

Find current information, facts, statistics, and expert opinions.
Cite your sources with URLs."""

            research_prompt = generate_budget_prompt(
                contract=researcher.contract,
                task_description=research_task,
                current_usage=None,  # Starting fresh
            )

            research_output = researcher.run(
                user_id="eval_user",
                session_id=f"eval_researcher_{topic.id}",
                message=research_prompt,
            )
            result.raw_outputs["researcher"] = research_output["response"]
            result.tokens_by_agent["researcher"] = research_output["total_tokens"]

            # Phase 2: Analysis (with contract)
            if self.verbose:
                print("  [Analyzer] Analyzing findings...")

            # Generate budget-aware prompt for analyzer
            analysis_task = f"""Analyze the following research findings:

{research_output["response"]}

Identify:
1. Key patterns and trends
2. Important insights
3. Connections between sources
4. Most significant findings

Structure your analysis into clear themes."""

            analysis_prompt = generate_budget_prompt(
                contract=analyzer.contract,
                task_description=analysis_task,
                current_usage=None,  # Starting fresh
            )

            analysis_output = analyzer.run(
                user_id="eval_user",
                session_id=f"eval_analyzer_{topic.id}",
                message=analysis_prompt,
            )
            result.raw_outputs["analyzer"] = analysis_output["response"]
            result.tokens_by_agent["analyzer"] = analysis_output["total_tokens"]

            # Phase 3: Report (with contract)
            if self.verbose:
                print("  [Reporter] Writing report...")

            # Generate budget-aware prompt for reporter
            report_task = f"""Write a comprehensive research report on:

Topic: {topic.title}

Based on this analysis:
{analysis_output["response"]}

Requirements:
- At least 2,000 words
- Include Introduction, Main Body (with sections), and Conclusion
- Include at least 5 citations with URLs
- Be professional and well-organized"""

            report_prompt = generate_budget_prompt(
                contract=reporter.contract,
                task_description=report_task,
                current_usage=None,  # Starting fresh
            )

            report_output = reporter.run(
                user_id="eval_user",
                session_id=f"eval_reporter_{topic.id}",
                message=report_prompt,
            )
            result.raw_outputs["reporter"] = report_output["response"]
            result.tokens_by_agent["reporter"] = report_output["total_tokens"]

            # Compile results
            result.report = report_output["response"]
            result.word_count = len(result.report.split())
            result.citation_count = self._count_citations(result.report)
            result.total_tokens = sum(result.tokens_by_agent.values())
            result.success = True

            # Check budget compliance
            result.budget_compliant = result.total_tokens <= self.PARENT_TOKENS

            # Get final delegation summary
            final_summary = delegating.get_delegation_summary()
            result.conservation_violations = 0 if final_summary["conservation_satisfied"] else 1

        except RuntimeError as e:
            if "Contract violated" in str(e) or "Conservation" in str(e):
                result.error = f"Budget violation: {e}"
                result.budget_compliant = False
            else:
                result.error = str(e)
            result.success = False

        except Exception as e:
            result.error = str(e)
            result.success = False

        result.execution_time_seconds = time.time() - start_time
        return result

    def _count_citations(self, text: str) -> int:
        """Count citations in text (URLs or [n] references)."""
        import re

        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, text)

        ref_pattern = r"\[\d+\]"
        refs = re.findall(ref_pattern, text)

        return len(set(urls)) + len(set(refs))
