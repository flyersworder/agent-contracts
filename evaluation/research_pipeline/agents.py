"""Multi-agent research pipeline agents for COINE 2026 evaluation.

This module defines the specialized agents for the research report generation
pipeline:
- Researcher: Web search and data gathering
- Analyzer: Pattern identification and insight generation
- Reporter: Synthesis and report writing

Budget Allocation (from SUBMISSION_PLAN.md):
- Parent Contract: 100,000 tokens total
- Orchestrator: 10,000 tokens (coordination)
- Researcher: 40,000 tokens (web search, data gathering)
- Analyzer: 25,000 tokens (pattern identification)
- Reporter: 25,000 tokens (synthesis, writing)
"""

from dataclasses import dataclass, field
from typing import Any

# Type checking imports
try:
    from google.adk.agents import LlmAgent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models import LlmResponse
    from google.adk.planners import BuiltInPlanner
    from google.adk.tools import google_search
    from google.genai.types import ThinkingConfig

    GOOGLE_ADK_AVAILABLE = True
except ImportError:
    GOOGLE_ADK_AVAILABLE = False
    LlmAgent = Any
    CallbackContext = Any
    LlmResponse = Any
    BuiltInPlanner = Any
    ThinkingConfig = Any
    google_search = None


# Session state key for grounding metadata
GROUNDING_STATE_KEY = "grounding_metadata"


@dataclass
class GroundingTracker:
    """Tracks grounding metadata (web searches) from Gemini model calls.

    The google_search tool in ADK is a "grounding tool" - it's built into
    the model and doesn't appear as a function call. Instead, we track it
    via the grounding_metadata in the LlmResponse.

    Attributes:
        web_search_queries: List of search queries performed
        grounding_chunks: List of source URLs/titles used
        search_count: Total number of web searches performed
    """

    web_search_queries: list[str] = field(default_factory=list)
    grounding_chunks: list[dict[str, str]] = field(default_factory=list)
    search_count: int = 0

    def add_from_response(self, grounding_metadata: Any) -> None:
        """Extract and store grounding data from an LlmResponse.

        Args:
            grounding_metadata: The grounding_metadata from LlmResponse
        """
        if grounding_metadata is None:
            return

        # Extract web search queries
        if (
            hasattr(grounding_metadata, "web_search_queries")
            and grounding_metadata.web_search_queries
        ):
            for query in grounding_metadata.web_search_queries:
                if query not in self.web_search_queries:
                    self.web_search_queries.append(query)
                    self.search_count += 1

        # Extract grounding chunks (source URLs)
        if hasattr(grounding_metadata, "grounding_chunks") and grounding_metadata.grounding_chunks:
            for chunk in grounding_metadata.grounding_chunks:
                if hasattr(chunk, "web") and chunk.web:
                    chunk_info = {
                        "uri": getattr(chunk.web, "uri", ""),
                        "title": getattr(chunk.web, "title", ""),
                        "domain": getattr(chunk.web, "domain", ""),
                    }
                    if chunk_info not in self.grounding_chunks:
                        self.grounding_chunks.append(chunk_info)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "web_search_queries": self.web_search_queries,
            "grounding_chunks": self.grounding_chunks,
            "search_count": self.search_count,
        }


def create_grounding_callback() -> tuple[Any, GroundingTracker]:
    """Create an after_model_callback that tracks grounding metadata.

    Returns:
        Tuple of (callback_function, tracker) where tracker accumulates
        grounding data across all model calls.
    """
    tracker = GroundingTracker()

    def after_model_callback(
        callback_context: "CallbackContext",
        llm_response: "LlmResponse",
    ) -> "LlmResponse | None":
        """Capture grounding metadata from each model response.

        This callback is invoked after each LLM call, allowing us to
        inspect the grounding_metadata which contains web search information.
        """
        if llm_response is not None and hasattr(llm_response, "grounding_metadata"):
            tracker.add_from_response(llm_response.grounding_metadata)
        # Return None to pass through the original response unchanged
        return None

    return after_model_callback, tracker


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for a pipeline agent.

    Attributes:
        name: Agent name (used in contract ID)
        model: LLM model to use
        instruction: Agent instruction/system prompt
        token_budget: Token budget allocation
        description: Task description for the agent
        thinking_budget: Token budget for reasoning/thinking (0 = disabled)
    """

    name: str
    model: str
    instruction: str
    token_budget: int
    description: str
    thinking_budget: int = 0  # Default: no thinking (model default)


# Thinking budget for balanced reasoning (Flash-Lite has thinking off by default)
# Medium budget: enough for structured reasoning without excessive overhead
THINKING_BUDGET_MEDIUM = 1024  # ~1K tokens for reasoning


# Agent configurations matching SUBMISSION_PLAN.md
ORCHESTRATOR_CONFIG = AgentConfig(
    name="orchestrator",
    model="gemini-2.5-flash-lite",
    instruction="""You are a research report orchestrator. Your job is to:
1. Understand the research topic
2. Delegate tasks to specialized agents (researcher, analyzer, reporter)
3. Coordinate the workflow and ensure quality
4. Validate final output meets success criteria

Be efficient with your coordination. Focus on high-level guidance.""",
    token_budget=10_000,
    description="Coordinate research workflow and validate output",
    thinking_budget=512,  # Light thinking for coordination
)

RESEARCHER_CONFIG = AgentConfig(
    name="researcher",
    model="gemini-2.5-flash-lite",
    instruction="""You are a research specialist. Your job is to:
1. Search for relevant information on the given topic
2. Find factual data, statistics, and expert opinions
3. Identify key sources and citations
4. Compile raw research findings

IMPORTANT: Be strategic with web searches. Each search costs resources.
- Use your existing knowledge first for foundational context
- Search only for CURRENT data, recent developments, or specific facts you don't know
- Combine related queries into single, well-crafted searches
- Aim for quality sources over quantity of searches

Cite your sources with URLs when possible.""",
    token_budget=40_000,
    description="Web search and data gathering",
    thinking_budget=THINKING_BUDGET_MEDIUM,  # Reasoning helps plan search strategy
)

ANALYZER_CONFIG = AgentConfig(
    name="analyzer",
    model="gemini-2.5-flash-lite",
    instruction="""You are a research analyst. Your job is to:
1. Analyze the research findings provided
2. Identify patterns, trends, and key insights
3. Extract the most important information
4. Structure the analysis into clear themes

Focus on INSIGHTS and PATTERNS. Connect dots between different sources.
Highlight what's most important for the final report.""",
    token_budget=25_000,
    description="Pattern identification and insight generation",
    thinking_budget=THINKING_BUDGET_MEDIUM,  # Reasoning helps identify patterns
)

REPORTER_CONFIG = AgentConfig(
    name="reporter",
    model="gemini-2.5-flash-lite",
    instruction="""You are a report writer. Your job is to:
1. Synthesize the analysis into a coherent report
2. Write clear, professional prose
3. Structure the report with sections: Introduction, Main Body, Conclusion
4. Include citations and references

The report should be:
- At least 2,000 words
- Include at least 5 citations
- Cover all key aspects of the topic
- Be well-organized with clear sections""",
    token_budget=25_000,
    description="Report synthesis and writing",
    thinking_budget=THINKING_BUDGET_MEDIUM,  # Reasoning helps structure narrative
)

# All agent configurations
AGENT_CONFIGS = {
    "orchestrator": ORCHESTRATOR_CONFIG,
    "researcher": RESEARCHER_CONFIG,
    "analyzer": ANALYZER_CONFIG,
    "reporter": REPORTER_CONFIG,
}


def _create_thinking_planner(thinking_budget: int) -> "BuiltInPlanner | None":
    """Create a BuiltInPlanner with thinking configuration.

    Args:
        thinking_budget: Token budget for thinking/reasoning.
            If 0, returns None (use model default behavior).

    Returns:
        BuiltInPlanner with ThinkingConfig, or None if thinking disabled
    """
    if thinking_budget <= 0:
        return None

    if not GOOGLE_ADK_AVAILABLE:
        return None

    return BuiltInPlanner(
        thinking_config=ThinkingConfig(
            include_thoughts=True,
            thinking_budget=thinking_budget,
        )
    )


def create_researcher_agent(
    grounding_callback: Any | None = None,
) -> "LlmAgent":
    """Create a researcher agent for data gathering with Google Search.

    The researcher uses the google_search tool to find current information,
    facts, statistics, and expert opinions on research topics.

    Args:
        grounding_callback: Optional after_model_callback to track grounding
            metadata (web searches). Create with create_grounding_callback().

    Returns:
        LlmAgent configured for research tasks with web search capability

    Raises:
        ImportError: If google-adk is not installed
    """
    if not GOOGLE_ADK_AVAILABLE:
        raise ImportError("google-adk is required. Install with: uv sync --extra google-adk")

    kwargs: dict[str, Any] = {
        "name": RESEARCHER_CONFIG.name,
        "model": RESEARCHER_CONFIG.model,
        "instruction": RESEARCHER_CONFIG.instruction,
        "tools": [google_search],  # Enable web search for current information
    }

    # Add thinking planner if configured
    planner = _create_thinking_planner(RESEARCHER_CONFIG.thinking_budget)
    if planner is not None:
        kwargs["planner"] = planner

    # Add grounding callback if provided
    if grounding_callback is not None:
        kwargs["after_model_callback"] = grounding_callback

    return LlmAgent(**kwargs)


def create_analyzer_agent() -> "LlmAgent":
    """Create an analyzer agent for pattern identification.

    Returns:
        LlmAgent configured for analysis tasks

    Raises:
        ImportError: If google-adk is not installed
    """
    if not GOOGLE_ADK_AVAILABLE:
        raise ImportError("google-adk is required. Install with: uv sync --extra google-adk")

    kwargs: dict[str, Any] = {
        "name": ANALYZER_CONFIG.name,
        "model": ANALYZER_CONFIG.model,
        "instruction": ANALYZER_CONFIG.instruction,
    }

    # Add thinking planner if configured
    planner = _create_thinking_planner(ANALYZER_CONFIG.thinking_budget)
    if planner is not None:
        kwargs["planner"] = planner

    return LlmAgent(**kwargs)


def create_reporter_agent() -> "LlmAgent":
    """Create a reporter agent for synthesis and writing.

    Returns:
        LlmAgent configured for report writing

    Raises:
        ImportError: If google-adk is not installed
    """
    if not GOOGLE_ADK_AVAILABLE:
        raise ImportError("google-adk is required. Install with: uv sync --extra google-adk")

    kwargs: dict[str, Any] = {
        "name": REPORTER_CONFIG.name,
        "model": REPORTER_CONFIG.model,
        "instruction": REPORTER_CONFIG.instruction,
    }

    # Add thinking planner if configured
    planner = _create_thinking_planner(REPORTER_CONFIG.thinking_budget)
    if planner is not None:
        kwargs["planner"] = planner

    return LlmAgent(**kwargs)


def create_orchestrator_agent(
    sub_agents: list["LlmAgent"] | None = None,
) -> "LlmAgent":
    """Create an orchestrator agent that coordinates the pipeline.

    Args:
        sub_agents: Optional list of sub-agents to coordinate

    Returns:
        LlmAgent configured as orchestrator

    Raises:
        ImportError: If google-adk is not installed
    """
    if not GOOGLE_ADK_AVAILABLE:
        raise ImportError("google-adk is required. Install with: uv sync --extra google-adk")

    kwargs: dict[str, Any] = {
        "name": ORCHESTRATOR_CONFIG.name,
        "model": ORCHESTRATOR_CONFIG.model,
        "instruction": ORCHESTRATOR_CONFIG.instruction,
        "sub_agents": sub_agents or [],
    }

    # Add thinking planner if configured
    planner = _create_thinking_planner(ORCHESTRATOR_CONFIG.thinking_budget)
    if planner is not None:
        kwargs["planner"] = planner

    return LlmAgent(**kwargs)


def create_all_agents() -> dict[str, "LlmAgent"]:
    """Create all pipeline agents.

    Returns:
        Dictionary mapping agent names to LlmAgent instances

    Raises:
        ImportError: If google-adk is not installed
    """
    researcher = create_researcher_agent()
    analyzer = create_analyzer_agent()
    reporter = create_reporter_agent()
    orchestrator = create_orchestrator_agent(sub_agents=[researcher, analyzer, reporter])

    return {
        "orchestrator": orchestrator,
        "researcher": researcher,
        "analyzer": analyzer,
        "reporter": reporter,
    }
