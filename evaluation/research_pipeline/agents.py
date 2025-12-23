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

from dataclasses import dataclass
from typing import Any

# Type checking imports
try:
    from google.adk.agents import LlmAgent
    from google.adk.tools import google_search

    GOOGLE_ADK_AVAILABLE = True
except ImportError:
    GOOGLE_ADK_AVAILABLE = False
    LlmAgent = Any
    google_search = None


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for a pipeline agent.

    Attributes:
        name: Agent name (used in contract ID)
        model: LLM model to use
        instruction: Agent instruction/system prompt
        token_budget: Token budget allocation
        description: Task description for the agent
    """

    name: str
    model: str
    instruction: str
    token_budget: int
    description: str


# Agent configurations matching SUBMISSION_PLAN.md
ORCHESTRATOR_CONFIG = AgentConfig(
    name="orchestrator",
    model="gemini-2.5-flash",
    instruction="""You are a research report orchestrator. Your job is to:
1. Understand the research topic
2. Delegate tasks to specialized agents (researcher, analyzer, reporter)
3. Coordinate the workflow and ensure quality
4. Validate final output meets success criteria

Be efficient with your coordination. Focus on high-level guidance.""",
    token_budget=10_000,
    description="Coordinate research workflow and validate output",
)

RESEARCHER_CONFIG = AgentConfig(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="""You are a research specialist. Your job is to:
1. Search for relevant information on the given topic
2. Find factual data, statistics, and expert opinions
3. Identify key sources and citations
4. Compile raw research findings

Focus on gathering FACTS and DATA. Use web search to find current information.
Cite your sources with URLs when possible.""",
    token_budget=40_000,
    description="Web search and data gathering",
)

ANALYZER_CONFIG = AgentConfig(
    name="analyzer",
    model="gemini-2.5-flash",
    instruction="""You are a research analyst. Your job is to:
1. Analyze the research findings provided
2. Identify patterns, trends, and key insights
3. Extract the most important information
4. Structure the analysis into clear themes

Focus on INSIGHTS and PATTERNS. Connect dots between different sources.
Highlight what's most important for the final report.""",
    token_budget=25_000,
    description="Pattern identification and insight generation",
)

REPORTER_CONFIG = AgentConfig(
    name="reporter",
    model="gemini-2.5-flash",
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
)

# All agent configurations
AGENT_CONFIGS = {
    "orchestrator": ORCHESTRATOR_CONFIG,
    "researcher": RESEARCHER_CONFIG,
    "analyzer": ANALYZER_CONFIG,
    "reporter": REPORTER_CONFIG,
}


def create_researcher_agent() -> "LlmAgent":
    """Create a researcher agent for data gathering with Google Search.

    The researcher uses the google_search tool to find current information,
    facts, statistics, and expert opinions on research topics.

    Returns:
        LlmAgent configured for research tasks with web search capability

    Raises:
        ImportError: If google-adk is not installed
    """
    if not GOOGLE_ADK_AVAILABLE:
        raise ImportError("google-adk is required. Install with: uv sync --extra google-adk")

    return LlmAgent(
        name=RESEARCHER_CONFIG.name,
        model=RESEARCHER_CONFIG.model,
        instruction=RESEARCHER_CONFIG.instruction,
        tools=[google_search],  # Enable web search for current information
    )


def create_analyzer_agent() -> "LlmAgent":
    """Create an analyzer agent for pattern identification.

    Returns:
        LlmAgent configured for analysis tasks

    Raises:
        ImportError: If google-adk is not installed
    """
    if not GOOGLE_ADK_AVAILABLE:
        raise ImportError("google-adk is required. Install with: uv sync --extra google-adk")

    return LlmAgent(
        name=ANALYZER_CONFIG.name,
        model=ANALYZER_CONFIG.model,
        instruction=ANALYZER_CONFIG.instruction,
    )


def create_reporter_agent() -> "LlmAgent":
    """Create a reporter agent for synthesis and writing.

    Returns:
        LlmAgent configured for report writing

    Raises:
        ImportError: If google-adk is not installed
    """
    if not GOOGLE_ADK_AVAILABLE:
        raise ImportError("google-adk is required. Install with: uv sync --extra google-adk")

    return LlmAgent(
        name=REPORTER_CONFIG.name,
        model=REPORTER_CONFIG.model,
        instruction=REPORTER_CONFIG.instruction,
    )


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

    return LlmAgent(
        name=ORCHESTRATOR_CONFIG.name,
        model=ORCHESTRATOR_CONFIG.model,
        instruction=ORCHESTRATOR_CONFIG.instruction,
        sub_agents=sub_agents or [],
    )


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
