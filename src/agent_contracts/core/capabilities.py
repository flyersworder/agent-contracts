"""Agent capabilities, execution configuration, and agent specifications.

This module contains the model-agnostic capability definitions and
model-specific execution configuration for Agent Contracts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent_contracts.core.skillspec import SkillSpec

logger = logging.getLogger(__name__)


@dataclass
class AgentSpec:
    """Model-agnostic specification for an agent in multi-agent scenarios.

    Defines what a sub-agent can do, independent of which LLM runs it.
    The model choice is made at execution time via ExecutionConfig.

    Attributes:
        name: Unique identifier for this agent
        role: Description of agent's role in the workflow
        tools: Tools available to this agent
        skills: Skills this agent can use
        instructions: Instructions for this agent (model-agnostic)
    """

    name: str
    role: str = ""
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    instructions: str = ""


class CoordinationPattern(Enum):
    """Patterns for multi-agent coordination.

    Defines how multiple agents work together to complete a contract.
    """

    SEQUENTIAL = "sequential"  # Agents execute one after another
    PARALLEL = "parallel"  # Agents execute simultaneously
    HIERARCHICAL = "hierarchical"  # Parent delegates to children
    COLLABORATIVE = "collaborative"  # Agents negotiate and collaborate
    COMPETITIVE = "competitive"  # Agents compete, best result wins


@dataclass
class ExecutionConfig:
    """Model-specific execution configuration.

    This class contains LLM-specific settings that control how the contract
    is executed. These are separate from Capabilities because they are
    implementation details, not part of what the agent can do.

    Attributes:
        model: LLM model to use (e.g., "gpt-4o", "claude-sonnet-4-20250514")
        provider: LLM provider (auto-detected if not specified)
        temperature: LLM temperature setting (0-2)
        max_retries: Maximum retry attempts on failure
        timeout_seconds: Maximum execution time per LLM call

    Example:
        >>> config = ExecutionConfig(
        ...     model="gpt-4o",
        ...     temperature=0.7,
        ...     max_retries=3
        ... )
    """

    model: str = "gpt-4o"
    provider: str | None = None
    temperature: float = 0.7
    max_retries: int = 3
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate execution configuration."""
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")

        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")

        if self.provider is None:
            self.provider = self._detect_provider()

    def _detect_provider(self) -> str:
        """Auto-detect LLM provider from model name."""
        model_lower = self.model.lower()

        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        elif "llama" in model_lower:
            return "meta"
        elif "deepseek" in model_lower:
            return "deepseek"
        else:
            return "unknown"


@dataclass
class Capabilities:
    """Model-agnostic agent capabilities specification (extended S in whitepaper).

    This class defines WHAT an agent can access and do, independent of which
    LLM executes it. Capabilities are portable across different models because
    they use standardized interfaces:

    - **MCP (Model Context Protocol)**: Standard for exposing tools/resources
    - **Agent Skills**: Reusable, composable agent behaviors (agentskills.io standard)
    - **Tools**: Function-calling interfaces (standardized across LLMs)

    The separation from ExecutionConfig means you can:
    1. Define capabilities once
    2. Execute with different models (GPT-4, Claude, Gemini, etc.)
    3. Share capability definitions across teams/projects

    Skills can be specified as:
    - Simple strings: `["code-review", "research"]` for basic skill references
    - Rich SkillSpec objects: Full skill definitions with instructions

    Attributes:
        tools: External tools available (function names, MCP tool URIs, etc.)
        mcp_servers: MCP server URIs for extended capabilities
        skills: Agent skills - strings for simple references, SkillSpec for rich definitions
        resources: Named resources the agent can access (files, APIs, databases)
        agents: Sub-agents for multi-agent scenarios
        coordination: Pattern for multi-agent coordination
        instructions: Base instructions for the agent (model-agnostic)

    Example:
        >>> # Simple skills (backward compatible)
        >>> caps = Capabilities(
        ...     tools=["web_search", "calculator"],
        ...     skills=["code-review", "research"],
        ... )
        >>>
        >>> # Rich skill definitions (agentskills.io standard)
        >>> code_review = SkillSpec(
        ...     name="code-reviewer",
        ...     description="Review code for best practices and security",
        ...     instructions="1. Read the code\\n2. Check for issues\\n3. Report findings",
        ...     allowed_tools=["Read", "Grep"],
        ... )
        >>> caps = Capabilities(
        ...     skills=[code_review, "simple-skill"],  # Mix strings and SkillSpec
        ... )
        >>>
        >>> # Lookup skills by name
        >>> skill = caps.get_skill("code-reviewer")
        >>> if skill:
        ...     print(skill.instructions)
        >>>
        >>> # Get all skill names (for discovery)
        >>> print(caps.skill_names)  # ["code-reviewer", "simple-skill"]
        >>>
        >>> # Multi-agent workflow
        >>> caps = Capabilities(
        ...     agents={
        ...         "researcher": AgentSpec(
        ...             name="researcher",
        ...             role="Gathers information from sources",
        ...             tools=["web_search", "document_reader"]
        ...         ),
        ...         "analyst": AgentSpec(
        ...             name="analyst",
        ...             role="Analyzes and synthesizes findings"
        ...         ),
        ...     },
        ...     coordination=CoordinationPattern.SEQUENTIAL
        ... )
    """

    tools: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    skills: list[str | SkillSpec] = field(default_factory=list)
    resources: dict[str, str] = field(default_factory=dict)  # name -> URI/path
    agents: dict[str, AgentSpec] | None = None
    coordination: CoordinationPattern | None = None
    instructions: str | None = None

    def __post_init__(self) -> None:
        """Validate capabilities configuration."""
        # Auto-set coordination pattern for multi-agent configs
        if self.agents and not self.coordination:
            object.__setattr__(self, "coordination", CoordinationPattern.SEQUENTIAL)

    @property
    def is_multi_agent(self) -> bool:
        """Check if this is a multi-agent configuration."""
        return self.agents is not None and len(self.agents) > 0

    @property
    def agent_count(self) -> int:
        """Get the number of agents (1 for single-agent, N for multi-agent)."""
        if self.agents:
            return len(self.agents)
        return 1

    # --- Skill helper methods ---

    @property
    def skill_names(self) -> list[str]:
        """Get all skill names for discovery/display.

        Returns:
            List of skill names (strings), extracting names from SkillSpec objects.
        """
        return [skill.name if isinstance(skill, SkillSpec) else skill for skill in self.skills]

    @property
    def skill_specs(self) -> list[SkillSpec]:
        """Get only the SkillSpec objects (not string references).

        Returns:
            List of SkillSpec objects with full skill definitions.
        """
        return [skill for skill in self.skills if isinstance(skill, SkillSpec)]

    def get_skill(self, name: str) -> SkillSpec | None:
        """Look up a skill by name.

        Args:
            name: The skill name to look up

        Returns:
            SkillSpec if found and is a rich definition, None otherwise.
            Note: Returns None for string-only skill references.
        """
        for skill in self.skills:
            if isinstance(skill, SkillSpec) and skill.name == name:
                return skill
        return None

    def has_skill(self, name: str) -> bool:
        """Check if a skill exists (by name or SkillSpec).

        Args:
            name: The skill name to check

        Returns:
            True if skill exists (as string or SkillSpec), False otherwise.
        """
        for skill in self.skills:
            if isinstance(skill, SkillSpec):
                if skill.name == name:
                    return True
            elif skill == name:
                return True
        return False

    @property
    def total_metadata_tokens(self) -> int:
        """Estimate total tokens for skill metadata (for progressive disclosure).

        This is the cost of loading skill names and descriptions for discovery,
        before any skill is activated.
        """
        total = 0
        for skill in self.skills:
            if isinstance(skill, SkillSpec):
                total += skill.metadata_tokens
            else:
                # String skills: just the name (~2-3 tokens)
                total += len(skill) // 4 + 1
        return total

    @property
    def total_instruction_tokens(self) -> int:
        """Estimate total tokens for all skill instructions.

        This would be the cost of loading ALL skills at once.
        Progressive disclosure loads instructions only when activated.
        """
        return sum(
            skill.instruction_tokens for skill in self.skills if isinstance(skill, SkillSpec)
        )

    # --- LiteLLM Integration Methods ---

    def to_mcp_tools(self) -> list[dict[str, Any]]:
        """Convert MCP servers to LiteLLM MCP tool format.

        LiteLLM supports MCP tools via the `tools` parameter with type="mcp".
        This method converts our mcp_servers list to the format LiteLLM expects.

        Returns:
            List of LiteLLM MCP tool definitions

        Example:
            >>> caps = Capabilities(mcp_servers=["https://mcp.example.com/api"])
            >>> caps.to_mcp_tools()
            [{"type": "mcp", "server_label": "mcp-server-0",
              "server_url": "https://mcp.example.com/api", "require_approval": "never"}]
        """
        mcp_tools = []
        for i, server_url in enumerate(self.mcp_servers):
            # Extract a label from the URL or use index-based name
            label = f"mcp-server-{i}"
            if "://" in server_url:
                # Try to extract domain as label
                try:
                    from urllib.parse import urlparse

                    parsed = urlparse(server_url)
                    if parsed.netloc:
                        # Use domain without TLD as label (e.g., "mcp" from "mcp.example.com")
                        label = parsed.netloc.split(".")[0]
                except Exception:
                    logger.debug("Failed to parse MCP server URL for label", exc_info=True)

            mcp_tools.append(
                {
                    "type": "mcp",
                    "server_label": label,
                    "server_url": server_url,
                    "require_approval": "never",  # Default to no approval
                }
            )
        return mcp_tools

    def to_litellm_tools(
        self,
        tool_definitions: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Get all tools in LiteLLM-compatible format.

        Combines:
        1. Tool definitions (if provided) - for tools listed in self.tools
        2. MCP server tools

        Args:
            tool_definitions: Optional dict mapping tool names to their full definitions.
                Each definition should follow the OpenAI function calling format:
                {"type": "function", "function": {"name": ..., "parameters": ...}}

        Returns:
            List of LiteLLM-compatible tool definitions

        Example:
            >>> caps = Capabilities(
            ...     tools=["get_weather", "search"],
            ...     mcp_servers=["https://mcp.example.com/api"]
            ... )
            >>> definitions = {
            ...     "get_weather": {
            ...         "type": "function",
            ...         "function": {
            ...             "name": "get_weather",
            ...             "description": "Get weather for a location",
            ...             "parameters": {"type": "object", "properties": {...}}
            ...         }
            ...     }
            ... }
            >>> tools = caps.to_litellm_tools(tool_definitions=definitions)
        """
        all_tools: list[dict[str, Any]] = []

        # Add function tools if definitions are provided
        if tool_definitions:
            for tool_name in self.tools:
                if tool_name in tool_definitions:
                    all_tools.append(tool_definitions[tool_name])

        # Add MCP tools
        all_tools.extend(self.to_mcp_tools())

        return all_tools

    def has_tools(self) -> bool:
        """Check if any tools or MCP servers are configured.

        Returns:
            True if tools or mcp_servers are defined
        """
        return len(self.tools) > 0 or len(self.mcp_servers) > 0

    def get_code_execution_tool(self) -> dict[str, Any] | None:
        """Get Anthropic code execution tool if applicable.

        This is a convenience method for Anthropic's code_execution capability.
        Only returns a tool definition if "code_execution" is in the tools list.

        Returns:
            Code execution tool definition for Anthropic, or None
        """
        if "code_execution" in self.tools:
            return {
                "type": "code_execution_20250825",
                "name": "code_execution",
            }
        return None

    def to_anthropic_skills(self) -> list[dict[str, Any]]:
        """Convert SkillSpecs to Anthropic container.skills format.

        Anthropic supports skills via the `container.skills` parameter.
        This method converts our SkillSpec objects to Anthropic's format.

        Note: This only works with Anthropic models. For other models,
        skill instructions should be included in the system prompt.

        Returns:
            List of Anthropic skill definitions (for container.skills parameter)

        Example:
            >>> skill = SkillSpec(name="code-reviewer", ...)
            >>> caps = Capabilities(skills=[skill])
            >>> caps.to_anthropic_skills()
            [{"type": "anthropic", "skill_id": "code-reviewer", "version": "latest"}]
        """
        anthropic_skills = []
        for skill in self.skills:
            if isinstance(skill, SkillSpec):
                anthropic_skills.append(
                    {
                        "type": "anthropic",
                        "skill_id": skill.name,
                        "version": skill.version or "latest",
                    }
                )
        return anthropic_skills

    def get_skill_instructions(self, active_skills: list[str] | None = None) -> str:
        """Get skill instructions for prompt injection.

        For models that don't support native skills, this method generates
        a formatted string of skill instructions to include in the system prompt.

        Args:
            active_skills: Optional list of skill names to include.
                If None, includes all skills with instructions.

        Returns:
            Formatted skill instructions string
        """
        instructions_parts = []

        for skill in self.skills:
            # Check if this skill should be included (is SkillSpec, matches filter, has instructions)
            if (
                isinstance(skill, SkillSpec)
                and (active_skills is None or skill.name in active_skills)
                and skill.instructions
            ):
                instructions_parts.append(
                    f"## Skill: {skill.name}\n"
                    f"Description: {skill.description or 'No description'}\n"
                    f"Instructions:\n{skill.instructions}"
                )

        if instructions_parts:
            return "\n\n".join(instructions_parts)
        return ""
