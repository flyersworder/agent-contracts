"""SkillSpec: Agent skill specification following the agentskills.io open standard.

This module implements the SkillSpec dataclass for defining reusable agent skills
with progressive disclosure, YAML frontmatter serialization, and name validation
per the agentskills.io specification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Regex pattern for valid skill names (agentskills.io specification).
# Structure: must start with a letter, lowercase alphanumeric + single
# hyphens, no consecutive hyphens, no leading/trailing hyphen. The 1-64
# character length bound is checked separately in __post_init__.
_SKILL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SkillSpec:
    """Agent Skill specification following the agentskills.io open standard.

    Skills are reusable instruction packages that provide domain-specific expertise.
    They follow the progressive disclosure pattern:
    - Metadata (name, description) loads at startup for discovery
    - Full instructions load only when the skill is activated
    - References and scripts load on-demand as needed

    This implementation is compatible with the open standard adopted by:
    Microsoft (VS Code, GitHub), OpenAI, Cursor, Goose, Amp, and others.

    File Structure (when exported to filesystem):
        my-skill/
        ├── SKILL.md        # Required: frontmatter + instructions
        ├── reference.md    # Optional: detailed documentation
        ├── scripts/        # Optional: helper scripts
        └── assets/         # Optional: templates, images

    Attributes:
        name: Unique identifier (1-64 chars, lowercase alphanumeric + hyphens)
        description: What the skill does and when to use it (1-1024 chars)
        instructions: Main skill content (step-by-step guidance, examples)
        version: Semantic version string (default: "1.0.0")
        license: License name or reference (optional)
        allowed_tools: Tools the skill can use (optional, restricts access)
        references: Additional documentation files (name -> content)
        scripts: Helper scripts (name -> content or path)
        metadata: Custom key-value pairs for extensibility

    Example:
        >>> skill = SkillSpec(
        ...     name="code-reviewer",
        ...     description="Review code for best practices, security issues, "
        ...                 "and test coverage. Use when reviewing PRs or "
        ...                 "analyzing code quality.",
        ...     instructions='''
        ...     ## Instructions
        ...     1. Read the target files
        ...     2. Check for common issues:
        ...        - Error handling
        ...        - Security vulnerabilities
        ...        - Test coverage
        ...     3. Provide detailed feedback
        ...     ''',
        ...     allowed_tools=["Read", "Grep", "Glob"],
        ... )
        >>>
        >>> # Use in Capabilities
        >>> caps = Capabilities(
        ...     skills=[skill, "simple-skill-name"],  # Mix of SkillSpec and strings
        ...     tools=["web_search"],
        ... )
    """

    # Required fields (agentskills.io specification)
    name: str
    description: str

    # Main content
    instructions: str = ""

    # Optional metadata fields
    version: str = "1.0.0"
    license: str | None = None
    allowed_tools: list[str] | None = None

    # Progressive disclosure content (loaded on-demand)
    references: dict[str, str] = field(default_factory=dict)
    scripts: dict[str, str] = field(default_factory=dict)

    # Extensibility
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate skill specification against agentskills.io standard."""
        # Validate name format and length (1-64 chars per agentskills.io)
        if not (1 <= len(self.name) <= 64) or not _SKILL_NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Skill name must be 1-64 lowercase alphanumeric characters with hyphens, "
                f"cannot start/end with hyphen or have consecutive hyphens. Got: '{self.name}'"
            )

        # Validate description length
        if len(self.description) == 0:
            raise ValueError("Skill description cannot be empty")
        if len(self.description) > 1024:
            raise ValueError(
                f"Skill description must be ≤1024 characters, got {len(self.description)}"
            )

    def to_frontmatter(self) -> str:
        """Generate YAML frontmatter for SKILL.md export.

        Returns:
            YAML frontmatter string (without --- delimiters)
        """
        lines = [
            f"name: {self.name}",
            f"description: {self.description}",
        ]

        if self.version != "1.0.0":
            lines.append(f"version: {self.version}")

        if self.license:
            lines.append(f"license: {self.license}")

        if self.allowed_tools:
            lines.append(f"allowed-tools: {' '.join(self.allowed_tools)}")

        if self.metadata:
            lines.append("metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def to_skill_md(self) -> str:
        """Generate complete SKILL.md content.

        Returns:
            Full SKILL.md file content with frontmatter and instructions
        """
        return f"---\n{self.to_frontmatter()}\n---\n\n{self.instructions}"

    @classmethod
    def from_skill_md(cls, content: str, name: str | None = None) -> SkillSpec:
        """Parse SKILL.md content into SkillSpec.

        Args:
            content: Raw SKILL.md file content
            name: Override name (if not in frontmatter or for validation)

        Returns:
            Parsed SkillSpec instance

        Raises:
            ValueError: If content is invalid or missing required fields
        """
        # Split frontmatter and content
        if not content.startswith("---"):
            raise ValueError("SKILL.md must start with YAML frontmatter (---)")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError("SKILL.md must have closing --- for frontmatter")

        frontmatter_text = parts[1].strip()
        instructions = parts[2].strip()

        # Parse YAML frontmatter (simple parser for common cases)
        frontmatter: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        in_metadata = False

        for line in frontmatter_text.split("\n"):
            line = line.rstrip()
            if not line:
                continue

            if line.startswith("  ") and in_metadata:
                # Metadata value
                key_val = line.strip().split(":", 1)
                if len(key_val) == 2:
                    metadata[key_val[0].strip()] = key_val[1].strip()
            elif ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key == "metadata":
                    in_metadata = True
                else:
                    in_metadata = False
                    frontmatter[key] = value

        # Extract required fields
        skill_name = name or frontmatter.get("name")
        if not skill_name:
            raise ValueError("SKILL.md missing required 'name' field")

        description = frontmatter.get("description")
        if not description:
            raise ValueError("SKILL.md missing required 'description' field")

        # Extract optional fields
        allowed_tools = None
        if "allowed-tools" in frontmatter:
            allowed_tools = frontmatter["allowed-tools"].split()

        return cls(
            name=skill_name,
            description=description,
            instructions=instructions,
            version=frontmatter.get("version", "1.0.0"),
            license=frontmatter.get("license"),
            allowed_tools=allowed_tools,
            metadata=metadata if metadata else {},
        )

    @property
    def metadata_tokens(self) -> int:
        """Estimate tokens for metadata (name + description).

        Used for progressive disclosure budgeting.
        """
        # Rough estimate: ~4 chars per token
        return (len(self.name) + len(self.description)) // 4

    @property
    def instruction_tokens(self) -> int:
        """Estimate tokens for full instructions.

        Used for progressive disclosure budgeting.
        """
        return len(self.instructions) // 4
