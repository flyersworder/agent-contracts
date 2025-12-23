"""Tests for Capabilities, AgentSpec, ExecutionConfig, SkillSpec, and related classes.

This module tests the model-agnostic capability specification system,
model-specific execution configuration, and the agentskills.io-compliant
SkillSpec implementation.
"""

import pytest

from agent_contracts.core.contract import (
    AgentSpec,
    Capabilities,
    Contract,
    CoordinationPattern,
    ExecutionConfig,
    ResourceConstraints,
    SkillSpec,
)


class TestAgentSpec:
    """Tests for AgentSpec dataclass."""

    def test_create_minimal_agent_spec(self) -> None:
        """Test creating AgentSpec with only required fields."""
        agent = AgentSpec(name="researcher")
        assert agent.name == "researcher"
        assert agent.role == ""
        assert agent.tools == []
        assert agent.skills == []
        assert agent.instructions == ""

    def test_create_full_agent_spec(self) -> None:
        """Test creating AgentSpec with all fields."""
        agent = AgentSpec(
            name="analyst",
            role="Analyzes data and provides insights",
            tools=["calculator", "data_visualizer"],
            skills=["data-analysis"],
            instructions="Focus on statistical accuracy",
        )
        assert agent.name == "analyst"
        assert agent.role == "Analyzes data and provides insights"
        assert agent.tools == ["calculator", "data_visualizer"]
        assert agent.skills == ["data-analysis"]
        assert agent.instructions == "Focus on statistical accuracy"


class TestCoordinationPattern:
    """Tests for CoordinationPattern enum."""

    def test_coordination_patterns_exist(self) -> None:
        """Test that all expected coordination patterns exist."""
        assert CoordinationPattern.SEQUENTIAL.value == "sequential"
        assert CoordinationPattern.PARALLEL.value == "parallel"
        assert CoordinationPattern.HIERARCHICAL.value == "hierarchical"
        assert CoordinationPattern.COLLABORATIVE.value == "collaborative"
        assert CoordinationPattern.COMPETITIVE.value == "competitive"


class TestExecutionConfig:
    """Tests for ExecutionConfig dataclass (model-specific settings)."""

    def test_create_minimal_execution_config(self) -> None:
        """Test creating ExecutionConfig with defaults."""
        config = ExecutionConfig()
        assert config.model == "gpt-4o"
        assert config.provider == "openai"  # auto-detected
        assert config.temperature == 0.7
        assert config.max_retries == 3
        assert config.timeout_seconds is None

    def test_create_execution_config_with_model(self) -> None:
        """Test creating ExecutionConfig with custom model."""
        config = ExecutionConfig(model="claude-sonnet-4-20250514", temperature=0.5)
        assert config.model == "claude-sonnet-4-20250514"
        assert config.provider == "anthropic"  # auto-detected
        assert config.temperature == 0.5

    def test_auto_detect_provider_openai(self) -> None:
        """Test auto-detection of OpenAI provider."""
        config = ExecutionConfig(model="gpt-4o")
        assert config.provider == "openai"

        config = ExecutionConfig(model="gpt-3.5-turbo")
        assert config.provider == "openai"

        config = ExecutionConfig(model="o1-preview")
        assert config.provider == "openai"

    def test_auto_detect_provider_anthropic(self) -> None:
        """Test auto-detection of Anthropic provider."""
        config = ExecutionConfig(model="claude-sonnet-4-20250514")
        assert config.provider == "anthropic"

        config = ExecutionConfig(model="claude-3-opus")
        assert config.provider == "anthropic"

    def test_auto_detect_provider_google(self) -> None:
        """Test auto-detection of Google provider."""
        config = ExecutionConfig(model="gemini-1.5-pro")
        assert config.provider == "google"

        config = ExecutionConfig(model="gemini-2.0-flash")
        assert config.provider == "google"

    def test_auto_detect_provider_mistral(self) -> None:
        """Test auto-detection of Mistral provider."""
        config = ExecutionConfig(model="mistral-large")
        assert config.provider == "mistral"

        config = ExecutionConfig(model="mixtral-8x7b")
        assert config.provider == "mistral"

    def test_auto_detect_provider_meta(self) -> None:
        """Test auto-detection of Meta provider."""
        config = ExecutionConfig(model="llama-3.1-70b")
        assert config.provider == "meta"

    def test_auto_detect_provider_deepseek(self) -> None:
        """Test auto-detection of DeepSeek provider."""
        config = ExecutionConfig(model="deepseek-coder")
        assert config.provider == "deepseek"

    def test_auto_detect_provider_unknown(self) -> None:
        """Test auto-detection with unknown model."""
        config = ExecutionConfig(model="some-custom-model")
        assert config.provider == "unknown"

    def test_explicit_provider_overrides_detection(self) -> None:
        """Test that explicit provider is not overwritten."""
        config = ExecutionConfig(model="gpt-4o", provider="custom-provider")
        assert config.provider == "custom-provider"

    def test_temperature_validation(self) -> None:
        """Test that temperature is validated."""
        with pytest.raises(ValueError, match="temperature must be in"):
            ExecutionConfig(temperature=-0.1)

        with pytest.raises(ValueError, match="temperature must be in"):
            ExecutionConfig(temperature=2.5)

    def test_temperature_boundary_values(self) -> None:
        """Test temperature at boundary values."""
        config = ExecutionConfig(temperature=0)
        assert config.temperature == 0

        config = ExecutionConfig(temperature=2)
        assert config.temperature == 2

    def test_max_retries_validation(self) -> None:
        """Test that max_retries is validated."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ExecutionConfig(max_retries=-1)

    def test_max_retries_zero_allowed(self) -> None:
        """Test that zero retries is allowed."""
        config = ExecutionConfig(max_retries=0)
        assert config.max_retries == 0


class TestCapabilities:
    """Tests for Capabilities dataclass (model-agnostic capabilities)."""

    def test_create_minimal_capabilities(self) -> None:
        """Test creating Capabilities with defaults (model-agnostic)."""
        caps = Capabilities()
        # Model-agnostic: only capability fields, no model settings
        assert caps.tools == []
        assert caps.mcp_servers == []
        assert caps.skills == []
        assert caps.resources == {}
        assert caps.agents is None
        assert caps.coordination is None
        assert caps.instructions is None

    def test_create_capabilities_with_tools(self) -> None:
        """Test creating Capabilities with tools."""
        caps = Capabilities(
            tools=["web_search", "calculator"],
        )
        assert caps.tools == ["web_search", "calculator"]

    def test_create_capabilities_with_mcp_servers(self) -> None:
        """Test creating Capabilities with MCP servers."""
        caps = Capabilities(
            mcp_servers=["filesystem", "github"],
        )
        assert caps.mcp_servers == ["filesystem", "github"]

    def test_create_capabilities_with_skills(self) -> None:
        """Test creating Capabilities with skills."""
        caps = Capabilities(
            skills=["code-review", "research"],
        )
        assert caps.skills == ["code-review", "research"]

    def test_create_capabilities_with_resources(self) -> None:
        """Test creating Capabilities with named resources."""
        caps = Capabilities(
            resources={
                "database": "postgres://localhost/mydb",
                "api": "https://api.example.com",
            },
        )
        assert caps.resources["database"] == "postgres://localhost/mydb"
        assert caps.resources["api"] == "https://api.example.com"

    def test_create_capabilities_with_instructions(self) -> None:
        """Test creating Capabilities with instructions."""
        caps = Capabilities(
            instructions="You are a helpful research assistant.",
        )
        assert caps.instructions == "You are a helpful research assistant."

    def test_is_multi_agent_false_by_default(self) -> None:
        """Test that is_multi_agent is False by default."""
        caps = Capabilities()
        assert caps.is_multi_agent is False
        assert caps.agent_count == 1

    def test_is_multi_agent_true_with_agents(self) -> None:
        """Test that is_multi_agent is True when agents defined."""
        caps = Capabilities(
            agents={
                "researcher": AgentSpec(name="researcher"),
                "writer": AgentSpec(name="writer"),
            }
        )
        assert caps.is_multi_agent is True
        assert caps.agent_count == 2

    def test_is_multi_agent_false_with_empty_dict(self) -> None:
        """Test that is_multi_agent is False with empty agents dict."""
        caps = Capabilities(agents={})
        assert caps.is_multi_agent is False
        assert caps.agent_count == 1  # Still returns 1 for single-agent fallback

    def test_auto_coordination_with_agents(self) -> None:
        """Test that coordination defaults to SEQUENTIAL when agents defined."""
        caps = Capabilities(
            agents={
                "agent1": AgentSpec(name="agent1"),
                "agent2": AgentSpec(name="agent2"),
            }
        )
        assert caps.coordination == CoordinationPattern.SEQUENTIAL

    def test_explicit_coordination_pattern(self) -> None:
        """Test explicit coordination pattern."""
        caps = Capabilities(
            agents={
                "agent1": AgentSpec(name="agent1"),
                "agent2": AgentSpec(name="agent2"),
            },
            coordination=CoordinationPattern.PARALLEL,
        )
        assert caps.coordination == CoordinationPattern.PARALLEL

    def test_full_multi_agent_configuration(self) -> None:
        """Test complete multi-agent configuration (model-agnostic)."""
        caps = Capabilities(
            agents={
                "researcher": AgentSpec(
                    name="researcher",
                    role="Gathers information",
                    tools=["web_search"],
                    skills=["research"],
                ),
                "analyst": AgentSpec(
                    name="analyst",
                    role="Analyzes data",
                    skills=["data-analysis"],
                ),
                "writer": AgentSpec(
                    name="writer",
                    role="Produces report",
                    skills=["writing"],
                ),
            },
            coordination=CoordinationPattern.SEQUENTIAL,
            instructions="You are a research team",
        )

        assert caps.is_multi_agent is True
        assert caps.agent_count == 3
        assert caps.coordination == CoordinationPattern.SEQUENTIAL
        assert caps.agents is not None
        assert "researcher" in caps.agents
        assert caps.agents["researcher"].tools == ["web_search"]
        assert caps.agents["researcher"].skills == ["research"]
        assert caps.instructions == "You are a research team"


class TestContractWithCapabilities:
    """Tests for Contract with Capabilities integration."""

    def test_contract_with_capabilities_and_execution(self) -> None:
        """Test creating Contract with Capabilities + ExecutionConfig."""
        contract = Contract(
            id="test",
            name="Test Contract",
            capabilities=Capabilities(
                tools=["web_search"],
                skills=["research"],
            ),
            execution=ExecutionConfig(model="gpt-4o"),
        )
        assert contract.capabilities is not None
        assert contract.capabilities.tools == ["web_search"]
        assert contract.execution is not None
        assert contract.execution.model == "gpt-4o"

    def test_contract_with_capabilities_only(self) -> None:
        """Test creating Contract with Capabilities but no ExecutionConfig."""
        contract = Contract(
            id="test",
            name="Test Contract",
            capabilities=Capabilities(tools=["calculator"]),
        )
        assert contract.capabilities is not None
        assert contract.capabilities.tools == ["calculator"]
        # No execution config provided
        assert contract.execution is None

    def test_contract_skills_creates_capabilities(self) -> None:
        """Test that skills field creates Capabilities if not provided."""
        contract = Contract(
            id="test",
            name="Test Contract",
            skills=["code-review", "research"],
        )
        assert contract.capabilities is not None
        assert contract.capabilities.skills == ["code-review", "research"]

    def test_contract_skills_merged_into_capabilities(self) -> None:
        """Test that skills are merged into existing Capabilities."""
        contract = Contract(
            id="test",
            name="Test Contract",
            skills=["new-skill"],
            capabilities=Capabilities(
                skills=["existing-skill"],
            ),
        )
        assert contract.capabilities is not None
        # Both skills should be present (merged)
        assert "new-skill" in contract.capabilities.skills
        assert "existing-skill" in contract.capabilities.skills

    def test_contract_without_skills_or_capabilities(self) -> None:
        """Test Contract without skills or capabilities."""
        contract = Contract(
            id="test",
            name="Test Contract",
        )
        # capabilities remains None if neither skills nor capabilities provided
        assert contract.capabilities is None

    def test_contract_execute_requires_capabilities(self) -> None:
        """Test that execute() raises error if no capabilities."""
        contract = Contract(
            id="test",
            name="Test Contract",
            resources=ResourceConstraints(tokens=1000),
        )
        with pytest.raises(ValueError, match="must have capabilities defined"):
            contract.execute(query="test")

    def test_contract_with_full_configuration(self) -> None:
        """Test Contract with full configuration."""
        contract = Contract(
            id="research-task",
            name="Research Task",
            description="Research a topic and provide summary",
            resources=ResourceConstraints(tokens=10000, cost_usd=0.50),
            capabilities=Capabilities(
                tools=["web_search"],
                mcp_servers=["filesystem://local"],
                skills=["research", "summarize"],
                instructions="You are a helpful research assistant.",
            ),
            execution=ExecutionConfig(
                model="gpt-4o",
                temperature=0.7,
                max_retries=3,
            ),
        )
        assert contract.id == "research-task"
        # Model-agnostic capabilities
        assert contract.capabilities is not None
        assert contract.capabilities.tools == ["web_search"]
        assert contract.capabilities.skills == ["research", "summarize"]
        assert contract.capabilities.instructions == "You are a helpful research assistant."
        # Model-specific execution config
        assert contract.execution is not None
        assert contract.execution.model == "gpt-4o"
        assert contract.execution.temperature == 0.7
        assert contract.execution.max_retries == 3


class TestSkillSpec:
    """Tests for SkillSpec dataclass (agentskills.io standard)."""

    def test_create_minimal_skill_spec(self) -> None:
        """Test creating SkillSpec with only required fields."""
        skill = SkillSpec(
            name="code-reviewer",
            description="Review code for best practices and security issues.",
        )
        assert skill.name == "code-reviewer"
        assert skill.description == "Review code for best practices and security issues."
        assert skill.instructions == ""
        assert skill.version == "1.0.0"
        assert skill.license is None
        assert skill.allowed_tools is None
        assert skill.references == {}
        assert skill.scripts == {}
        assert skill.metadata == {}

    def test_create_full_skill_spec(self) -> None:
        """Test creating SkillSpec with all fields."""
        skill = SkillSpec(
            name="data-analyst",
            description="Analyze datasets and produce insights.",
            instructions="## Steps\n1. Load data\n2. Clean data\n3. Analyze",
            version="2.0.0",
            license="MIT",
            allowed_tools=["Read", "Grep", "calculator"],
            references={"guide": "See reference.md for details"},
            scripts={"validate": "python validate.py"},
            metadata={"author": "test", "category": "analytics"},
        )
        assert skill.name == "data-analyst"
        assert skill.version == "2.0.0"
        assert skill.license == "MIT"
        assert skill.allowed_tools == ["Read", "Grep", "calculator"]
        assert skill.references["guide"] == "See reference.md for details"
        assert skill.scripts["validate"] == "python validate.py"
        assert skill.metadata["author"] == "test"

    def test_name_validation_valid_names(self) -> None:
        """Test that valid names are accepted."""
        # Single character (minimum)
        skill = SkillSpec(name="a", description="Test")
        assert skill.name == "a"

        # With hyphens
        skill = SkillSpec(name="code-review", description="Test")
        assert skill.name == "code-review"

        # With numbers
        skill = SkillSpec(name="skill123", description="Test")
        assert skill.name == "skill123"

        # Long name (within 64 char limit)
        long_name = "a" + "b" * 62  # 63 chars
        skill = SkillSpec(name=long_name, description="Test")
        assert skill.name == long_name

    def test_name_validation_invalid_names(self) -> None:
        """Test that invalid names are rejected."""
        # Starts with hyphen
        with pytest.raises(ValueError, match="Skill name must be"):
            SkillSpec(name="-invalid", description="Test")

        # Ends with hyphen
        with pytest.raises(ValueError, match="Skill name must be"):
            SkillSpec(name="invalid-", description="Test")

        # Contains uppercase
        with pytest.raises(ValueError, match="Skill name must be"):
            SkillSpec(name="Invalid", description="Test")

        # Contains underscore
        with pytest.raises(ValueError, match="Skill name must be"):
            SkillSpec(name="invalid_name", description="Test")

        # Contains space
        with pytest.raises(ValueError, match="Skill name must be"):
            SkillSpec(name="invalid name", description="Test")

        # Starts with number
        with pytest.raises(ValueError, match="Skill name must be"):
            SkillSpec(name="1invalid", description="Test")

    def test_description_validation(self) -> None:
        """Test description length validation."""
        # Empty description
        with pytest.raises(ValueError, match="cannot be empty"):
            SkillSpec(name="test", description="")

        # Description too long (>1024 chars)
        long_desc = "x" * 1025
        with pytest.raises(ValueError, match="must be ≤1024"):
            SkillSpec(name="test", description=long_desc)

        # Exactly 1024 chars (should work)
        exact_desc = "x" * 1024
        skill = SkillSpec(name="test", description=exact_desc)
        assert len(skill.description) == 1024

    def test_to_frontmatter(self) -> None:
        """Test YAML frontmatter generation."""
        skill = SkillSpec(
            name="test-skill",
            description="A test skill for testing.",
            allowed_tools=["Read", "Write"],
            metadata={"author": "test"},
        )
        frontmatter = skill.to_frontmatter()

        assert "name: test-skill" in frontmatter
        assert "description: A test skill for testing." in frontmatter
        assert "allowed-tools: Read Write" in frontmatter
        assert "author: test" in frontmatter
        # Version should not appear (default 1.0.0)
        assert "version:" not in frontmatter

    def test_to_frontmatter_with_custom_version(self) -> None:
        """Test frontmatter includes non-default version."""
        skill = SkillSpec(
            name="test",
            description="Test",
            version="2.0.0",
            license="Apache-2.0",
        )
        frontmatter = skill.to_frontmatter()

        assert "version: 2.0.0" in frontmatter
        assert "license: Apache-2.0" in frontmatter

    def test_to_skill_md(self) -> None:
        """Test complete SKILL.md generation."""
        skill = SkillSpec(
            name="code-reviewer",
            description="Review code for issues.",
            instructions="## Instructions\n1. Read code\n2. Find issues",
        )
        skill_md = skill.to_skill_md()

        assert skill_md.startswith("---\n")
        assert "name: code-reviewer" in skill_md
        assert "---\n\n## Instructions" in skill_md

    def test_from_skill_md_basic(self) -> None:
        """Test parsing basic SKILL.md content."""
        content = """---
name: test-skill
description: A simple test skill.
---

## Instructions
Do the thing."""

        skill = SkillSpec.from_skill_md(content)
        assert skill.name == "test-skill"
        assert skill.description == "A simple test skill."
        assert "Do the thing" in skill.instructions

    def test_from_skill_md_with_options(self) -> None:
        """Test parsing SKILL.md with optional fields."""
        content = """---
name: advanced-skill
description: An advanced skill.
version: 2.1.0
license: MIT
allowed-tools: Read Grep Write
metadata:
  author: test-author
  category: testing
---

Instructions here."""

        skill = SkillSpec.from_skill_md(content)
        assert skill.name == "advanced-skill"
        assert skill.version == "2.1.0"
        assert skill.license == "MIT"
        assert skill.allowed_tools == ["Read", "Grep", "Write"]
        assert skill.metadata["author"] == "test-author"

    def test_from_skill_md_override_name(self) -> None:
        """Test name override in from_skill_md."""
        content = """---
name: original-name
description: Test skill.
---

Content."""

        skill = SkillSpec.from_skill_md(content, name="override-name")
        assert skill.name == "override-name"

    def test_from_skill_md_invalid_no_frontmatter(self) -> None:
        """Test error on missing frontmatter."""
        with pytest.raises(ValueError, match="must start with YAML frontmatter"):
            SkillSpec.from_skill_md("No frontmatter here")

    def test_from_skill_md_invalid_no_closing(self) -> None:
        """Test error on missing closing frontmatter delimiter."""
        with pytest.raises(ValueError, match="must have closing"):
            SkillSpec.from_skill_md("---\nname: test\n")

    def test_from_skill_md_missing_required(self) -> None:
        """Test error on missing required fields."""
        with pytest.raises(ValueError, match="missing required 'name'"):
            SkillSpec.from_skill_md("---\ndescription: Test\n---\nContent")

        with pytest.raises(ValueError, match="missing required 'description'"):
            SkillSpec.from_skill_md("---\nname: test\n---\nContent")

    def test_roundtrip_skill_md(self) -> None:
        """Test that to_skill_md and from_skill_md are inverses."""
        original = SkillSpec(
            name="roundtrip-test",
            description="Testing roundtrip serialization.",
            instructions="## Steps\n1. First step\n2. Second step",
            version="1.5.0",
            license="CC-BY-4.0",
            allowed_tools=["Read", "Write"],
        )

        # Export to SKILL.md
        skill_md = original.to_skill_md()

        # Parse back
        parsed = SkillSpec.from_skill_md(skill_md)

        assert parsed.name == original.name
        assert parsed.description == original.description
        assert original.instructions in parsed.instructions  # May have whitespace diff
        assert parsed.version == original.version
        assert parsed.license == original.license
        assert parsed.allowed_tools == original.allowed_tools

    def test_metadata_tokens(self) -> None:
        """Test metadata token estimation."""
        skill = SkillSpec(
            name="short",  # 5 chars
            description="A" * 100,  # 100 chars
        )
        # ~105 chars / 4 ≈ 26 tokens
        assert skill.metadata_tokens > 20
        assert skill.metadata_tokens < 40

    def test_instruction_tokens(self) -> None:
        """Test instruction token estimation."""
        skill = SkillSpec(
            name="test",
            description="Test",
            instructions="x" * 400,  # 400 chars
        )
        # 400 chars / 4 = 100 tokens
        assert skill.instruction_tokens == 100


class TestCapabilitiesWithSkillSpec:
    """Tests for Capabilities with SkillSpec integration."""

    def test_capabilities_with_string_skills(self) -> None:
        """Test backward compatibility with string skills."""
        caps = Capabilities(skills=["code-review", "research"])
        assert caps.skills == ["code-review", "research"]
        assert caps.skill_names == ["code-review", "research"]
        assert caps.skill_specs == []

    def test_capabilities_with_skill_spec(self) -> None:
        """Test Capabilities with SkillSpec objects."""
        skill = SkillSpec(
            name="code-reviewer",
            description="Review code",
            instructions="Do review",
        )
        caps = Capabilities(skills=[skill])

        assert len(caps.skills) == 1
        assert caps.skill_names == ["code-reviewer"]
        assert caps.skill_specs == [skill]

    def test_capabilities_mixed_skills(self) -> None:
        """Test Capabilities with mixed string and SkillSpec skills."""
        skill = SkillSpec(
            name="code-reviewer",
            description="Review code",
        )
        caps = Capabilities(skills=[skill, "simple-skill", "another-skill"])

        assert len(caps.skills) == 3
        assert caps.skill_names == ["code-reviewer", "simple-skill", "another-skill"]
        assert len(caps.skill_specs) == 1

    def test_get_skill_returns_skill_spec(self) -> None:
        """Test get_skill returns SkillSpec when found."""
        skill = SkillSpec(name="test-skill", description="Test")
        caps = Capabilities(skills=[skill, "other"])

        result = caps.get_skill("test-skill")
        assert result is skill

    def test_get_skill_returns_none_for_string(self) -> None:
        """Test get_skill returns None for string-only skills."""
        caps = Capabilities(skills=["string-skill"])
        assert caps.get_skill("string-skill") is None

    def test_get_skill_returns_none_for_missing(self) -> None:
        """Test get_skill returns None for missing skills."""
        caps = Capabilities(skills=["existing"])
        assert caps.get_skill("missing") is None

    def test_has_skill_with_skill_spec(self) -> None:
        """Test has_skill finds SkillSpec by name."""
        skill = SkillSpec(name="test-skill", description="Test")
        caps = Capabilities(skills=[skill])
        assert caps.has_skill("test-skill") is True
        assert caps.has_skill("missing") is False

    def test_has_skill_with_string(self) -> None:
        """Test has_skill finds string skills."""
        caps = Capabilities(skills=["string-skill"])
        assert caps.has_skill("string-skill") is True
        assert caps.has_skill("missing") is False

    def test_total_metadata_tokens(self) -> None:
        """Test total_metadata_tokens calculation."""
        skill = SkillSpec(
            name="test",
            description="A" * 100,
        )
        caps = Capabilities(skills=[skill, "simple"])

        # SkillSpec contributes ~26 tokens, string ~2-3 tokens
        assert caps.total_metadata_tokens > 20

    def test_total_instruction_tokens(self) -> None:
        """Test total_instruction_tokens calculation."""
        skill1 = SkillSpec(
            name="skill1",
            description="Test",
            instructions="x" * 400,  # 100 tokens
        )
        skill2 = SkillSpec(
            name="skill2",
            description="Test",
            instructions="y" * 200,  # 50 tokens
        )
        caps = Capabilities(skills=[skill1, skill2, "string-skill"])

        assert caps.total_instruction_tokens == 150  # 100 + 50

    def test_total_instruction_tokens_no_skill_specs(self) -> None:
        """Test total_instruction_tokens with only string skills."""
        caps = Capabilities(skills=["a", "b", "c"])
        assert caps.total_instruction_tokens == 0


class TestContractSkillMerging:
    """Tests for Contract skill merging with SkillSpec."""

    def test_contract_skills_string_creates_capabilities(self) -> None:
        """Test string skills create Capabilities."""
        contract = Contract(
            id="test",
            name="Test",
            skills=["skill1", "skill2"],
        )
        assert contract.capabilities is not None
        assert contract.capabilities.skill_names == ["skill1", "skill2"]

    def test_contract_skills_skillspec_creates_capabilities(self) -> None:
        """Test SkillSpec skills create Capabilities."""
        skill = SkillSpec(name="test-skill", description="Test")
        contract = Contract(
            id="test",
            name="Test",
            skills=[skill],
        )
        assert contract.capabilities is not None
        assert contract.capabilities.skill_names == ["test-skill"]
        assert contract.capabilities.get_skill("test-skill") is skill

    def test_contract_merges_skills_into_capabilities(self) -> None:
        """Test skills merge into existing Capabilities."""
        skill = SkillSpec(name="new-skill", description="New skill")
        contract = Contract(
            id="test",
            name="Test",
            skills=[skill],
            capabilities=Capabilities(skills=["existing"]),
        )
        assert contract.capabilities is not None
        # Both skills should be present
        assert "existing" in contract.capabilities.skill_names
        assert "new-skill" in contract.capabilities.skill_names

    def test_skill_spec_takes_precedence_over_string(self) -> None:
        """Test SkillSpec replaces string with same name."""
        skill = SkillSpec(
            name="duplicate",
            description="The SkillSpec version",
            instructions="Full instructions",
        )
        contract = Contract(
            id="test",
            name="Test",
            skills=[skill],
            capabilities=Capabilities(skills=["duplicate"]),  # String version
        )
        assert contract.capabilities is not None
        # SkillSpec should replace string
        result = contract.capabilities.get_skill("duplicate")
        assert result is not None
        assert result.instructions == "Full instructions"

    def test_merge_preserves_unique_skills(self) -> None:
        """Test merge doesn't create duplicates."""
        skill1 = SkillSpec(name="skill1", description="First")
        skill2 = SkillSpec(name="skill2", description="Second")
        contract = Contract(
            id="test",
            name="Test",
            skills=[skill1, "skill3"],
            capabilities=Capabilities(skills=[skill2, "skill4"]),
        )
        assert contract.capabilities is not None
        names = contract.capabilities.skill_names
        assert len(names) == 4
        assert set(names) == {"skill1", "skill2", "skill3", "skill4"}
