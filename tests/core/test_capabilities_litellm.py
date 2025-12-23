"""Tests for Capabilities LiteLLM integration methods.

These tests validate the methods that convert Capabilities to LiteLLM-compatible
formats for tools, MCP servers, and skills.
"""

from agent_contracts.core.contract import Capabilities, SkillSpec


class TestToMcpTools:
    """Tests for to_mcp_tools() method."""

    def test_empty_mcp_servers(self) -> None:
        """Test with no MCP servers."""
        caps = Capabilities()
        result = caps.to_mcp_tools()
        assert result == []

    def test_single_mcp_server(self) -> None:
        """Test with a single MCP server."""
        caps = Capabilities(mcp_servers=["https://mcp.example.com/api"])
        result = caps.to_mcp_tools()

        assert len(result) == 1
        assert result[0]["type"] == "mcp"
        assert result[0]["server_url"] == "https://mcp.example.com/api"
        assert result[0]["require_approval"] == "never"
        # Label extracted from domain
        assert result[0]["server_label"] == "mcp"

    def test_multiple_mcp_servers(self) -> None:
        """Test with multiple MCP servers."""
        caps = Capabilities(
            mcp_servers=[
                "https://mcp1.example.com/api",
                "https://mcp2.example.com/api",
            ]
        )
        result = caps.to_mcp_tools()

        assert len(result) == 2
        assert result[0]["server_url"] == "https://mcp1.example.com/api"
        assert result[1]["server_url"] == "https://mcp2.example.com/api"

    def test_mcp_server_without_url_scheme(self) -> None:
        """Test with MCP server that doesn't have a URL scheme."""
        caps = Capabilities(mcp_servers=["localhost:8080"])
        result = caps.to_mcp_tools()

        assert len(result) == 1
        assert result[0]["server_label"] == "mcp-server-0"


class TestToLitellmTools:
    """Tests for to_litellm_tools() method."""

    def test_empty_capabilities(self) -> None:
        """Test with no tools or MCP servers."""
        caps = Capabilities()
        result = caps.to_litellm_tools()
        assert result == []

    def test_mcp_only(self) -> None:
        """Test with only MCP servers."""
        caps = Capabilities(mcp_servers=["https://mcp.example.com/api"])
        result = caps.to_litellm_tools()

        assert len(result) == 1
        assert result[0]["type"] == "mcp"

    def test_tools_without_definitions(self) -> None:
        """Test with tools but no definitions (only MCP tools returned)."""
        caps = Capabilities(
            tools=["get_weather", "search"],
            mcp_servers=["https://mcp.example.com/api"],
        )
        result = caps.to_litellm_tools()

        # Only MCP tool is returned (no definitions for function tools)
        assert len(result) == 1
        assert result[0]["type"] == "mcp"

    def test_tools_with_definitions(self) -> None:
        """Test with tools and their definitions."""
        caps = Capabilities(
            tools=["get_weather", "search"],
            mcp_servers=["https://mcp.example.com/api"],
        )
        definitions = {
            "get_weather": {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        }
        result = caps.to_litellm_tools(tool_definitions=definitions)

        # get_weather function + MCP tool (search not in definitions)
        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[1]["type"] == "mcp"


class TestHasTools:
    """Tests for has_tools() method."""

    def test_no_tools(self) -> None:
        """Test with no tools or MCP servers."""
        caps = Capabilities()
        assert caps.has_tools() is False

    def test_has_tools_only(self) -> None:
        """Test with only tools."""
        caps = Capabilities(tools=["get_weather"])
        assert caps.has_tools() is True

    def test_has_mcp_only(self) -> None:
        """Test with only MCP servers."""
        caps = Capabilities(mcp_servers=["https://mcp.example.com"])
        assert caps.has_tools() is True

    def test_has_both(self) -> None:
        """Test with both tools and MCP servers."""
        caps = Capabilities(
            tools=["get_weather"],
            mcp_servers=["https://mcp.example.com"],
        )
        assert caps.has_tools() is True


class TestGetCodeExecutionTool:
    """Tests for get_code_execution_tool() method."""

    def test_no_code_execution(self) -> None:
        """Test without code_execution in tools."""
        caps = Capabilities(tools=["get_weather"])
        result = caps.get_code_execution_tool()
        assert result is None

    def test_has_code_execution(self) -> None:
        """Test with code_execution in tools."""
        caps = Capabilities(tools=["code_execution", "get_weather"])
        result = caps.get_code_execution_tool()

        assert result is not None
        assert result["type"] == "code_execution_20250825"
        assert result["name"] == "code_execution"


class TestToAnthropicSkills:
    """Tests for to_anthropic_skills() method."""

    def test_no_skills(self) -> None:
        """Test with no skills."""
        caps = Capabilities()
        result = caps.to_anthropic_skills()
        assert result == []

    def test_string_skills_only(self) -> None:
        """Test with only string skills (not SkillSpec)."""
        caps = Capabilities(skills=["code-review", "research"])
        result = caps.to_anthropic_skills()
        # String skills are not converted to Anthropic format
        assert result == []

    def test_skill_specs(self) -> None:
        """Test with SkillSpec objects."""
        skill = SkillSpec(
            name="code-reviewer",
            description="Review code",
            instructions="Look for bugs",
        )
        caps = Capabilities(skills=[skill])
        result = caps.to_anthropic_skills()

        assert len(result) == 1
        assert result[0]["type"] == "anthropic"
        assert result[0]["skill_id"] == "code-reviewer"
        # SkillSpec has default version="1.0.0", not "latest"
        assert result[0]["version"] == "1.0.0"

    def test_skill_spec_with_version(self) -> None:
        """Test SkillSpec with explicit version."""
        skill = SkillSpec(
            name="code-reviewer",
            description="Review code",
            instructions="Look for bugs",
            version="1.0.0",
        )
        caps = Capabilities(skills=[skill])
        result = caps.to_anthropic_skills()

        assert result[0]["version"] == "1.0.0"

    def test_mixed_skills(self) -> None:
        """Test with mix of string skills and SkillSpecs."""
        skill = SkillSpec(
            name="code-reviewer",
            description="Review code",
            instructions="Look for bugs",
        )
        caps = Capabilities(skills=[skill, "simple-skill"])
        result = caps.to_anthropic_skills()

        # Only SkillSpec is converted
        assert len(result) == 1
        assert result[0]["skill_id"] == "code-reviewer"


class TestGetSkillInstructions:
    """Tests for get_skill_instructions() method."""

    def test_no_skills(self) -> None:
        """Test with no skills."""
        caps = Capabilities()
        result = caps.get_skill_instructions()
        assert result == ""

    def test_string_skills_only(self) -> None:
        """Test with only string skills."""
        caps = Capabilities(skills=["code-review", "research"])
        result = caps.get_skill_instructions()
        assert result == ""

    def test_skill_with_instructions(self) -> None:
        """Test SkillSpec with instructions."""
        skill = SkillSpec(
            name="code-reviewer",
            description="Review code for issues",
            instructions="1. Read the code\n2. Find bugs\n3. Report findings",
        )
        caps = Capabilities(skills=[skill])
        result = caps.get_skill_instructions()

        assert "## Skill: code-reviewer" in result
        assert "Description: Review code for issues" in result
        assert "Instructions:" in result
        assert "1. Read the code" in result

    def test_multiple_skills(self) -> None:
        """Test multiple SkillSpecs."""
        skill1 = SkillSpec(
            name="code-reviewer",
            description="Review code",
            instructions="Check for bugs",
        )
        skill2 = SkillSpec(
            name="researcher",
            description="Research topics",
            instructions="Search and summarize",
        )
        caps = Capabilities(skills=[skill1, skill2])
        result = caps.get_skill_instructions()

        assert "## Skill: code-reviewer" in result
        assert "## Skill: researcher" in result

    def test_active_skills_filter(self) -> None:
        """Test filtering by active skills."""
        skill1 = SkillSpec(
            name="code-reviewer",
            description="Review code",
            instructions="Check for bugs",
        )
        skill2 = SkillSpec(
            name="researcher",
            description="Research topics",
            instructions="Search and summarize",
        )
        caps = Capabilities(skills=[skill1, skill2])
        result = caps.get_skill_instructions(active_skills=["code-reviewer"])

        assert "## Skill: code-reviewer" in result
        assert "## Skill: researcher" not in result

    def test_skill_without_instructions(self) -> None:
        """Test SkillSpec without instructions."""
        skill = SkillSpec(
            name="code-reviewer",
            description="Review code",
            # No instructions
        )
        caps = Capabilities(skills=[skill])
        result = caps.get_skill_instructions()
        assert result == ""


class TestCapabilitiesIntegration:
    """Integration tests for Capabilities with Contract."""

    def test_contract_with_full_capabilities(self) -> None:
        """Test Contract with fully-featured Capabilities."""
        from agent_contracts.core.contract import Contract, ResourceConstraints

        skill = SkillSpec(
            name="code-reviewer",
            description="Review code",
            instructions="Find bugs",
        )
        caps = Capabilities(
            tools=["get_weather", "code_execution"],
            mcp_servers=["https://mcp.example.com/api"],
            skills=[skill, "simple-skill"],
        )
        contract = Contract(
            id="test",
            name="Test Contract",
            resources=ResourceConstraints(tokens=1000),
            capabilities=caps,
        )

        # Verify all capability methods work through contract
        assert contract.capabilities.has_tools() is True
        assert len(contract.capabilities.to_mcp_tools()) == 1
        assert contract.capabilities.get_code_execution_tool() is not None
        assert len(contract.capabilities.to_anthropic_skills()) == 1
        assert "## Skill: code-reviewer" in contract.capabilities.get_skill_instructions()
