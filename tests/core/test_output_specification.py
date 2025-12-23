"""Tests for OutputSpecification structured output support.

These tests validate the enhanced OutputSpecification class that supports:
- Pydantic BaseModel classes for type-safe structured output
- JSON Schema dicts for schema-based validation
- LiteLLM response_format conversion
- Output validation against schemas
"""

import json

import pytest
from pydantic import BaseModel

from agent_contracts.core.contract import OutputSpecification


class SimpleResponse(BaseModel):
    """Simple test response model."""

    answer: str
    confidence: float


class ComplexResponse(BaseModel):
    """Complex test response model with nested types."""

    title: str
    items: list[str]
    metadata: dict[str, str]


class TestOutputSpecificationBasics:
    """Test basic OutputSpecification functionality."""

    def test_default_output_specification(self) -> None:
        """Test default OutputSpecification has no structured output."""
        output = OutputSpecification()
        assert output.schema is None
        assert output.pydantic_model is None
        assert output.strict is True
        assert output.min_quality == 0.0
        assert output.name is None
        assert not output.has_structured_output()

    def test_output_specification_with_json_schema(self) -> None:
        """Test OutputSpecification with JSON Schema."""
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["answer", "score"],
        }
        output = OutputSpecification(schema=schema, name="test_response")

        assert output.schema == schema
        assert output.pydantic_model is None
        assert output.name == "test_response"
        assert output.has_structured_output()

    def test_output_specification_with_pydantic_model(self) -> None:
        """Test OutputSpecification with Pydantic model."""
        output = OutputSpecification(pydantic_model=SimpleResponse)

        assert output.schema is None
        assert output.pydantic_model is SimpleResponse
        assert output.has_structured_output()

    def test_cannot_specify_both_schema_and_pydantic_model(self) -> None:
        """Test that specifying both schema and pydantic_model raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            OutputSpecification(
                schema={"type": "object"},
                pydantic_model=SimpleResponse,
            )

    def test_invalid_pydantic_model_type(self) -> None:
        """Test that non-Pydantic class raises error."""
        with pytest.raises(ValueError, match="must be a Pydantic BaseModel class"):
            OutputSpecification(pydantic_model=dict)  # type: ignore[arg-type]

    def test_min_quality_validation(self) -> None:
        """Test min_quality bounds validation."""
        # Valid values
        OutputSpecification(min_quality=0.0)
        OutputSpecification(min_quality=0.5)
        OutputSpecification(min_quality=1.0)

        # Invalid values
        with pytest.raises(ValueError, match="min_quality must be in"):
            OutputSpecification(min_quality=-0.1)
        with pytest.raises(ValueError, match="min_quality must be in"):
            OutputSpecification(min_quality=1.1)


class TestToResponseFormat:
    """Test conversion to LiteLLM response_format."""

    def test_no_schema_returns_none(self) -> None:
        """Test that no schema returns None."""
        output = OutputSpecification()
        assert output.to_response_format() is None

    def test_pydantic_model_returns_class(self) -> None:
        """Test that Pydantic model returns the class directly."""
        output = OutputSpecification(pydantic_model=SimpleResponse)
        result = output.to_response_format()

        assert result is SimpleResponse

    def test_json_schema_returns_wrapped_format(self) -> None:
        """Test that JSON Schema returns LiteLLM-wrapped format."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        output = OutputSpecification(schema=schema, name="my_response", strict=True)
        result = output.to_response_format()

        assert isinstance(result, dict)
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "my_response"
        assert result["json_schema"]["strict"] is True
        assert result["json_schema"]["schema"] == schema

    def test_json_schema_default_name(self) -> None:
        """Test that JSON Schema uses default name if not specified."""
        output = OutputSpecification(schema={"type": "object"})
        result = output.to_response_format()

        assert isinstance(result, dict)
        assert result["json_schema"]["name"] == "response"

    def test_json_schema_non_strict_mode(self) -> None:
        """Test JSON Schema with strict=False."""
        output = OutputSpecification(schema={"type": "object"}, strict=False)
        result = output.to_response_format()

        assert isinstance(result, dict)
        assert result["json_schema"]["strict"] is False


class TestValidateOutput:
    """Test output validation functionality."""

    def test_no_schema_always_valid(self) -> None:
        """Test that no schema means any output is valid."""
        output = OutputSpecification()

        is_valid, error = output.validate_output('{"anything": "works"}')
        assert is_valid is True
        assert error is None

    def test_pydantic_validation_success(self) -> None:
        """Test successful Pydantic validation."""
        output = OutputSpecification(pydantic_model=SimpleResponse)

        is_valid, error = output.validate_output('{"answer": "42", "confidence": 0.95}')
        assert is_valid is True
        assert error is None

    def test_pydantic_validation_with_dict(self) -> None:
        """Test Pydantic validation with dict input."""
        output = OutputSpecification(pydantic_model=SimpleResponse)

        is_valid, error = output.validate_output({"answer": "42", "confidence": 0.95})
        assert is_valid is True
        assert error is None

    def test_pydantic_validation_failure(self) -> None:
        """Test failed Pydantic validation."""
        output = OutputSpecification(pydantic_model=SimpleResponse)

        # Missing required field
        is_valid, error = output.validate_output('{"answer": "42"}')
        assert is_valid is False
        assert error is not None
        assert "Pydantic validation failed" in error

    def test_pydantic_validation_wrong_type(self) -> None:
        """Test Pydantic validation with wrong type."""
        output = OutputSpecification(pydantic_model=SimpleResponse)

        # confidence should be float, not string
        is_valid, error = output.validate_output('{"answer": "42", "confidence": "high"}')
        assert is_valid is False
        assert error is not None

    def test_invalid_json_string(self) -> None:
        """Test validation with invalid JSON string."""
        output = OutputSpecification(pydantic_model=SimpleResponse)

        is_valid, error = output.validate_output("not valid json")
        assert is_valid is False
        assert error is not None
        assert "Invalid JSON" in error

    def test_json_schema_validation_success(self) -> None:
        """Test successful JSON Schema validation."""
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["answer", "score"],
        }
        output = OutputSpecification(schema=schema)

        is_valid, error = output.validate_output('{"answer": "test", "score": 85}')
        # Note: This will succeed if jsonschema is installed, skip if not
        assert is_valid is True
        assert error is None

    def test_json_schema_validation_failure(self) -> None:
        """Test failed JSON Schema validation."""
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["answer", "score"],
        }
        output = OutputSpecification(schema=schema)

        # Missing required field
        is_valid, error = output.validate_output('{"answer": "test"}')

        # If jsonschema is installed, this should fail
        # If not installed, it should pass (skip validation)
        try:
            import jsonschema  # noqa: F401

            assert is_valid is False
            assert error is not None
            assert "JSON Schema validation failed" in error
        except ImportError:
            # jsonschema not installed - validation is skipped
            assert is_valid is True


class TestComplexModels:
    """Test with more complex Pydantic models."""

    def test_complex_nested_model(self) -> None:
        """Test validation with nested model types."""
        output = OutputSpecification(pydantic_model=ComplexResponse)

        valid_data = {
            "title": "Test",
            "items": ["a", "b", "c"],
            "metadata": {"key": "value"},
        }
        is_valid, error = output.validate_output(json.dumps(valid_data))
        assert is_valid is True
        assert error is None

    def test_complex_model_response_format(self) -> None:
        """Test response_format with complex model."""
        output = OutputSpecification(pydantic_model=ComplexResponse)
        result = output.to_response_format()

        assert result is ComplexResponse


class TestIntegrationWithContract:
    """Test OutputSpecification integration with Contract class."""

    def test_contract_with_output_specification(self) -> None:
        """Test creating Contract with OutputSpecification."""
        from agent_contracts.core.contract import Contract, ResourceConstraints

        output_spec = OutputSpecification(pydantic_model=SimpleResponse)
        contract = Contract(
            id="test-contract",
            name="Test Contract",
            resources=ResourceConstraints(tokens=1000),
            outputs=output_spec,
        )

        assert contract.outputs.has_structured_output()
        assert contract.outputs.to_response_format() is SimpleResponse

    def test_contract_default_output_specification(self) -> None:
        """Test Contract with default OutputSpecification."""
        from agent_contracts.core.contract import Contract, ResourceConstraints

        contract = Contract(
            id="test-contract",
            name="Test Contract",
            resources=ResourceConstraints(tokens=1000),
        )

        assert not contract.outputs.has_structured_output()
        assert contract.outputs.to_response_format() is None
