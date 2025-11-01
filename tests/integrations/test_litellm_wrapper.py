"""Unit tests for litellm integration."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from agent_contracts.core import Contract, ContractState, ResourceConstraints, TemporalConstraints
from agent_contracts.integrations import ContractedLLM, ContractViolationError


class TestContractedLLM:
    """Tests for ContractedLLM wrapper."""

    def test_create_contracted_llm(self) -> None:
        """Test creating a contracted LLM wrapper."""
        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        llm = ContractedLLM(contract)

        assert llm.contract == contract
        assert llm.enforcer is not None
        assert llm.auto_start is True
        assert not llm._started

    def test_manual_start_stop(self) -> None:
        """Test manual start and stop."""
        contract = Contract(id="test", name="Test")
        llm = ContractedLLM(contract, auto_start=False)

        assert not llm._started

        llm.start()
        assert llm._started
        assert contract.state == ContractState.ACTIVE

        llm.stop()
        assert not llm._started

    def test_start_already_started_raises_error(self) -> None:
        """Test that starting already started LLM raises error."""
        contract = Contract(id="test", name="Test")
        llm = ContractedLLM(contract, auto_start=False)
        llm.start()

        with pytest.raises(RuntimeError, match="already started"):
            llm.start()

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_completion_basic(self, mock_completion: MagicMock) -> None:
        """Test basic completion call."""
        # Setup mock response
        mock_completion.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-4o-mini",
        }

        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        llm = ContractedLLM(contract)

        response = llm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}])

        # Verify response
        assert response["choices"][0]["message"]["content"] == "Hello!"

        # Verify usage tracking
        assert llm.enforcer.monitor.usage.tokens == 15
        assert llm.enforcer.monitor.usage.api_calls == 1
        assert llm.enforcer.monitor.usage.cost_usd > 0

        # Verify auto-start
        assert llm._started

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_completion_with_cost_from_response(self, mock_completion: MagicMock) -> None:
        """Test completion using cost from litellm response."""
        mock_completion.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-4o-mini",
            "_hidden_params": {"response_cost": 0.0001},
        }

        contract = Contract(id="test", name="Test")
        llm = ContractedLLM(contract)

        llm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}])

        # Should use cost from response
        assert llm.enforcer.monitor.usage.cost_usd == 0.0001

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_completion_violation_strict_mode(self, mock_completion: MagicMock) -> None:
        """Test that violations raise errors in strict mode."""
        mock_completion.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 500, "total_tokens": 2000},
            "model": "gpt-4o-mini",
        }

        # Set tight token limit
        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        llm = ContractedLLM(contract, strict_mode=True)

        # Should raise violation error
        with pytest.raises(ContractViolationError, match="Contract violated"):
            llm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}])

        # Contract should be violated
        assert contract.state == ContractState.VIOLATED

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_completion_violation_lenient_mode(self, mock_completion: MagicMock) -> None:
        """Test that violations don't raise errors in lenient mode."""
        mock_completion.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 500, "total_tokens": 2000},
            "model": "gpt-4o-mini",
        }

        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        llm = ContractedLLM(contract, strict_mode=False)

        # Should not raise error
        response = llm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}])

        assert response is not None
        assert llm.enforcer.monitor.is_violated()
        # Contract still active in lenient mode
        assert contract.state == ContractState.ACTIVE

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_completion_api_call_error(self, mock_completion: MagicMock) -> None:
        """Test that API errors are tracked."""
        mock_completion.side_effect = Exception("API Error")

        contract = Contract(id="test", name="Test")
        llm = ContractedLLM(contract)

        with pytest.raises(Exception, match="API Error"):
            llm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}])

        # Should still track the failed API call
        assert llm.enforcer.monitor.usage.api_calls == 1

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_multiple_completions(self, mock_completion: MagicMock) -> None:
        """Test multiple completion calls."""
        mock_completion.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-4o-mini",
        }

        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        llm = ContractedLLM(contract)

        # Make multiple calls
        for i in range(5):
            llm.completion(
                model="gpt-4o-mini", messages=[{"role": "user", "content": f"Message {i}"}]
            )

        # Verify cumulative usage
        assert llm.enforcer.monitor.usage.tokens == 75  # 15 * 5
        assert llm.enforcer.monitor.usage.api_calls == 5

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_streaming_completion(self, mock_completion: MagicMock) -> None:
        """Test streaming completion."""
        # Mock streaming response
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}], "usage": None},
            {"choices": [{"delta": {"content": " world"}}], "usage": None},
            {
                "choices": [{"delta": {"content": "!"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        ]
        mock_completion.return_value = iter(chunks)

        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        llm = ContractedLLM(contract)

        # Collect streaming response
        collected = []
        for chunk in llm.streaming_completion(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}]
        ):
            collected.append(chunk)

        assert len(collected) == 3
        assert llm.enforcer.monitor.usage.api_calls == 1

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_temporal_constraint_violation(self, mock_completion: MagicMock) -> None:
        """Test temporal constraint violations."""
        import time

        mock_completion.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-4o-mini",
        }

        # Very short duration limit
        contract = Contract(
            id="test",
            name="Test",
            temporal=TemporalConstraints(max_duration=timedelta(seconds=0.01)),
        )
        llm = ContractedLLM(contract, strict_mode=True)
        llm.start()

        time.sleep(0.02)  # Exceed duration

        with pytest.raises(ContractViolationError, match="Temporal constraints"):
            llm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}])

    def test_get_usage_summary(self) -> None:
        """Test getting usage summary."""
        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        llm = ContractedLLM(contract)

        summary = llm.get_usage_summary()

        assert "usage" in summary
        assert "percentages" in summary
        assert "contract_state" in summary

    def test_add_callback(self) -> None:
        """Test adding event callbacks."""
        contract = Contract(id="test", name="Test")
        llm = ContractedLLM(contract)

        events = []

        def callback(event):  # type: ignore[no-untyped-def]
            events.append(event)

        llm.add_callback(callback)
        llm.start()

        assert len(events) == 1  # Start event

    def test_context_manager(self) -> None:
        """Test using ContractedLLM as context manager."""
        contract = Contract(id="test", name="Test")
        llm = ContractedLLM(contract, auto_start=False)

        assert not llm._started

        with llm:
            assert llm._started

        assert not llm._started

    def test_repr(self) -> None:
        """Test string representation."""
        contract = Contract(id="test-123", name="Test")
        llm = ContractedLLM(contract, auto_start=False)

        repr_str = repr(llm)
        assert "test-123" in repr_str
        assert "NOT_STARTED" in repr_str

        llm.start()
        repr_str = repr(llm)
        assert "STARTED" in repr_str


class TestIntegration:
    """Integration tests for litellm wrapper."""

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_realistic_conversation(self, mock_completion: MagicMock) -> None:
        """Test realistic multi-turn conversation."""
        # Setup mock to return different responses
        responses = [
            {
                "choices": [{"message": {"content": "Hello! How can I help?"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                "model": "gpt-4o-mini",
            },
            {
                "choices": [{"message": {"content": "Sure, I can explain that."}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
                "model": "gpt-4o-mini",
            },
            {
                "choices": [{"message": {"content": "Is there anything else?"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                "model": "gpt-4o-mini",
            },
        ]
        mock_completion.side_effect = responses

        # Create contract with reasonable limits
        contract = Contract(
            id="conversation",
            name="Conversation Agent",
            resources=ResourceConstraints(tokens=500, api_calls=10, cost_usd=0.1),
        )

        llm = ContractedLLM(contract)

        # Turn 1
        response1 = llm.completion(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}]
        )
        assert response1 is not None
        assert llm.enforcer.monitor.usage.tokens == 30

        # Turn 2
        response2 = llm.completion(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "Explain"}]
        )
        assert response2 is not None
        assert llm.enforcer.monitor.usage.tokens == 95

        # Turn 3
        response3 = llm.completion(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "Thanks"}]
        )
        assert response3 is not None
        assert llm.enforcer.monitor.usage.tokens == 205

        # All calls should succeed
        assert llm.enforcer.monitor.usage.api_calls == 3
        assert not llm.enforcer.monitor.is_violated()

    @patch("agent_contracts.integrations.litellm_wrapper.completion")
    def test_budget_exhaustion(self, mock_completion: MagicMock) -> None:
        """Test behavior when budget is exhausted."""
        mock_completion.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 300, "total_tokens": 800},
            "model": "gpt-4",
        }

        # Tight budget
        contract = Contract(
            id="budget-test",
            name="Budget Test",
            resources=ResourceConstraints(tokens=1000, api_calls=3),
        )

        llm = ContractedLLM(contract, strict_mode=True)

        # First call should succeed
        llm.completion(model="gpt-4", messages=[{"role": "user", "content": "Question 1"}])

        # Second call should violate (800 + 800 > 1000)
        with pytest.raises(ContractViolationError):
            llm.completion(model="gpt-4", messages=[{"role": "user", "content": "Question 2"}])

        assert contract.state == ContractState.VIOLATED
