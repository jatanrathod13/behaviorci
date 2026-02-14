"""Tests for mock provider."""

from __future__ import annotations

import pytest

from promptguard.providers.mock import MockProvider
from promptguard.exceptions import ProviderAPIError


class TestMockProvider:
    """Tests for MockProvider class."""

    @pytest.mark.asyncio
    async def test_default_response(self) -> None:
        """Test default deterministic response."""
        provider = MockProvider()
        response = await provider.generate("Hello")

        assert response.content is not None
        assert response.model == "mock-v1"
        assert response.latency_ms == 10.0

    @pytest.mark.asyncio
    async def test_custom_responses(self) -> None:
        """Test custom response mapping."""
        provider = MockProvider(
            responses={"Hello": "World"},
            default_response="Default",
        )

        r1 = await provider.generate("Hello")
        assert r1.content == "World"

        r2 = await provider.generate("Other")
        assert r2.content == "Default"

    @pytest.mark.asyncio
    async def test_default_response_fallback(self) -> None:
        """Test default_response when no match."""
        provider = MockProvider(default_response="Fallback")
        response = await provider.generate("Anything")

        assert response.content == "Fallback"

    @pytest.mark.asyncio
    async def test_fail_on_trigger(self) -> None:
        """Test failure trigger."""
        provider = MockProvider(fail_on=["error"])

        with pytest.raises(ProviderAPIError, match="Mock failure"):
            await provider.generate("This will error out")

    @pytest.mark.asyncio
    async def test_call_history(self) -> None:
        """Test call history tracking."""
        provider = MockProvider()

        await provider.generate("First")
        await provider.generate("Second")

        assert len(provider.call_history) == 2
        assert provider.call_history[0] == "First"
        assert provider.call_history[1] == "Second"

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Test history reset."""
        provider = MockProvider()
        await provider.generate("Test")
        provider.reset()

        assert len(provider.call_history) == 0

    def test_validate_config(self) -> None:
        """Test config validation always passes."""
        provider = MockProvider()
        provider.validate_config()  # Should not raise
