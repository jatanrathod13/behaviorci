"""Mock provider for testing without API calls."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from promptguard.providers.base import LLMProvider, ProviderResponse
from promptguard.providers.registry import register_provider


@register_provider("mock")
class MockProvider(LLMProvider):
    """Mock LLM provider for testing.

    Generates deterministic responses based on input hash.
    Useful for:
    - Local development without API keys
    - CI testing
    - Validating bundle structure
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        responses: dict[str, str] | None = None,
        default_response: str | None = None,
        latency_ms: float = 10.0,
        fail_on: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize mock provider.

        Args:
            model: Model name (ignored, always "mock")
            temperature: Temperature (ignored)
            max_tokens: Max tokens (ignored)
            responses: Optional dict mapping prompts to responses
            default_response: Default response when no match found
            latency_ms: Simulated latency
            fail_on: List of prompt substrings that trigger errors
            **kwargs: Additional options
        """
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.responses = responses or {}
        self.default_response = default_response
        self.latency = latency_ms
        self.fail_on = fail_on or []
        self.call_history: list[str] = []

    @property
    def default_model(self) -> str:
        """Default model name."""
        return "mock-v1"

    def validate_config(self) -> None:
        """Mock provider is always valid."""
        pass

    async def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        """Generate mock response.

        Response is determined by:
        1. Exact match in responses dict
        2. Default response if set
        3. Deterministic hash-based response

        Args:
            prompt: Input prompt
            **kwargs: Ignored

        Returns:
            ProviderResponse with mock content
        """
        self.call_history.append(prompt)

        # Check for failure triggers
        for trigger in self.fail_on:
            if trigger in prompt:
                from promptguard.exceptions import ProviderAPIError

                raise ProviderAPIError(f"Mock failure triggered by: {trigger}")

        # Look for exact match
        if prompt in self.responses:
            content = self.responses[prompt]
        elif self.default_response is not None:
            content = self.default_response
        else:
            # Generate deterministic response from hash
            content = self._generate_deterministic(prompt)

        return ProviderResponse(
            content=content,
            raw_response={"mock": True, "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()},
            model=self.default_model,
            usage={"input_tokens": len(prompt) // 4, "output_tokens": len(content) // 4},
            latency_ms=self.latency,
        )

    def _generate_deterministic(self, prompt: str) -> str:
        """Generate deterministic response from prompt hash.

        Args:
            prompt: Input prompt

        Returns:
            Deterministic response string
        """
        # Create hash for determinism
        h = hashlib.sha256(prompt.encode()).hexdigest()[:8]

        # Generate structured mock response
        response = {
            "id": f"mock-{h}",
            "result": f"Mock response for prompt hash {h}",
            "confidence": 0.95,
        }

        return json.dumps(response)

    def reset(self) -> None:
        """Reset call history."""
        self.call_history.clear()
