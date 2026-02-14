"""OpenAI provider adapter."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from promptguard.exceptions import ProviderAPIError, ProviderConfigError
from promptguard.providers.base import LLMProvider, ProviderResponse
from promptguard.providers.registry import register_provider


@register_provider("openai")
class OpenAIProvider(LLMProvider):
    """OpenAI API provider adapter.

    Supports GPT-4, GPT-3.5, and other OpenAI models.
    Requires OPENAI_API_KEY environment variable.
    """

    API_BASE = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            model: Model to use (default: gpt-4o-mini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: API key (defaults to OPENAI_API_KEY env var)
            **kwargs: Additional options
        """
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    @property
    def default_model(self) -> str:
        """Default model for OpenAI."""
        return "gpt-4o-mini"

    def validate_config(self) -> None:
        """Validate OpenAI configuration."""
        if not self.api_key:
            raise ProviderConfigError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

    async def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        """Generate completion using OpenAI API.

        Args:
            prompt: Prompt text
            **kwargs: Additional parameters

        Returns:
            ProviderResponse with generated content
        """
        self.validate_config()

        # Merge kwargs with defaults
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        model = kwargs.get("model", self.model)

        # Build request
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.perf_counter()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.API_BASE}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_body = e.response.text
                raise ProviderAPIError(
                    f"OpenAI API error ({e.response.status_code}): {error_body}"
                ) from e
            except httpx.RequestError as e:
                raise ProviderAPIError(f"OpenAI request failed: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage")

        return ProviderResponse(
            content=content,
            raw_response=data,
            model=data.get("model"),
            usage=usage,
            latency_ms=latency_ms,
        )
