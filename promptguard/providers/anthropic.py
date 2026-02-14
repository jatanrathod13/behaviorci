"""Anthropic (Claude) provider adapter."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from promptguard.exceptions import ProviderAPIError, ProviderConfigError
from promptguard.providers.base import LLMProvider, ProviderResponse
from promptguard.providers.registry import register_provider


@register_provider("anthropic")
class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider adapter.

    Supports Claude 3 and other Anthropic models.
    Requires ANTHROPIC_API_KEY environment variable.
    """

    API_BASE = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            model: Model to use (default: claude-3-haiku-20240307)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (default: 1024)
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional options
        """
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        # Anthropic requires max_tokens
        if self.max_tokens is None:
            self.max_tokens = 1024

    @property
    def default_model(self) -> str:
        """Default model for Anthropic."""
        return "claude-3-haiku-20240307"

    def validate_config(self) -> None:
        """Validate Anthropic configuration."""
        if not self.api_key:
            raise ProviderConfigError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )

    async def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        """Generate completion using Anthropic API.

        Args:
            prompt: Prompt text
            **kwargs: Additional parameters

        Returns:
            ProviderResponse with generated content
        """
        self.validate_config()

        # Merge kwargs with defaults
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens) or 1024
        model = kwargs.get("model", self.model)

        # Build request (Anthropic message format)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

        start_time = time.perf_counter()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.API_BASE}/messages",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_body = e.response.text
                raise ProviderAPIError(
                    f"Anthropic API error ({e.response.status_code}): {error_body}"
                ) from e
            except httpx.RequestError as e:
                raise ProviderAPIError(f"Anthropic request failed: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000
        data = response.json()

        # Extract content from Anthropic response format
        content_blocks = data.get("content", [])
        content = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )

        usage = data.get("usage")

        return ProviderResponse(
            content=content,
            raw_response=data,
            model=data.get("model"),
            usage=usage,
            latency_ms=latency_ms,
        )
