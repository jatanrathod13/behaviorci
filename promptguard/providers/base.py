"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderResponse:
    """Response from an LLM provider.

    Attributes:
        content: Generated text content
        raw_response: Full response from provider API
        model: Model used for generation
        usage: Token usage statistics
        latency_ms: Request latency in milliseconds
    """

    content: str
    raw_response: dict[str, Any] | None = None
    model: str | None = None
    usage: dict[str, int] | None = None
    latency_ms: float | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM provider adapters.

    Each provider implementation handles:
    - API authentication
    - Request formatting
    - Response parsing
    - Error handling
    """

    name: str = "base"

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize provider with configuration.

        Args:
            model: Model identifier (provider-specific)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific options
        """
        self.model = model or self.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.options = kwargs

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model to use if none specified."""
        ...

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        """Generate completion for the given prompt.

        Args:
            prompt: Full prompt text to send to the model
            **kwargs: Additional parameters to override defaults

        Returns:
            ProviderResponse with generated content

        Raises:
            ProviderConfigError: If provider is misconfigured
            ProviderAPIError: If API call fails
        """
        ...

    @abstractmethod
    def validate_config(self) -> None:
        """Validate provider configuration.

        Raises:
            ProviderConfigError: If configuration is invalid
        """
        ...
