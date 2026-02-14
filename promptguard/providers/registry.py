"""Provider registry for dynamic provider lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from promptguard.providers.base import LLMProvider

# Global registry mapping provider names to classes
_PROVIDERS: dict[str, type[LLMProvider]] = {}


def register_provider(name: str) -> type[LLMProvider]:
    """Decorator to register a provider class.

    Args:
        name: Provider name for lookup

    Returns:
        Decorator function
    """

    def decorator(cls: type[LLMProvider]) -> type[LLMProvider]:
        _PROVIDERS[name.lower()] = cls
        cls.name = name.lower()
        return cls

    return decorator  # type: ignore[return-value]


def get_provider(name: str, **kwargs: object) -> LLMProvider:
    """Get a provider instance by name.

    Args:
        name: Provider name (case-insensitive)
        **kwargs: Configuration passed to provider constructor

    Returns:
        Initialized provider instance

    Raises:
        ValueError: If provider name is not registered
    """
    # Import providers to trigger registration
    from promptguard.providers import openai, anthropic, mock  # noqa: F401

    name_lower = name.lower()
    if name_lower not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS.keys()))
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")

    provider_cls = _PROVIDERS[name_lower]
    return provider_cls(**kwargs)


def list_providers() -> list[str]:
    """List all registered provider names."""
    # Import to ensure registration
    from promptguard.providers import openai, anthropic, mock  # noqa: F401

    return sorted(_PROVIDERS.keys())
