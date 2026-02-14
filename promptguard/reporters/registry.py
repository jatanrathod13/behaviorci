"""Reporter registry for dynamic reporter lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from promptguard.reporters.base import Reporter

# Global registry mapping reporter names to classes
_REPORTERS: dict[str, type[Reporter]] = {}


def register_reporter(name: str) -> type[Reporter]:
    """Decorator to register a reporter class.

    Args:
        name: Reporter name for lookup

    Returns:
        Decorator function
    """

    def decorator(cls: type[Reporter]) -> type[Reporter]:
        _REPORTERS[name.lower()] = cls
        cls.name = name.lower()
        return cls

    return decorator  # type: ignore[return-value]


def get_reporter(name: str) -> Reporter:
    """Get a reporter instance by name.

    Args:
        name: Reporter name (case-insensitive)

    Returns:
        Initialized reporter instance

    Raises:
        ValueError: If reporter name is not registered
    """
    # Import reporters to trigger registration
    from promptguard.reporters import console, json_reporter, markdown  # noqa: F401

    name_lower = name.lower()
    if name_lower not in _REPORTERS:
        available = ", ".join(sorted(_REPORTERS.keys()))
        raise ValueError(f"Unknown reporter '{name}'. Available: {available}")

    reporter_cls = _REPORTERS[name_lower]
    return reporter_cls()


def list_reporters() -> list[str]:
    """List all registered reporter names."""
    from promptguard.reporters import console, json_reporter, markdown  # noqa: F401

    return sorted(_REPORTERS.keys())
