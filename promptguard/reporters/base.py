"""Abstract base class for reporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from promptguard.runner.engine import RunResult


class Reporter(ABC):
    """Abstract base class for report generators.

    Reporters convert RunResult into formatted output
    for different consumers (humans, CI, etc.).
    """

    name: str = "base"

    @abstractmethod
    def emit(self, result: RunResult, verbose: bool = False) -> str:
        """Generate report content from run result.

        Args:
            result: Complete run result
            verbose: Include detailed case-by-case output

        Returns:
            Formatted report string
        """
        ...
