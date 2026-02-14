"""Diff module for comparing runs against baselines."""

from behaviorci.diff.comparator import Comparator
from behaviorci.diff.models import DiffEntry, DiffResult, DiffType, MetricChange

__all__ = [
    "Comparator",
    "DiffEntry",
    "DiffResult",
    "DiffType",
    "MetricChange",
]
