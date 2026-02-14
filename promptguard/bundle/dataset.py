"""Dataset loading and parsing for Behavior Bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from promptguard.exceptions import DatasetError


@dataclass
class DatasetCase:
    """A single test case from the dataset.

    Attributes:
        id: Unique identifier for this case (auto-generated if not provided)
        input: Input data to substitute into the prompt
        expected_output: Optional expected output for comparison
        metadata: Additional metadata for the case
    """

    input: Dict[str, Any]
    id: Optional[str] = None
    expected_output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if self.id is None:
            # Use hash of input as ID
            self.id = f"case_{hash(json.dumps(self.input, sort_keys=True)) % 10000:04d}"


@dataclass
class Dataset:
    """A collection of test cases loaded from a JSONL file.

    Supports iteration and indexing for test execution.
    """

    cases: List[DatasetCase]
    path: Path
    
    def __len__(self) -> int:
        """Return number of cases in dataset."""
        return len(self.cases)

    def __iter__(self) -> Iterator[DatasetCase]:
        """Iterate over cases."""
        return iter(self.cases)

    def __getitem__(self, index: int) -> DatasetCase:
        """Get case by index."""
        return self.cases[index]

    @classmethod
    def load(cls, path: "Path | str") -> Dataset:
        """Load dataset from a JSONL file.

        Each line should be a JSON object with at least an 'input' field.
        Optional fields: 'id', 'expected_output', 'metadata'.

        Args:
            path: Path to the JSONL file

        Returns:
            Dataset instance with parsed cases

        Raises:
            DatasetError: If file cannot be loaded or parsed
        """
        path = Path(path)
        
        if not path.exists():
            raise DatasetError(f"Dataset file not found: {path}")
        
        if not path.suffix == ".jsonl":
            raise DatasetError(f"Dataset must be a .jsonl file, got: {path.suffix}")

        cases: List[DatasetCase] = []
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise DatasetError(
                            f"Invalid JSON on line {line_num}: {e}"
                        ) from e
                    
                    if not isinstance(data, dict):
                        raise DatasetError(
                            f"Line {line_num} must be a JSON object, got {type(data).__name__}"
                        )
                    
                    if "input" not in data:
                        raise DatasetError(
                            f"Line {line_num} missing required 'input' field"
                        )
                    
                    case = DatasetCase(
                        id=data.get("id"),
                        input=data["input"],
                        expected_output=data.get("expected_output"),
                        metadata=data.get("metadata", {}),
                    )
                    cases.append(case)
        
        except OSError as e:
            raise DatasetError(f"Failed to read dataset file: {e}") from e

        if not cases:
            raise DatasetError(f"Dataset is empty: {path}")

        return cls(cases=cases, path=path)
