"""Tests for dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptguard.bundle.dataset import Dataset, DatasetCase
from promptguard.exceptions import DatasetError


class TestDataset:
    """Tests for Dataset class."""

    def test_load_valid_dataset(self, tmp_path: Path) -> None:
        """Test loading a valid JSONL dataset."""
        dataset_path = tmp_path / "test.jsonl"
        cases = [
            {"input": {"q": "A"}, "expected_output": "1"},
            {"input": {"q": "B"}, "expected_output": "2"},
        ]
        dataset_path.write_text("\n".join(json.dumps(c) for c in cases))

        dataset = Dataset.load(dataset_path)

        assert len(dataset) == 2
        assert dataset[0].input == {"q": "A"}
        assert dataset[1].expected_output == "2"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test loading non-existent file."""
        with pytest.raises(DatasetError, match="not found"):
            Dataset.load(tmp_path / "missing.jsonl")

    def test_load_empty_dataset(self, tmp_path: Path) -> None:
        """Test loading empty dataset."""
        dataset_path = tmp_path / "empty.jsonl"
        dataset_path.write_text("")

        with pytest.raises(DatasetError, match="empty"):
            Dataset.load(dataset_path)

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON."""
        dataset_path = tmp_path / "invalid.jsonl"
        dataset_path.write_text("not json")

        with pytest.raises(DatasetError, match="Invalid JSON"):
            Dataset.load(dataset_path)

    def test_load_missing_input_field(self, tmp_path: Path) -> None:
        """Test loading case without input field."""
        dataset_path = tmp_path / "no_input.jsonl"
        dataset_path.write_text('{"output": "test"}')

        with pytest.raises(DatasetError, match="missing required 'input'"):
            Dataset.load(dataset_path)

    def test_iteration(self, tmp_path: Path) -> None:
        """Test dataset iteration."""
        dataset_path = tmp_path / "iter.jsonl"
        dataset_path.write_text('{"input": {"q": "A"}}\n{"input": {"q": "B"}}')

        dataset = Dataset.load(dataset_path)
        inputs = [c.input["q"] for c in dataset]

        assert inputs == ["A", "B"]


class TestDatasetCase:
    """Tests for DatasetCase class."""

    def test_auto_id_generation(self) -> None:
        """Test automatic ID generation."""
        case = DatasetCase(input={"q": "test"})
        assert case.id is not None
        assert case.id.startswith("case_")

    def test_explicit_id(self) -> None:
        """Test explicit ID is preserved."""
        case = DatasetCase(id="my-id", input={"q": "test"})
        assert case.id == "my-id"

    def test_metadata(self) -> None:
        """Test metadata handling."""
        case = DatasetCase(
            input={"q": "test"},
            metadata={"source": "manual", "priority": 1},
        )
        assert case.metadata["source"] == "manual"
