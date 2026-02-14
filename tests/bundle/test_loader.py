"""Tests for bundle loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptguard.bundle.loader import BundleLoader, load_bundle
from promptguard.bundle.models import BundleConfig
from promptguard.exceptions import BundleNotFoundError, BundleValidationError


class TestBundleLoader:
    """Tests for BundleLoader class."""

    def test_load_valid_bundle(self, complete_bundle: Path) -> None:
        """Test loading a valid bundle."""
        loader = load_bundle(complete_bundle)

        assert loader.config.name == "test-bundle"
        assert loader.config.version == "1.0"
        assert loader.config.provider.name == "mock"

    def test_load_missing_bundle(self, tmp_path: Path) -> None:
        """Test loading a non-existent bundle."""
        with pytest.raises(BundleNotFoundError):
            load_bundle(tmp_path / "missing.yaml")

    def test_load_invalid_yaml(self, tmp_bundle_dir: Path) -> None:
        """Test loading invalid YAML."""
        bundle_path = tmp_bundle_dir / "bundle.yaml"
        bundle_path.write_text("invalid: yaml: content:")

        with pytest.raises(BundleValidationError):
            load_bundle(bundle_path)

    def test_load_missing_required_field(self, tmp_bundle_dir: Path) -> None:
        """Test loading bundle with missing required field."""
        bundle_path = tmp_bundle_dir / "bundle.yaml"
        bundle_path.write_text("name: test\nversion: '1.0'\n")

        with pytest.raises(BundleValidationError):
            load_bundle(bundle_path)

    def test_load_missing_referenced_file(
        self, tmp_bundle_dir: Path, sample_bundle_yaml: str
    ) -> None:
        """Test loading bundle with missing referenced files."""
        bundle_path = tmp_bundle_dir / "bundle.yaml"
        bundle_path.write_text(sample_bundle_yaml)
        # Don't create the referenced files

        with pytest.raises(BundleValidationError, match="Prompt file not found"):
            load_bundle(bundle_path)

    def test_render_prompt(self, complete_bundle: Path) -> None:
        """Test prompt template rendering."""
        loader = load_bundle(complete_bundle)

        rendered = loader.render_prompt({"question": "What is 2+2?"})
        assert "What is 2+2?" in rendered

    def test_dataset_loading(self, complete_bundle: Path) -> None:
        """Test dataset is loaded correctly."""
        loader = load_bundle(complete_bundle)

        assert len(loader.dataset) == 3
        assert loader.dataset[0].input["question"] == "What is 2+2?"


class TestBundleConfig:
    """Tests for BundleConfig model."""

    def test_default_values(self) -> None:
        """Test default values are applied."""
        config = BundleConfig(
            name="test",
            prompt_path="prompt.md",
            dataset_path="dataset.jsonl",
        )

        assert config.version == "1.0"
        assert config.provider.name == "openai"
        assert config.provider.temperature == 0.0
        assert config.thresholds == []

    def test_empty_path_rejected(self) -> None:
        """Test empty paths are rejected."""
        with pytest.raises(ValueError, match="Path cannot be empty"):
            BundleConfig(
                name="test",
                prompt_path="",
                dataset_path="dataset.jsonl",
            )
