"""Bundle loader for parsing and validating Behavior Bundles."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateError
from pydantic import ValidationError

from promptguard.bundle.dataset import Dataset
from promptguard.bundle.models import BundleConfig
from promptguard.exceptions import BundleNotFoundError, BundleValidationError

if TYPE_CHECKING:
    from jinja2 import Template


class BundleLoader:
    """Loader for Behavior Bundles.

    Handles parsing bundle.yaml, validating configuration,
    resolving file references, and loading prompt templates.
    """

    def __init__(self, bundle_path: Union[Path, str]) -> None:
        """Initialize loader with path to bundle.yaml.

        Args:
            bundle_path: Path to the bundle.yaml file
        """
        self.bundle_path = Path(bundle_path)
        self.bundle_dir = self.bundle_path.parent
        self._config: Optional[BundleConfig] = None
        self._prompt_template: Optional["Template"] = None
        self._dataset: Optional[Dataset] = None

    @property
    def config(self) -> BundleConfig:
        """Get validated bundle configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    @property
    def prompt_template(self) -> "Template":
        """Get Jinja2 prompt template."""
        if self._prompt_template is None:
            self._prompt_template = self._load_prompt()
        return self._prompt_template

    @property
    def dataset(self) -> Dataset:
        """Get loaded dataset."""
        if self._dataset is None:
            self._dataset = self._load_dataset()
        return self._dataset

    def _load_config(self) -> BundleConfig:
        """Load and validate bundle.yaml configuration.

        Returns:
            Validated BundleConfig

        Raises:
            BundleNotFoundError: If bundle.yaml doesn't exist
            BundleValidationError: If configuration is invalid
        """
        if not self.bundle_path.exists():
            raise BundleNotFoundError(f"Bundle file not found: {self.bundle_path}")

        try:
            with open(self.bundle_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise BundleValidationError(f"Invalid YAML in bundle: {e}") from e
        except OSError as e:
            raise BundleNotFoundError(f"Failed to read bundle file: {e}") from e

        if not isinstance(raw_config, dict):
            raise BundleValidationError("Bundle must be a YAML mapping")

        try:
            config = BundleConfig.model_validate(raw_config)
        except ValidationError as e:
            # Format validation errors nicely
            errors = []
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                errors.append(f"  - {loc}: {err['msg']}")
            raise BundleValidationError(
                f"Bundle validation failed:\n" + "\n".join(errors)
            ) from e

        # Validate that referenced files exist
        self._validate_file_references(config)

        return config

    def _validate_file_references(self, config: BundleConfig) -> None:
        """Validate that all file references in config exist.

        Args:
            config: Bundle configuration to validate

        Raises:
            BundleValidationError: If any referenced file is missing
        """
        prompt_path = self.bundle_dir / config.prompt_path
        if not prompt_path.exists():
            raise BundleValidationError(f"Prompt file not found: {prompt_path}")

        dataset_path = self.bundle_dir / config.dataset_path
        if not dataset_path.exists():
            raise BundleValidationError(f"Dataset file not found: {dataset_path}")

        if config.output_contract and config.output_contract.schema_path:
            schema_path = self.bundle_dir / config.output_contract.schema_path
            if not schema_path.exists():
                raise BundleValidationError(f"Schema file not found: {schema_path}")

    def _load_prompt(self) -> "Template":
        """Load prompt template using Jinja2.

        Returns:
            Jinja2 Template object

        Raises:
            BundleValidationError: If template cannot be loaded
        """
        config = self.config
        prompt_path = self.bundle_dir / config.prompt_path

        try:
            env = Environment(
                loader=FileSystemLoader(self.bundle_dir),
                autoescape=False,  # Prompts shouldn't be HTML-escaped
            )
            template = env.get_template(config.prompt_path)
            return template
        except TemplateError as e:
            raise BundleValidationError(f"Failed to load prompt template: {e}") from e

    def _load_dataset(self) -> Dataset:
        """Load dataset from JSONL file.

        Returns:
            Dataset instance

        Raises:
            BundleValidationError: If dataset cannot be loaded
        """
        config = self.config
        dataset_path = self.bundle_dir / config.dataset_path
        return Dataset.load(dataset_path)

    def render_prompt(self, variables: Dict[str, object]) -> str:
        """Render prompt template with given variables.

        Args:
            variables: Variables to substitute into template

        Returns:
            Rendered prompt string
        """
        return self.prompt_template.render(**variables)


def load_bundle(path: Union[Path, str]) -> BundleLoader:
    """Convenience function to load a bundle.

    Args:
        path: Path to bundle.yaml

    Returns:
        Initialized BundleLoader with validated config
    """
    loader = BundleLoader(path)
    # Force validation on load
    _ = loader.config
    return loader
