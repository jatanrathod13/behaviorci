"""Loader for Agent Bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateError
from pydantic import ValidationError

from behaviorci.agent.models import (
    AgentBundle,
    EvaluationConfig,
    InvariantConfig,
    Task,
    ThresholdConfig,
)
from behaviorci.exceptions import BundleNotFoundError, BundleValidationError

if TYPE_CHECKING:
    from jinja2 import Template


class AgentLoader:
    """Loader for Agent Bundles.

    Handles parsing agent.yaml, loading tasks.jsonl, evaluation.yaml,
    thresholds.yaml, and resolving file references.
    """

    def __init__(self, bundle_path: Union[Path, str]) -> None:
        """Initialize loader with path to agent.yaml.

        Args:
            bundle_path: Path to the agent.yaml file (or directory containing it)
        """
        bundle_path = Path(bundle_path)
        if bundle_path.is_dir():
            bundle_path = bundle_path / "agent.yaml"

        self.bundle_path = bundle_path
        self.bundle_dir = self.bundle_path.parent

        self._config: Optional[AgentBundle] = None
        self._system_prompt: Optional[str] = None
        self._system_prompt_template: Optional["Template"] = None
        self._tasks: Optional[List[Task]] = None
        self._evaluation: Optional[EvaluationConfig] = None
        self._thresholds: Optional[List[ThresholdConfig]] = None

    @property
    def config(self) -> AgentBundle:
        """Get validated agent configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    @property
    def system_prompt(self) -> str:
        """Get system prompt content (rendered)."""
        if self._system_prompt is None:
            self._system_prompt = self._load_system_prompt()
        return self._system_prompt

    @property
    def system_prompt_template(self) -> "Template":
        """Get Jinja2 template for system prompt."""
        if self._system_prompt_template is None:
            self._system_prompt_template = self._load_system_prompt_template()
        return self._system_prompt_template

    @property
    def tasks(self) -> List[Task]:
        """Get loaded tasks."""
        if self._tasks is None:
            self._tasks = self._load_tasks()
        return self._tasks

    @property
    def evaluation(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        if self._evaluation is None:
            self._evaluation = self._load_evaluation()
        return self._evaluation

    @property
    def thresholds(self) -> List[ThresholdConfig]:
        """Get threshold configuration."""
        if self._thresholds is None:
            self._thresholds = self._load_thresholds()
        return self._thresholds

    def _load_config(self) -> AgentBundle:
        """Load and validate agent.yaml configuration.

        Returns:
            Validated AgentBundle

        Raises:
            BundleNotFoundError: If agent.yaml doesn't exist
            BundleValidationError: If configuration is invalid
        """
        if not self.bundle_path.exists():
            raise BundleNotFoundError(f"Agent bundle not found: {self.bundle_path}")

        try:
            with open(self.bundle_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise BundleValidationError(f"Invalid YAML in agent.yaml: {e}") from e
        except OSError as e:
            raise BundleNotFoundError(f"Failed to read agent.yaml: {e}") from e

        if not isinstance(raw_config, dict):
            raise BundleValidationError("Agent bundle must be a YAML mapping")

        try:
            config = AgentBundle.model_validate(raw_config)
        except ValidationError as e:
            errors = []
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                errors.append(f"  - {loc}: {err['msg']}")
            raise BundleValidationError(
                f"Agent bundle validation failed:\n" + "\n".join(errors)
            ) from e

        self._validate_file_references(config)

        return config

    def _validate_file_references(self, config: AgentBundle) -> None:
        """Validate that all file references in config exist.

        Args:
            config: AgentBundle configuration to validate

        Raises:
            BundleValidationError: If any referenced file is missing
        """
        # Validate system-prompt.md exists
        system_prompt_path = self.bundle_dir / "system-prompt.md"
        if not system_prompt_path.exists():
            raise BundleValidationError(
                f"System prompt file not found: {system_prompt_path}"
            )

        # Validate tasks.jsonl exists
        tasks_path = self.bundle_dir / "tasks.jsonl"
        if not tasks_path.exists():
            raise BundleValidationError(f"Tasks file not found: {tasks_path}")

    def _load_system_prompt(self) -> str:
        """Load system prompt file content.

        Returns:
            System prompt string

        Raises:
            BundleValidationError: If system prompt cannot be loaded
        """
        system_prompt_path = self.bundle_dir / "system-prompt.md"

        try:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            raise BundleValidationError(
                f"Failed to read system prompt: {e}"
            ) from e

    def _load_system_prompt_template(self) -> "Template":
        """Load system prompt as Jinja2 template.

        Returns:
            Jinja2 Template object

        Raises:
            BundleValidationError: If template cannot be loaded
        """
        system_prompt_path = self.bundle_dir / "system-prompt.md"

        try:
            env = Environment(
                loader=FileSystemLoader(self.bundle_dir),
                autoescape=False,
            )
            return env.get_template("system-prompt.md")
        except TemplateError as e:
            raise BundleValidationError(
                f"Failed to load system prompt template: {e}"
            ) from e

    def _load_tasks(self) -> List[Task]:
        """Load tasks from JSONL file.

        Returns:
            List of Task objects

        Raises:
            BundleValidationError: If tasks cannot be loaded
        """
        tasks_path = self.bundle_dir / "tasks.jsonl"

        try:
            tasks = []
            with open(tasks_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        task_data = json.loads(line)
                        task = Task.model_validate(task_data)
                        tasks.append(task)
                    except json.JSONDecodeError as e:
                        raise BundleValidationError(
                            f"Invalid JSON in tasks.jsonl line {line_num}: {e}"
                        )
                    except ValidationError as e:
                        raise BundleValidationError(
                            f"Invalid task format in tasks.jsonl line {line_num}: {e}"
                        )
            return tasks
        except OSError as e:
            raise BundleValidationError(f"Failed to read tasks file: {e}") from e

    def _load_evaluation(self) -> EvaluationConfig:
        """Load evaluation configuration.

        Returns:
            EvaluationConfig (with defaults if file doesn't exist)
        """
        eval_path = self.bundle_dir / "evaluation.yaml"

        if not eval_path.exists():
            return EvaluationConfig()

        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                raw_eval = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise BundleValidationError(f"Invalid YAML in evaluation.yaml: {e}") from e

        if not isinstance(raw_eval, dict):
            raise BundleValidationError("evaluation.yaml must be a YAML mapping")

        try:
            # Handle the nested 'evaluation' key in the YAML
            if "evaluation" in raw_eval:
                raw_eval = raw_eval["evaluation"]

            return EvaluationConfig.model_validate(raw_eval)
        except ValidationError as e:
            errors = []
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                errors.append(f"  - {loc}: {err['msg']}")
            raise BundleValidationError(
                f"Evaluation validation failed:\n" + "\n".join(errors)
            ) from e

    def _load_thresholds(self) -> List[ThresholdConfig]:
        """Load threshold configuration.

        Returns:
            List of ThresholdConfig (empty list if file doesn't exist)
        """
        thresholds_path = self.bundle_dir / "thresholds.yaml"

        if not thresholds_path.exists():
            return []

        try:
            with open(thresholds_path, "r", encoding="utf-8") as f:
                raw_thresholds = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise BundleValidationError(
                f"Invalid YAML in thresholds.yaml: {e}"
            ) from e

        if not isinstance(raw_thresholds, dict):
            raise BundleValidationError("thresholds.yaml must be a YAML mapping")

        try:
            threshold_list = raw_thresholds.get("thresholds", [])
            return [ThresholdConfig.model_validate(t) for t in threshold_list]
        except ValidationError as e:
            errors = []
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                errors.append(f"  - {loc}: {err['msg']}")
            raise BundleValidationError(
                f"Threshold validation failed:\n" + "\n".join(errors)
            ) from e

    def render_system_prompt(self, variables: Dict[str, object] = None) -> str:
        """Render system prompt template with given variables.

        Args:
            variables: Variables to substitute into template

        Returns:
            Rendered prompt string
        """
        if variables is None:
            return self.system_prompt
        return self.system_prompt_template.render(**variables)

    def get_tools(self) -> Dict[str, any]:
        """Get tool definitions for the agent.

        Returns:
            Dictionary mapping tool names to their configurations
        """
        tools = {}
        for tool in self.config.tools:
            if hasattr(tool, "name"):
                tools[tool.name] = tool
            elif hasattr(tool, "server"):
                # MCP tool - add its commands
                for cmd in tool.commands:
                    tools[cmd] = tool
        return tools


def load_agent_bundle(path: Union[Path, str]) -> AgentLoader:
    """Convenience function to load an agent bundle.

    Args:
        path: Path to agent.yaml or directory containing it

    Returns:
        Initialized AgentLoader with validated config
    """
    loader = AgentLoader(path)
    # Force validation on load
    _ = loader.config
    return loader
