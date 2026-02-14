"""Output contract evaluator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from promptguard.bundle.models import OutputContract


@dataclass
class EvaluationResult:
    """Result of evaluating a single output against a contract.

    Attributes:
        passed: Whether all checks passed
        schema_valid: Whether output matched JSON schema
        schema_errors: List of schema validation errors
        invariants_passed: Dict of invariant name to pass/fail
        invariant_errors: Dict of invariant name to error message
        output_parsed: Parsed output (if JSON)
    """

    passed: bool
    schema_valid: bool = True
    schema_errors: list[str] = field(default_factory=list)
    invariants_passed: dict[str, bool] = field(default_factory=dict)
    invariant_errors: dict[str, str] = field(default_factory=dict)
    output_parsed: Any = None


class Evaluator:
    """Evaluates LLM outputs against output contracts.

    Supports:
    - JSON Schema validation
    - Invariant rules (Python expressions)
    """

    def __init__(self, contract: OutputContract | None, bundle_dir: Path) -> None:
        """Initialize evaluator with contract.

        Args:
            contract: Output contract to evaluate against
            bundle_dir: Bundle directory for resolving schema paths
        """
        self.contract = contract
        self.bundle_dir = bundle_dir
        self._schema: dict[str, Any] | None = None

    @property
    def schema(self) -> dict[str, Any] | None:
        """Load JSON schema if configured."""
        if self._schema is None and self.contract and self.contract.schema_path:
            schema_path = self.bundle_dir / self.contract.schema_path
            with open(schema_path, "r", encoding="utf-8") as f:
                self._schema = json.load(f)
        return self._schema

    def evaluate(self, output: str) -> EvaluationResult:
        """Evaluate output against the contract.

        Args:
            output: Raw LLM output string

        Returns:
            EvaluationResult with detailed pass/fail info
        """
        if self.contract is None:
            # No contract = everything passes
            return EvaluationResult(passed=True)

        result = EvaluationResult(passed=True)

        # Try to parse output as JSON
        output_parsed: Any = None
        try:
            output_parsed = json.loads(output)
            result.output_parsed = output_parsed
        except json.JSONDecodeError:
            # Not JSON, that's okay for some contracts
            pass

        # Schema validation
        if self.schema is not None:
            if output_parsed is None:
                result.schema_valid = False
                result.schema_errors = ["Output is not valid JSON"]
                result.passed = False
            else:
                schema_errors = self._validate_schema(output_parsed)
                if schema_errors:
                    result.schema_valid = False
                    result.schema_errors = schema_errors
                    result.passed = False

        # Invariant evaluation
        if self.contract.invariants:
            for invariant in self.contract.invariants:
                passed, error = self._evaluate_invariant(invariant, output, output_parsed)
                result.invariants_passed[invariant] = passed
                if error:
                    result.invariant_errors[invariant] = error
                if not passed:
                    result.passed = False

        return result

    def _validate_schema(self, data: Any) -> list[str]:
        """Validate data against JSON schema.

        Args:
            data: Parsed JSON data to validate

        Returns:
            List of validation error messages
        """
        if self.schema is None:
            return []

        validator = jsonschema.Draft7Validator(self.schema)
        errors = list(validator.iter_errors(data))

        return [f"{e.json_path}: {e.message}" for e in errors]

    def _evaluate_invariant(
        self, invariant: str, output_raw: str, output_parsed: Any
    ) -> tuple[bool, str | None]:
        """Evaluate a single invariant expression.

        Args:
            invariant: Python expression to evaluate
            output_raw: Raw output string
            output_parsed: Parsed JSON output (or None)

        Returns:
            Tuple of (passed, error_message)
        """
        # Create evaluation context
        context = {
            "output": output_parsed if output_parsed is not None else output_raw,
            "raw_output": output_raw,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "isinstance": isinstance,
            "all": all,
            "any": any,
        }

        try:
            # Safely evaluate the invariant
            result = eval(invariant, {"__builtins__": {}}, context)
            if not isinstance(result, bool):
                return False, f"Invariant did not return boolean: {type(result).__name__}"
            return result, None
        except Exception as e:
            return False, f"Invariant evaluation error: {e}"
