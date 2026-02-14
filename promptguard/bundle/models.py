"""Pydantic models for Behavior Bundle specification."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ThresholdConfig(BaseModel):
    """Configuration for a single threshold check.

    Thresholds determine whether a bundle run passes or fails.
    """

    metric: str = Field(
        description="Metric to evaluate (e.g., 'pass_rate', 'schema_valid_rate')"
    )
    operator: Literal[">=", ">", "<=", "<", "=="] = Field(
        default=">=", description="Comparison operator"
    )
    value: float = Field(description="Threshold value to compare against")

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float) -> float:
        """Ensure threshold value is between 0 and 1 for rate metrics."""
        if v < 0:
            raise ValueError("Threshold value must be non-negative")
        return v


class OutputContract(BaseModel):
    """Output contract defining expected structure and invariants.

    Contracts consist of:
    - JSON schema for structure validation
    - Invariants as string rules to evaluate
    """

    schema_path: Optional[str] = Field(
        default=None, description="Path to JSON schema file (relative to bundle)"
    )
    invariants: List[str] = Field(
        default_factory=list,
        description="List of invariant rules to check (e.g., 'len(output) < 500')",
    )


class ProviderConfig(BaseModel):
    """Configuration for the LLM provider."""

    name: str = Field(default="openai", description="Provider name")
    model: Optional[str] = Field(default=None, description="Model to use (provider-specific)")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Maximum tokens to generate")


class BundleConfig(BaseModel):
    """Configuration for a Behavior Bundle.

    A Behavior Bundle defines expected LLM behavior as:
    - Prompt template
    - Test dataset
    - Output contract (optional)
    - Pass/fail thresholds
    """

    name: str = Field(description="Human-readable bundle name")
    version: str = Field(default="1.0", description="Bundle version")
    description: Optional[str] = Field(default=None, description="Bundle description")

    # Required file references
    prompt_path: str = Field(description="Path to prompt template (relative to bundle)")
    dataset_path: str = Field(description="Path to dataset file (relative to bundle)")

    # Optional configuration
    output_contract: Optional[OutputContract] = Field(
        default=None, description="Output contract for validation"
    )
    thresholds: List[ThresholdConfig] = Field(
        default_factory=list, description="Pass/fail thresholds"
    )
    provider: ProviderConfig = Field(
        default_factory=ProviderConfig, description="LLM provider configuration"
    )

    # Execution options
    parallel: bool = Field(default=False, description="Run cases in parallel")
    timeout: float = Field(default=30.0, gt=0, description="Timeout per case in seconds")
    retries: int = Field(default=0, ge=0, description="Number of retries on failure")

    @field_validator("prompt_path", "dataset_path")
    @classmethod
    def validate_path_not_empty(cls, v: str) -> str:
        """Ensure paths are not empty strings."""
        if not v.strip():
            raise ValueError("Path cannot be empty")
        return v
