"""Shared test fixtures for PromptGuard tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator

import pytest


@pytest.fixture
def tmp_bundle_dir(tmp_path: Path) -> Path:
    """Create a temporary bundle directory."""
    bundle_dir = tmp_path / "test-bundle"
    bundle_dir.mkdir()
    return bundle_dir


@pytest.fixture
def sample_bundle_yaml() -> str:
    """Sample bundle.yaml content."""
    return """\
name: test-bundle
version: "1.0"
description: Test bundle for unit tests

prompt_path: prompt.md
dataset_path: dataset.jsonl

output_contract:
  schema_path: schema.json
  invariants:
    - "len(raw_output) < 500"

thresholds:
  - metric: pass_rate
    operator: ">="
    value: 0.8

provider:
  name: mock
  temperature: 0.0
"""


@pytest.fixture
def sample_prompt() -> str:
    """Sample prompt template."""
    return "Answer this question: {{ question }}"


@pytest.fixture
def sample_dataset() -> str:
    """Sample dataset JSONL."""
    cases = [
        {"input": {"question": "What is 2+2?"}, "expected_output": "4"},
        {"input": {"question": "What is the sky?"}, "expected_output": "blue"},
        {"input": {"question": "Capital of France?"}, "expected_output": "Paris"},
    ]
    return "\n".join(json.dumps(c) for c in cases)


@pytest.fixture
def sample_schema() -> str:
    """Sample JSON schema."""
    return json.dumps({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["answer"],
        "properties": {
            "answer": {"type": "string"}
        }
    })


@pytest.fixture
def complete_bundle(
    tmp_bundle_dir: Path,
    sample_bundle_yaml: str,
    sample_prompt: str,
    sample_dataset: str,
    sample_schema: str,
) -> Path:
    """Create a complete bundle with all files."""
    (tmp_bundle_dir / "bundle.yaml").write_text(sample_bundle_yaml)
    (tmp_bundle_dir / "prompt.md").write_text(sample_prompt)
    (tmp_bundle_dir / "dataset.jsonl").write_text(sample_dataset)
    (tmp_bundle_dir / "schema.json").write_text(sample_schema)
    return tmp_bundle_dir / "bundle.yaml"
