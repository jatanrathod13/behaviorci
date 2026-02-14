# PromptGuard

[![PyPI version](https://img.shields.io/pypi/v/promptguard.svg)](https://pypi.org/project/promptguard/)
[![Python versions](https://img.shields.io/pypi/pyversions/promptguard.svg)](https://pypi.org/project/promptguard/)
[![CI](https://github.com/jatanrathod13/promptguard/actions/workflows/ci.yaml/badge.svg)](https://github.com/jatanrathod13/promptguard/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/promptguard)](https://pypi.org/project/promptguard/)

**CI/CD for LLM behavior.**

> Prompts don't ship until behavior passes tests.

```
┌─────────────────────────────────────────────────────────────┐
│  Every prompt change = automated test run → fail if regression
└─────────────────────────────────────────────────────────────┘
```

Think **Jest for prompts** — write tests, run CI, block bad changes.

---

## Why PromptGuard?

LLM behavior silently regresses:
- Prompt tweaks break edge cases you forgot about
- Model updates shift outputs unexpectedly
- Fixes aren't captured as tests — same bugs resurface
- No way to know if "improvements" actually work

**PromptGuard turns LLM behavior into engineering:**

| Before | After |
|--------|-------|
| Manual prompt testing | Automated test suite |
| Hoping it works | Thresholds: 90% pass rate required |
| Unknown regressions | Diff reports show exactly what broke |
| No CI blocking | Merge gate: fails build if quality drops |

---

## What Makes It Different

| Feature | PromptGuard | PromptFoo | Langfuse |
|---------|------------|-----------|----------|
| **Focus** | Testing & regression detection | Prompt comparison | Observability |
| **CI Gate** | ✅ Native fail-on-regression | ❌ No | ❌ No |
| **Agent Testing** | ✅ Tool-calling behavior | ❌ | ❌ |
| **Regression Diff** | ✅ Compare runs over time | ❌ | ❌ |
| **Threshold Gating** | ✅ Pass/fail based on metrics | Partial | ❌ |
| **Self-hosted** | ✅ 100% local | ✅ | ❌ (hosted) |

### Comparison

```
PromptFoo = "Which prompt is better?" (experimentation)
Langfuse  = "How is our prompt performing?" (monitoring)
PromptGuard = "Did our prompt break?" (testing & gating)
```

**Not just evaluation — it's CI/CD.**

---

## Installation

```bash
pip install promptguard
```

---

## Quickstart

```bash
# 1. Create an example bundle
promptguard init bundles/my-test

# 2. Validate configuration
promptguard validate bundles/my-test/bundle.yaml

# 3. Run tests (mock provider - no API key needed)
promptguard run bundles/my-test/bundle.yaml --provider mock

# 4. Run with OpenAI
export OPENAI_API_KEY=sk-xxx
promptguard run bundles/my-test/bundle.yaml
```

**If thresholds fail → exit code 1 → CI fails.**

---

## Core Concept: Behavior Bundles

A **Behavior Bundle** is a version-controlled test suite for prompts:

```
my-bundle/
├── bundle.yaml    # Config: prompt, dataset, thresholds
├── prompt.md     # Prompt template with {{ variables }}
├── dataset.jsonl # Test cases (JSONL)
└── schema.json   # Output validation (optional)
```

### Example bundle.yaml

```yaml
name: customer-support
version: "1.0"

prompt_path: prompt.md
dataset_path: dataset.jsonl

output_contract:
  schema_path: schema.json
  invariants:
    - "len(raw_output) < 500"
    - "'error' not in raw_output.lower()"

thresholds:
  - metric: pass_rate
    operator: ">="
    value: 0.90  # 90% must pass

provider:
  name: openai
  model: gpt-4o-mini
  temperature: 0.0
```

### Agent Bundles (v0.2+)

Test AI agent tool-calling behavior:

```yaml
name: coding-agent
version: "1.0"

agent:
  type: tool-calling
  model: gpt-4o
  max_steps: 10

tools:
  - type: function
    name: read_file
    description: "Read a file"
  - type: function
    name: write_file
    description: "Write to a file"

tasks_path: tasks.jsonl
```

---

## Key Features

### Threshold Gating
```yaml
thresholds:
  - metric: pass_rate
    operator: ">="
    value: 0.90    # Must pass 90% of tests
  - metric: avg_latency_ms
    operator: "<"
    value: 2000    # Must respond under 2s
```

### Baseline & Diff (v0.2+)
```bash
# Save current run as baseline
promptguard promote bundles/my-feature/

# Compare against baseline
promptguard diff bundles/my-feature/
# Shows: new failures, fixed failures, metric changes
```

### Output Validation
- **JSON Schema** - Enforce output structure
- **Invariants** - Python expressions (e.g., `len < 500`, no profanity)
- **Regex patterns** - Match specific formats

---

## CI Integration

### GitHub Actions

```yaml
name: Behavior Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install PromptGuard
        run: pip install promptguard

      - name: Run tests
        run: promptguard run bundles/my-feature/bundle.yaml
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      # v0.2+: Block on regressions
      - name: Diff check
        run: promptguard diff bundles/my-feature/ --ci-gate
```

See [docs/ci-integration.md](docs/ci-integration.md) for GitLab, CircleCI, and Azure DevOps.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `promptguard init [path]` | Scaffold new bundle |
| `promptguard validate <bundle>` | Validate configuration |
| `promptguard run <bundle>` | Execute tests |
| `promptguard promote <bundle>` | Save as baseline (v0.2+) |
| `promptguard diff <bundle>` | Compare vs baseline (v0.2+) |
| `promptguard agent init` | Scaffold agent bundle (v0.2+) |
| `promptguard agent run` | Run agent tests (v0.2+) |

### Options

```bash
promptguard run bundle.yaml --provider openai   # Override provider
promptguard run bundle.yaml --format json       # JSON output
promptguard run bundle.yaml --output report.md  # Write to file
promptguard run bundle.yaml --ci-gate          # Fail if thresholds fail
```

---

## Providers

| Provider | Env Variable | Models |
|----------|-------------|--------|
| OpenAI | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini, etc. |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-opus, claude-3-sonnet |
| Mock | (none) | Deterministic test responses |

---

## Documentation

| Doc | Description |
|-----|-------------|
| [Quickstart](docs/quickstart.md) | Get started in 5 minutes |
| [Behavior Bundles](docs/behavior-bundles.md) | Full specification |
| [CLI Reference](docs/cli.md) | All commands & options |
| [CI Integration](docs/ci-integration.md) | GitHub, GitLab, CircleCI |
| [Architecture](docs/architecture.md) | System design |
| [FAQ](docs/faq.md) | Common questions |
| [Roadmap](docs/roadmap.md) | What's coming |

---

## Development

```bash
# Clone
git clone https://github.com/jatanrathod13/promptguard
cd promptguard

# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Test
pytest tests/ -v

# Lint
ruff check promptguard/

# Type check
mypy promptguard/
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with 🔥 by <a href="https://github.com/jatanrathod13">@jatanrathod13</a></sub>
</p>
