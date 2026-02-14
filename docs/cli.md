# CLI Reference

BehaviorCI provides a command-line interface for managing and running Behavior Bundles.

## Installation

```bash
pip install behaviorci
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `behaviorci init` | Scaffold a new example bundle |
| `behaviorci validate` | Validate bundle configuration |
| `behaviorci run` | Execute tests and emit report |
| `behaviorci promote` | Run bundle and save as baseline |
| `behaviorci diff` | Compare run against baseline |
| `behaviorci agent init` | Scaffold a new agent bundle |
| `behaviorci agent validate` | Validate agent bundle |
| `behaviorci agent run` | Run agent bundle tests |
| `behaviorci --version` | Show version |
| `behaviorci --help` | Show help |

---

## behaviorci init

Scaffold a new Behavior Bundle with example files.

```bash
behaviorci init [PATH] [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `PATH` | `bundles/example` | Directory to create bundle in |

### Options

| Option | Description |
|--------|-------------|
| `-f, --force` | Overwrite existing files |

### Example

```bash
# Create example bundle
behaviorci init bundles/my-feature

# Overwrite existing
behaviorci init bundles/my-feature --force
```

### Created Files

```
bundles/my-feature/
├── bundle.yaml       # Bundle configuration
├── prompt.md         # Prompt template with {{ variables }}
├── dataset.jsonl     # Example test cases
└── schema.json       # JSON schema for output validation
```

---

## behaviorci validate

Validate a bundle configuration without running tests.

```bash
behaviorci validate BUNDLE_PATH
```

### Arguments

| Argument | Description |
|----------|-------------|
| `BUNDLE_PATH` | Path to bundle.yaml file |

### Example

```bash
behaviorci validate bundles/my-feature/bundle.yaml
```

### Output

```
✓ Bundle is valid: my-feature

 Name        my-feature    
 Version     1.0           
 Provider    openai        
 Model       gpt-4o-mini   
 Dataset     dataset.jsonl 
 Cases       5             
 Thresholds  2
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Bundle is valid |
| `1` | Validation failed |

---

## behaviorci run

Execute a bundle's test cases and emit a report.

```bash
behaviorci run BUNDLE_PATH [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `BUNDLE_PATH` | Path to bundle.yaml file |

### Options

| Option | Description |
|--------|-------------|
| `-p, --provider NAME` | Override provider (openai, anthropic, mock) |
| `-m, --model NAME` | Override model |
| `-f, --format FORMAT` | Output format: console, json, markdown |
| `-o, --output PATH` | Write report to file |
| `-v, --verbose` | Show detailed output |
| `-q, --quiet` | Minimal output |

### Examples

```bash
# Basic run
behaviorci run bundles/my-feature/bundle.yaml

# With mock provider (no API calls)
behaviorci run bundles/my-feature/bundle.yaml --provider mock

# JSON output to file
behaviorci run bundles/my-feature/bundle.yaml --format json --output report.json

# Markdown for CI artifacts
behaviorci run bundles/my-feature/bundle.yaml --format markdown --output report.md

# Verbose with real provider
OPENAI_API_KEY=sk-xxx behaviorci run bundles/my-feature/bundle.yaml --verbose
```

### Console Output

```
Running bundle: my-feature (5 cases)

╭────────────────────────────────────────╮
│ my-feature — PASSED                    │
╰─────────────────────────── 1234ms ─────╯

┏━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric      ┃ Value ┃
┡━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Cases │     5 │
│ Passed      │     5 │
│ Failed      │     0 │
│ Pass Rate   │ 100%  │
└─────────────┴───────┘

Thresholds
┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Metric      ┃ Expected ┃ Actual ┃ Status ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ pass_rate   │ >= 0.90  │   1.00 │   ✓    │
└─────────────┴──────────┴────────┴────────┘

✓ All thresholds passed
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All thresholds passed |
| `1` | One or more thresholds failed |
| `2` | Error (invalid bundle, provider error, etc.) |

---

## Environment Variables

### Provider API Keys

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI (GPT-4, etc.) |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |

### Example

```bash
export OPENAI_API_KEY=sk-xxx
behaviorci run bundles/my-feature/bundle.yaml
```

---

## Output Formats

### Console (default)

Human-readable terminal output with colors and tables.

```bash
behaviorci run bundle.yaml --format console
```

### JSON

Machine-readable output for CI pipelines:

```bash
behaviorci run bundle.yaml --format json --output report.json
```

```json
{
  "bundle": "my-feature",
  "passed": true,
  "summary": {
    "total_cases": 5,
    "passed_cases": 5,
    "failed_cases": 0,
    "pass_rate": 1.0
  },
  "thresholds": {
    "passed": true,
    "results": [
      {
        "metric": "pass_rate",
        "passed": true,
        "actual": 1.0,
        "expected": 0.9,
        "operator": ">="
      }
    ]
  }
}
```

### Markdown

GitHub-friendly output for PR comments:

```bash
behaviorci run bundle.yaml --format markdown --output report.md
```

```markdown
# BehaviorCI Report: my-feature

**Status**: ✅ PASSED
**Duration**: 1234ms

## Summary
| Metric | Value |
|--------|-------|
| Total Cases | 5 |
| Passed | 5 |
| Failed | 0 |
| Pass Rate | 100.0% |
```

---

## behaviorci promote

Run a bundle and save results as a baseline for future comparisons.

```bash
behaviorci promote BUNDLE_PATH [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `BUNDLE_PATH` | Path to bundle.yaml |

### Options

| Option | Description |
|--------|-------------|
| `-q, --quiet` | Minimal output |
| `--ci-gate` | Exit non-zero if thresholds fail |

### Example

```bash
# Run and save as baseline
behaviorci promote bundles/my-feature/bundle.yaml

# Quiet mode for CI
behaviorci promote bundles/my-feature/bundle.yaml --quiet
```

---

## behaviorci diff

Compare current run against a baseline to detect regressions.

```bash
behaviorci diff BUNDLE_PATH [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `BUNDLE_PATH` | Path to bundle.yaml |

### Options

| Option | Description |
|--------|-------------|
| `-b, --baseline` | Baseline ID or "latest" (default: latest) |
| `-f, --format` | Output format: console, json, markdown |
| `-o, --output` | Write report to file |
| `-v, --verbose` | Show detailed output |
| `--ci-gate` | Exit non-zero if regressions found |

### Example

```bash
# Compare against latest baseline
behaviorci diff bundles/my-feature/bundle.yaml

# JSON output for CI
behaviorci diff bundles/my-feature/bundle.yaml --format json --output diff.json

# CI regression check
behaviorci diff bundles/my-feature/bundle.yaml --ci-gate
```

---

## Agent Bundle Commands

### behaviorci agent init

Scaffold a new Agent Bundle.

```bash
behaviorci agent init [PATH] [OPTIONS]
```

### behaviorci agent validate

Validate an Agent Bundle.

```bash
behaviorci agent validate BUNDLE_PATH
```

### behaviorci agent run

Run Agent Bundle tests.

```bash
behaviorci agent run BUNDLE_PATH [OPTIONS]
```
