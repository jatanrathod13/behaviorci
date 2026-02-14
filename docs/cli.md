# CLI Reference

PromptGuard provides a command-line interface for managing and running Behavior Bundles.

## Installation

```bash
pip install promptguard
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `promptguard init` | Scaffold a new example bundle |
| `promptguard validate` | Validate bundle configuration |
| `promptguard run` | Execute tests and emit report |
| `promptguard promote` | Run bundle and save as baseline |
| `promptguard diff` | Compare run against baseline |
| `promptguard agent init` | Scaffold a new agent bundle |
| `promptguard agent validate` | Validate agent bundle |
| `promptguard agent run` | Run agent bundle tests |
| `promptguard --version` | Show version |
| `promptguard --help` | Show help |

---

## promptguard init

Scaffold a new Behavior Bundle with example files.

```bash
promptguard init [PATH] [OPTIONS]
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
promptguard init bundles/my-feature

# Overwrite existing
promptguard init bundles/my-feature --force
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

## promptguard validate

Validate a bundle configuration without running tests.

```bash
promptguard validate BUNDLE_PATH
```

### Arguments

| Argument | Description |
|----------|-------------|
| `BUNDLE_PATH` | Path to bundle.yaml file |

### Example

```bash
promptguard validate bundles/my-feature/bundle.yaml
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

## promptguard run

Execute a bundle's test cases and emit a report.

```bash
promptguard run BUNDLE_PATH [OPTIONS]
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
promptguard run bundles/my-feature/bundle.yaml

# With mock provider (no API calls)
promptguard run bundles/my-feature/bundle.yaml --provider mock

# JSON output to file
promptguard run bundles/my-feature/bundle.yaml --format json --output report.json

# Markdown for CI artifacts
promptguard run bundles/my-feature/bundle.yaml --format markdown --output report.md

# Verbose with real provider
OPENAI_API_KEY=sk-xxx promptguard run bundles/my-feature/bundle.yaml --verbose
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
promptguard run bundles/my-feature/bundle.yaml
```

---

## Output Formats

### Console (default)

Human-readable terminal output with colors and tables.

```bash
promptguard run bundle.yaml --format console
```

### JSON

Machine-readable output for CI pipelines:

```bash
promptguard run bundle.yaml --format json --output report.json
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
promptguard run bundle.yaml --format markdown --output report.md
```

```markdown
# PromptGuard Report: my-feature

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

## promptguard promote

Run a bundle and save results as a baseline for future comparisons.

```bash
promptguard promote BUNDLE_PATH [OPTIONS]
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
promptguard promote bundles/my-feature/bundle.yaml

# Quiet mode for CI
promptguard promote bundles/my-feature/bundle.yaml --quiet
```

---

## promptguard diff

Compare current run against a baseline to detect regressions.

```bash
promptguard diff BUNDLE_PATH [OPTIONS]
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
promptguard diff bundles/my-feature/bundle.yaml

# JSON output for CI
promptguard diff bundles/my-feature/bundle.yaml --format json --output diff.json

# CI regression check
promptguard diff bundles/my-feature/bundle.yaml --ci-gate
```

---

## Agent Bundle Commands

### promptguard agent init

Scaffold a new Agent Bundle.

```bash
promptguard agent init [PATH] [OPTIONS]
```

### promptguard agent validate

Validate an Agent Bundle.

```bash
promptguard agent validate BUNDLE_PATH
```

### promptguard agent run

Run Agent Bundle tests.

```bash
promptguard agent run BUNDLE_PATH [OPTIONS]
```
