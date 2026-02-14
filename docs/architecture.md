# Architecture

PromptGuard is intentionally simple—a file-first, CI-friendly tool for validating LLM behavior.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLI Layer                               │
│  promptguard init | validate | run                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Bundle Loader                              │
│  • Parse bundle.yaml (Pydantic validation)                       │
│  • Load prompt template (Jinja2)                                 │
│  • Parse dataset (JSONL)                                         │
│  • Load JSON schema (optional)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Runner Engine                            │
│  • Execute test cases (sequential or parallel)                   │
│  • Call LLM provider for each case                               │
│  • Evaluate output contracts                                     │
│  • Compute metrics and apply thresholds                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  OpenAI Provider │ │ Anthropic Provider│ │   Mock Provider  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Reporters                               │
│  Console (Rich) | JSON | Markdown                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Exit Code (0 = pass, 1 = fail)
```

## Execution Flow

1. **Load Behavior Bundle** — Parse `bundle.yaml` and validate with Pydantic
2. **Resolve References** — Load prompt template, dataset, and optional schema
3. **Initialize Provider** — Create LLM provider instance with configuration
4. **Execute Cases** — For each dataset row:
   - Render prompt with input variables
   - Call LLM provider
   - Capture response and latency
5. **Evaluate Contracts** — For each response:
   - Validate against JSON schema (if configured)
   - Check invariant rules
6. **Compute Metrics** — Calculate pass rates, latencies, etc.
7. **Apply Thresholds** — Compare metrics against configured thresholds
8. **Emit Report** — Generate output in requested format
9. **Exit** — Return code 0 (pass) or 1 (fail)

## Core Components

### Bundle Module (`promptguard/bundle/`)

| File | Purpose |
|------|---------|
| `models.py` | Pydantic models defining bundle specification |
| `loader.py` | YAML parsing, validation, file reference resolution |
| `dataset.py` | JSONL parsing with iteration support |

### Provider Module (`promptguard/providers/`)

| File | Purpose |
|------|---------|
| `base.py` | Abstract `LLMProvider` base class |
| `registry.py` | Dynamic provider registration and lookup |
| `openai.py` | OpenAI API adapter (GPT-4, etc.) |
| `anthropic.py` | Anthropic API adapter (Claude) |
| `mock.py` | Deterministic mock for testing |

### Runner Module (`promptguard/runner/`)

| File | Purpose |
|------|---------|
| `engine.py` | Main execution orchestration |
| `evaluator.py` | JSON schema and invariant validation |
| `thresholds.py` | Metric computation and threshold checking |

### Reporter Module (`promptguard/reporters/`)

| File | Purpose |
|------|---------|
| `base.py` | Abstract `Reporter` base class |
| `console.py` | Rich terminal output with colors and tables |
| `json_reporter.py` | Machine-readable JSON output |
| `markdown.py` | GitHub-friendly markdown with tables |

## Design Principles

### File-First

Everything is defined in files:
- Bundle configuration in YAML
- Prompts in Markdown
- Test cases in JSONL
- Schemas in JSON

This enables:
- Version control with git
- Code review in PRs
- Easy diffing between versions

### CI-Friendly

- Non-zero exit codes on failure
- Machine-readable JSON output
- No interactive prompts
- Environment variable configuration

### Deterministic Where Possible

- Temperature defaults to 0.0
- Consistent case ordering
- Reproducible with mock provider

## Data Flow

```python
# Simplified execution flow
bundle = load_bundle("bundles/example/bundle.yaml")

provider = get_provider(bundle.config.provider.name)
evaluator = Evaluator(bundle.config.output_contract)
runner = Runner(bundle, provider)

result = await runner.run()

reporter = get_reporter("console")
print(reporter.emit(result))

sys.exit(0 if result.passed else 1)
```

## Error Handling

Custom exception hierarchy:

```
PromptGuardError
├── BundleError
│   ├── BundleNotFoundError
│   └── BundleValidationError
├── DatasetError
├── ProviderError
│   ├── ProviderConfigError
│   └── ProviderAPIError
├── ContractError
│   ├── SchemaValidationError
│   └── InvariantError
└── ThresholdError
```
