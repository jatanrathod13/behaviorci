# Quickstart

Get started with PromptGuard in 5 minutes.

## Installation

```bash
pip install promptguard
```

Verify installation:
```bash
promptguard --version
# promptguard, version 0.1.0
```

## Create Your First Bundle

Scaffold an example bundle:

```bash
promptguard init bundles/my-first-test
```

This creates:
```
bundles/my-first-test/
├── bundle.yaml       # Configuration
├── prompt.md         # Prompt template
├── dataset.jsonl     # Test cases
└── schema.json       # Output schema
```

## Validate the Bundle

Check that your configuration is valid:

```bash
promptguard validate bundles/my-first-test/bundle.yaml
```

Expected output:
```
✓ Bundle is valid: example-bundle

 Name        example-bundle 
 Version     1.0            
 Provider    mock           
 Model       (default)      
 Dataset     dataset.jsonl  
 Cases       3              
 Thresholds  1
```

## Run with Mock Provider

Test without API calls:

```bash
promptguard run bundles/my-first-test/bundle.yaml
```

The mock provider returns deterministic responses—great for testing your setup.

## Run with Real LLM

Set your API key and run:

```bash
# OpenAI
export OPENAI_API_KEY=sk-your-key-here
promptguard run bundles/my-first-test/bundle.yaml --provider openai

# Anthropic
export ANTHROPIC_API_KEY=your-key-here
promptguard run bundles/my-first-test/bundle.yaml --provider anthropic
```

## Customize Your Bundle

### Edit the Prompt

Open `bundles/my-first-test/prompt.md`:

```markdown
You are a helpful assistant.

User question: {{ question }}

Respond with a JSON object:
{
  "answer": "your answer",
  "confidence": 0.95
}
```

Variables like `{{ question }}` come from your dataset.

### Add Test Cases

Edit `bundles/my-first-test/dataset.jsonl`:

```jsonl
{"input": {"question": "What is 2+2?"}}
{"input": {"question": "What is the capital of France?"}}
{"input": {"question": "Who wrote Hamlet?"}}
```

Each line is a test case. The `input` object provides template variables.

### Configure Thresholds

Edit `bundles/my-first-test/bundle.yaml`:

```yaml
name: my-first-test
version: "1.0"

prompt_path: prompt.md
dataset_path: dataset.jsonl

output_contract:
  schema_path: schema.json
  invariants:
    - "len(raw_output) < 500"

thresholds:
  - metric: pass_rate
    operator: ">="
    value: 0.9  # 90% of tests must pass

provider:
  name: openai
  model: gpt-4o-mini
  temperature: 0.0
```

## Add to CI

Create `.github/workflows/behavior-tests.yaml`:

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
    - run: pip install promptguard
    - run: promptguard run bundles/my-first-test/bundle.yaml
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Now PRs will fail if your LLM behavior regresses!

## Next Steps

- 📖 [Behavior Bundles](behavior-bundles.md) — Full bundle specification
- 🖥️ [CLI Reference](cli.md) — All commands and options
- 🔧 [CI Integration](ci-integration.md) — GitHub, GitLab, CircleCI guides
- 🏗️ [Architecture](architecture.md) — How PromptGuard works
- ❓ [FAQ](faq.md) — Common questions

## Example: Real-World Bundle

Here's a more complete example:

```yaml
# bundles/customer-support/bundle.yaml
name: customer-support-responses
version: "2.1"
description: Tests customer support bot response quality

prompt_path: support-prompt.md
dataset_path: support-cases.jsonl

output_contract:
  schema_path: response-schema.json
  invariants:
    - "len(raw_output) < 2000"
    - "'apolog' in raw_output.lower() or 'sorry' in raw_output.lower()"
    - "output.get('tone') in ['helpful', 'empathetic', 'professional']"

thresholds:
  - metric: pass_rate
    operator: ">="
    value: 0.95
  - metric: schema_valid_rate
    operator: ">="
    value: 0.99

provider:
  name: openai
  model: gpt-4o
  temperature: 0.0
  max_tokens: 512

timeout: 30.0
retries: 2
```
