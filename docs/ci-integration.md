# CI Integration

PromptGuard is designed to integrate seamlessly with any CI/CD platform. Non-zero exit codes fail the build when behavior regresses.

## Quick Start

```yaml
# Any CI platform
- run: pip install promptguard
- run: promptguard run bundles/my-feature/bundle.yaml
```

If thresholds fail → exit code 1 → build fails.

---

## Test Runner Note

If your CI environment has globally installed pytest plugins that interfere with runs, disable plugin autoloading for clean test runs:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

---

## GitHub Actions

### Basic Workflow

```yaml
# .github/workflows/behavior-tests.yaml
name: Behavior Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  behavior-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    
    - name: Install PromptGuard
      run: pip install promptguard
    
    - name: Validate bundles
      run: |
        for bundle in bundles/*/bundle.yaml; do
          promptguard validate "$bundle"
        done
    
    - name: Run behavior tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: promptguard run bundles/my-feature/bundle.yaml
```

### With Artifacts

Save reports as CI artifacts:

```yaml
    - name: Run behavior tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        promptguard run bundles/my-feature/bundle.yaml \
          --format json \
          --output behavior-report.json
      continue-on-error: true
    
    - name: Upload report
      uses: actions/upload-artifact@v4
      with:
        name: behavior-report
        path: behavior-report.json
```

### Multiple Bundles

```yaml
    - name: Run all bundles
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        failed=0
        for bundle in bundles/*/bundle.yaml; do
          echo "Running $bundle..."
          if ! promptguard run "$bundle"; then
            failed=1
          fi
        done
        exit $failed
```

### Mock Provider (No API Keys)

For testing the pipeline without real API calls:

```yaml
    - name: Run with mock provider
      run: promptguard run bundles/my-feature/bundle.yaml --provider mock
```

---

## GitLab CI

```yaml
# .gitlab-ci.yml
behavior-tests:
  image: python:3.11
  stage: test
  
  before_script:
    - pip install promptguard
  
  script:
    - promptguard validate bundles/my-feature/bundle.yaml
    - promptguard run bundles/my-feature/bundle.yaml
  
  variables:
    OPENAI_API_KEY: $OPENAI_API_KEY
  
  artifacts:
    when: always
    paths:
      - behavior-report.json
    reports:
      dotenv: behavior-report.json
```

---

## CircleCI

```yaml
# .circleci/config.yml
version: 2.1

jobs:
  behavior-tests:
    docker:
      - image: cimg/python:3.11
    
    steps:
      - checkout
      
      - run:
          name: Install PromptGuard
          command: pip install promptguard
      
      - run:
          name: Run behavior tests
          command: promptguard run bundles/my-feature/bundle.yaml
          environment:
            OPENAI_API_KEY: ${OPENAI_API_KEY}

workflows:
  test:
    jobs:
      - behavior-tests
```

---

## Azure DevOps

```yaml
# azure-pipelines.yml
trigger:
  - main

pool:
  vmImage: ubuntu-latest

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.11'

- script: pip install promptguard
  displayName: Install PromptGuard

- script: promptguard run bundles/my-feature/bundle.yaml
  displayName: Run behavior tests
  env:
    OPENAI_API_KEY: $(OPENAI_API_KEY)
```

---

## Secrets Management

### Setting API Keys

| Platform | How to set secrets |
|----------|-------------------|
| GitHub Actions | Repository Settings → Secrets and variables → Actions |
| GitLab CI | Settings → CI/CD → Variables |
| CircleCI | Project Settings → Environment Variables |
| Azure DevOps | Pipelines → Library → Variable groups |

### Best Practices

1. **Never commit API keys** — Always use CI secrets
2. **Use mock for PRs** — Real API calls only on main branch
3. **Limit key permissions** — Use read-only keys where possible
4. **Rotate regularly** — Monthly rotation recommended

---

## PR Gating Strategies

### Strict: Block All Failures

```yaml
- run: promptguard run bundles/my-feature/bundle.yaml
  # Default: fails if any threshold not met
```

### Lenient: Warn But Don't Block

```yaml
- run: promptguard run bundles/my-feature/bundle.yaml
  continue-on-error: true
```

### Conditional: Different Rules for PRs vs Main

```yaml
- run: |
    if [ "$GITHUB_EVENT_NAME" = "pull_request" ]; then
      # Use mock for PRs
      promptguard run bundles/my-feature/bundle.yaml --provider mock
    else
      # Real tests on main
      promptguard run bundles/my-feature/bundle.yaml
    fi
```

---

## Caching

Speed up CI by caching dependencies:

### GitHub Actions

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-promptguard
```

---

## Parallel Execution

Run multiple bundles in parallel:

### GitHub Actions Matrix

```yaml
jobs:
  behavior-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        bundle: [feature-a, feature-b, feature-c]
      fail-fast: false
    
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    - run: pip install promptguard
    - run: promptguard run bundles/${{ matrix.bundle }}/bundle.yaml
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Exit Codes

| Code | Meaning | CI Impact |
|------|---------|-----------|
| `0` | All thresholds passed | Build passes |
| `1` | Thresholds failed | Build fails |
| `2` | Error (invalid config, API error) | Build fails |
