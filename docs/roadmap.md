# Roadmap

## Current Version: v0.2.0 ✅

### Core Features (Complete)
- [x] Behavior Bundle specification
- [x] Bundle loader with Pydantic validation
- [x] Dataset parsing (JSONL)
- [x] Prompt templating (Jinja2)
- [x] Provider adapters (OpenAI, Anthropic, Mock)
- [x] Runner engine with async execution
- [x] Output contract evaluation (JSON Schema + invariants)
- [x] Threshold gating
- [x] CLI: `init`, `validate`, `run`
- [x] Reporters: Console, JSON, Markdown
- [x] Example bundle
- [x] Comprehensive test suite

### Agent Bundle (v0.2 - Complete)
- [x] Agent Bundle specification (`agent.yaml`)
- [x] Tool calling agent support
- [x] Agent task definition (JSONL)
- [x] Agent evaluation rules
- [x] CLI: `agent init`, `agent validate`, `agent run`
- [x] System prompt management
- [x] Memory/execution configuration

### Diff & Baseline (v0.2 - Complete)
- [x] Baseline storage (local JSON files)
- [x] `behaviorci promote` — Save current run as baseline
- [x] `behaviorci diff` — Compare run against baseline
- [x] Diff report showing:
  - New failures
  - Fixed failures
  - Metric changes
- [x] PR comments integration (GitHub Actions)
- [x] Markdown diff reports for PRs
- [x] `--ci-gate` for CI/CD regression blocking

### Use Cases (v0.2)
- Detect which cases regressed after a prompt change
- Accept intentional changes by promoting new baseline
- Block PRs that introduce new failures
- Test AI agent tool-calling behavior

---

## v0.3: Policy Packs

**Theme**: Reusable, shareable rule sets for common patterns.

### Features
- [ ] Policy pack specification (YAML)
- [ ] Built-in packs:
  - `safety` — No harmful content indicators
  - `json-only` — Output must be valid JSON
  - `length-limits` — Response length bounds
- [ ] `behaviorci add-failure` — Add failed case to regression suite
- [ ] Failure taxonomies (categorize failures)
- [ ] Custom policy pack authoring

### Use Cases
- Apply standard rules across all bundles
- Share policies across teams
- Automatically grow regression test suite

---

## v0.4: Enhanced Providers

**Theme**: Better provider support and flexibility.

### Features
- [ ] Streaming response support
- [ ] Function calling / tool use testing
- [ ] Local model support (Ollama, llama.cpp)
- [ ] Azure OpenAI adapter
- [ ] Google Vertex AI adapter
- [ ] Custom provider plugin system

---

## v0.5: Advanced Evaluation

**Theme**: Smarter ways to evaluate LLM outputs.

### Features
- [ ] Semantic similarity scoring
- [ ] LLM-as-judge evaluation
- [ ] Expected output comparison
- [ ] Regex pattern matching
- [ ] Custom evaluator plugins

---

## Future Considerations

### Performance
- Parallel bundle execution
- Result caching
- Incremental runs (only changed bundles)

### Integrations
- GitHub App for native PR integration
- Slack/Discord notifications
- Dashboard UI (optional hosted service)

### Enterprise
- SSO / authentication
- Audit logging
- Role-based access control
- Multi-tenant support

---

## Contributing

Want to help shape the roadmap? 

1. Open an issue for feature requests
2. Comment on existing issues to show interest
3. Submit PRs for implementation

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## Release Schedule

| Version | Target | Status |
|---------|--------|--------|
| v0.1.0 | Q1 2026 | ✅ Complete |
| v0.2.0 | Q1 2026 | ✅ Complete |
| v0.3.0 | Q2 2026 | 🔜 Planned |
| v0.4.0 | Q3 2026 | 📋 Planned |
