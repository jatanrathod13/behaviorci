# FAQ

## General

**What is PromptGuard?**

PromptGuard is a CI/CD tool for LLM behavior testing. It validates that your prompts produce expected outputs and fails builds when behavior regresses.

**Is this prompt optimization?**

No. PromptGuard enforces constraints, not quality. It checks that outputs match schemas and pass invariants—it doesn't suggest improvements.

**Is this observability/monitoring?**

No. PromptGuard runs in CI before deployment. For production monitoring, use tools like LangSmith, Helicone, or custom logging.

**Can it run locally?**

Yes. Install with `pip install promptguard` and run `promptguard run bundles/my-feature/bundle.yaml`. Use `--provider mock` for testing without API keys.

---

## Design

**Why file-first?**

Because behavior must be:
- **Versioned** — Track changes in git
- **Reviewable** — PRs for prompt changes
- **Auditable** — History of what shipped
- **Portable** — Run anywhere

**Why YAML for configuration?**

YAML is human-readable, widely supported, and familiar to DevOps teams. It also supports comments for documentation.

**Why JSONL for datasets?**

JSONL (JSON Lines) enables:
- Streaming without loading entire file
- Easy appending of new cases
- Line-by-line diffs
- Standard format (no custom parser)

---

## Execution

**How are prompts rendered?**

PromptGuard uses [Jinja2](https://jinja.palletsprojects.com/) templating. Variables from the dataset's `input` field are substituted into `{{ variable }}` placeholders.

**What makes a test case pass?**

A case passes if:
1. Provider returns without error
2. Output matches JSON schema (if configured)
3. All invariants evaluate to `True`

**What makes a bundle pass?**

All configured thresholds must be met. For example:
```yaml
thresholds:
  - metric: pass_rate
    operator: ">="
    value: 0.9
```
This requires 90% of cases to pass.

**Are tests deterministic?**

With `temperature: 0.0` (the default), most LLMs produce near-deterministic outputs. However, some variation is possible. The mock provider is fully deterministic.

---

## Providers

**Which providers are supported?**

| Provider | Environment Variable | Models |
|----------|---------------------|--------|
| OpenAI | `OPENAI_API_KEY` | gpt-4, gpt-4o-mini, gpt-3.5-turbo |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-opus, claude-3-sonnet, claude-3-haiku |
| Mock | (none required) | Deterministic test responses |

**Can I add custom providers?**

Yes. Implement the `LLMProvider` abstract class and register with `@register_provider("my-provider")`.

**How do I avoid API costs in CI?**

Use `--provider mock` for validation and basic testing:
```bash
promptguard run bundle.yaml --provider mock
```
Reserve real providers for critical paths or scheduled runs.

---

## Contracts

**What's the difference between schema and invariants?**

- **Schema**: Validates structure (JSON Schema draft-07)
- **Invariants**: Validates content (Python expressions)

Use schema for "is it valid JSON with these fields?" and invariants for "does the content make sense?"

**Can invariants access expected_output?**

Not currently. Invariants only have access to:
- `output` — Parsed JSON (or raw string)
- `raw_output` — Raw response string

Comparison with expected outputs is planned for v0.2.

**What happens if output isn't valid JSON?**

If a schema is configured and output isn't JSON, schema validation fails. Invariants can still access `raw_output` as a string.

---

## CI Integration

**How do I fail PRs on behavior regression?**

PromptGuard returns non-zero exit codes when thresholds fail. CI platforms automatically fail the build/PR.

**Should I use real providers in CI?**

Depends on your needs:
- **Mock**: Fast, free, deterministic—use for config validation
- **Real**: Catches actual regressions—use for critical paths

Consider mock for PRs and real providers for main branch.

**How do I store API keys in CI?**

Use your platform's secrets management:
- GitHub: Settings → Secrets → Actions
- GitLab: Settings → CI/CD → Variables
- CircleCI: Project Settings → Environment Variables

Never commit API keys to git.

---

## Troubleshooting

**Bundle validation fails with "file not found"**

Check that paths in `bundle.yaml` are relative to the bundle directory:
```yaml
prompt_path: prompt.md      # ✓ Correct
prompt_path: ./prompt.md    # ✓ Also works
prompt_path: /abs/path.md   # ✗ Don't use absolute paths
```

**Schema validation fails unexpectedly**

1. Ensure LLM output is valid JSON (no markdown, no extra text)
2. Check schema allows the output structure
3. Use `--verbose` to see actual outputs

**All tests pass locally but fail in CI**

- Check API key is correctly set in CI secrets
- Verify Python version matches
- Consider rate limiting on shared CI runners

**Tests are slow**

- Use `parallel: true` in bundle.yaml for concurrent execution
- Consider smaller datasets for PR checks
- Use mock provider for non-critical tests

---

## Roadmap

**What's planned for v0.2?**

- PR comments with diff reports
- Baseline storage and comparison
- `promptguard diff` command

**What's planned for v0.3?**

- Policy packs (reusable rule sets)
- Failure taxonomies (categorize failures)
- `promptguard add-failure` command

**Can I contribute?**

Yes! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
