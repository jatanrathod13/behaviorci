# Contributing to PromptGuard

PromptGuard is **behavior release engineering**.

We optimize for:
- Correctness over features
- CI leverage over UX
- Small, reviewable PRs

---

## Dev setup

```bash
git clone https://github.com/<org>/promptguard
cd promptguard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## What makes a good PR

Every PR must include:

* CLI surface (or explicit rationale)
* At least one test
* Docs update
* Example bundle or fixture

---

## In scope

* Behavior bundles
* Evaluation runners
* Reporters
* CI integration

## Out of scope

* Prompt optimization
* Observability
* Agent orchestration
