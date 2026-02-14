# PromptGuard: The Agent Quality Platform

## From "Jest for Prompts" to "The CI/CD Infrastructure for AI Agents"

### Executive Summary

After analyzing the entire PromptGuard codebase, the current ecosystem (PromptFoo, Langfuse, LangSmith, DeepEval), and every major agentic framework (LangChain, CrewAI, OpenAI Agents SDK, Anthropic Claude Agent SDK, Google ADK), one thing is clear:

**No open-source tool bridges the gap between observability and testing for AI agents.**

- **Langfuse/LangSmith** watch agents in production but can't block bad deployments
- **PromptFoo** tests prompts in CI but treats agents as black boxes (input → output)
- **DeepEval** has metrics but no agent trajectory analysis
- **Each framework** (CrewAI, OpenAI SDK, Google ADK) has its own evaluation, but nothing works across them

PromptGuard should become the **middleware layer** that connects these worlds — consuming traces from observability tools, delegating security testing to PromptFoo, providing the CI/CD quality gates that neither Langfuse nor PromptFoo delivers well, and working with **any** agentic framework through a universal adapter system.

---

## The Architecture: Three Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LAYER 3: INTEGRATIONS                          │
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │LangChain │ │ CrewAI   │ │ OpenAI   │ │ Claude   │ │ Google   │ │
│  │/LangGraph│ │          │ │Agents SDK│ │Agent SDK │ │   ADK    │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│       │             │            │             │            │       │
│       └─────────────┴──────┬─────┴─────────────┴────────────┘       │
│                            │                                        │
│              ┌─────────────▼──────────────┐                        │
│              │   Universal Agent Adapter   │ ← OpenTelemetry-native│
│              │   (Framework Connectors)    │                        │
│              └─────────────┬──────────────┘                        │
├────────────────────────────┼────────────────────────────────────────┤
│                     LAYER 2: PLATFORM CORE                         │
│                            │                                        │
│  ┌────────────┐  ┌────────▼────────┐  ┌────────────────┐          │
│  │  Trace     │  │  Agent Test     │  │  Evaluation     │          │
│  │  Ingestion │  │  Engine         │  │  Engine          │          │
│  │            │  │                 │  │                  │          │
│  │ • Langfuse │  │ • Multi-trial   │  │ • Trajectory    │          │
│  │ • OTel     │  │ • Statistical   │  │ • LLM-as-Judge  │          │
│  │ • Custom   │  │ • Sandboxed     │  │ • Custom Plugin │          │
│  └─────┬──────┘  └────────┬────────┘  └───────┬────────┘          │
│        │                  │                    │                    │
│        └──────────────────┼────────────────────┘                    │
│                           │                                         │
│              ┌────────────▼──────────────┐                         │
│              │     Quality Gate Engine    │                         │
│              │  • Statistical thresholds  │                         │
│              │  • Regression detection    │                         │
│              │  • Cost budgets            │                         │
│              │  • Safety policies         │                         │
│              └────────────┬──────────────┘                         │
├────────────────────────────┼────────────────────────────────────────┤
│                     LAYER 1: CI/CD & OUTPUT                        │
│                            │                                        │
│  ┌──────────┐ ┌───────────▼──┐ ┌──────────┐ ┌─────────────────┐   │
│  │ GitHub   │ │  CLI /       │ │ Webhooks │ │  PromptFoo      │   │
│  │ Actions  │ │  pytest      │ │ Slack    │ │  (red teaming)  │   │
│  │ PR Gates │ │  Integration │ │ PagerDuty│ │  Delegation     │   │
│  └──────────┘ └──────────────┘ └──────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## LAYER 3: Universal Agent Adapters

### The Problem

Every framework defines agents differently:

| Framework | Agent | Tool | Execution | Trace Format |
|-----------|-------|------|-----------|--------------|
| **OpenAI Agents SDK** | `Agent(name, instructions, tools)` | `@function_tool` decorator | `Runner.run()` | OTel spans with `AgentSpanData` |
| **Claude Agent SDK** | `ClaudeAgent(allowed_tools)` | Built-in `Bash`, `Read`, etc. | Proactive agent loop | Custom traces |
| **LangChain/LangGraph** | `ChatModel` + `Tool` | `@tool` decorator | `RunnableSequence` / Graph | LangSmith / OTel callbacks |
| **CrewAI** | `Agent(role, goal, backstory)` | `@tool` decorator | `Crew.kickoff()` | Custom logging |
| **Google ADK** | `Agent(model, tools)` | `FunctionTool`, MCP | `Runner.run()` | OTel spans |

### The Solution: `promptguard.adapters`

A thin adapter layer that wraps each framework's agent into a standard `TestableAgent` interface:

```python
from promptguard.adapters import TestableAgent

class TestableAgent(Protocol):
    """Universal agent interface for testing."""

    async def run(self, input: str) -> AgentTrace:
        """Execute the agent and return a structured trace."""
        ...

    def get_tools(self) -> list[ToolDefinition]:
        """Return available tool definitions."""
        ...

    def get_config(self) -> AgentConfig:
        """Return agent configuration (model, temperature, etc.)."""
        ...
```

#### Adapter: OpenAI Agents SDK

```python
from promptguard.adapters.openai_agents import wrap_openai_agent
from agents import Agent, Runner

# User's existing OpenAI agent
agent = Agent(
    name="customer-support",
    instructions="You are a helpful support agent...",
    tools=[lookup_order, issue_refund],
)

# Wrap it for PromptGuard testing — one line
testable = wrap_openai_agent(agent)

# Now use it in a PromptGuard test bundle
result = await promptguard.test(
    agent=testable,
    dataset="tests/support-cases.jsonl",
    evaluators=[tool_trajectory, llm_judge("Was the response helpful?")],
    thresholds={"pass_rate": 0.9, "avg_cost": 0.50},
)
```

**How it works**: The adapter hooks into OpenAI's `add_trace_processor()` to capture `AgentSpanData`, `GenerationSpanData`, and `FunctionSpanData` as structured `AgentTrace` objects. Every tool call, handoff, and generation is captured with full arguments and results.

#### Adapter: Claude Agent SDK

```python
from promptguard.adapters.claude_agent import wrap_claude_agent

# User's existing Claude agent
agent = ClaudeAgent(
    allowed_tools=["Bash", "Read", "Write", "Glob", "Grep"],
    model="claude-sonnet-4-5-20250929",
)

testable = wrap_claude_agent(agent)
```

#### Adapter: LangChain / LangGraph

```python
from promptguard.adapters.langchain import wrap_langchain_agent

# User's existing LangGraph agent
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
# ... graph definition

testable = wrap_langchain_agent(graph.compile())
```

**How it works**: Hooks into LangChain's callback system (`BaseCallbackHandler`) to capture every chain invocation, tool call, and LLM generation as an `AgentTrace`.

#### Adapter: CrewAI

```python
from promptguard.adapters.crewai import wrap_crewai_crew

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
)

testable = wrap_crewai_crew(crew)
```

**How it works**: Wraps `Crew.kickoff()` and intercepts agent-to-agent delegation, tool calls, and task completion.

#### Adapter: Google ADK

```python
from promptguard.adapters.google_adk import wrap_adk_agent

agent = Agent(
    model="gemini-2.0-flash",
    tools=[search_tool, code_execution],
)

testable = wrap_adk_agent(agent)
```

#### Adapter: Raw OpenTelemetry (Any Framework)

```python
from promptguard.adapters.otel import wrap_otel_traces

# For any framework that emits OTel GenAI spans
testable = wrap_otel_traces(
    service_name="my-agent",
    endpoint="http://localhost:4318",  # OTel collector
)
```

This is the catch-all: any framework that emits OpenTelemetry GenAI semantic convention spans (agent creation, agent invocation, tool calls) can be tested.

---

### Scenario: "I use CrewAI today, might switch to OpenAI Agents SDK tomorrow"

A startup has a multi-agent research crew in CrewAI. They write PromptGuard test bundles:

```yaml
# bundles/research-crew/agent.yaml
name: research-crew
adapter: crewai
adapter_config:
  module: my_app.crews.research
  class: ResearchCrew

dataset_path: tasks.jsonl
evaluation:
  trajectory:
    required_tools: [web_search, summarize]
    forbidden_tools: [execute_code]
  output:
    type: llm-judge
    rubric: "Is the research report factually grounded with citations?"
  cost:
    max_per_task: 1.00

thresholds:
  - metric: task_success_rate
    operator: ">="
    value: 0.85
  - metric: trajectory_accuracy
    operator: ">="
    value: 0.80
```

Six months later, they migrate to OpenAI Agents SDK. They change ONE line:

```yaml
adapter: openai-agents
adapter_config:
  module: my_app.agents.research
  class: research_agent
```

**Same tests. Same thresholds. Same CI pipeline. Different framework.** This is the value proposition no other tool offers.

---

## LAYER 2: Platform Core

### 2A. Trace Ingestion — Bridge to Langfuse

**The key insight**: 89% of teams already have observability (Langfuse, LangSmith). Only 52% have evals. Don't make them choose — consume their existing traces.

```python
from promptguard.traces import LangfuseTraceSource, OTelTraceSource

# Pull production traces from Langfuse
source = LangfuseTraceSource(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com",
)

# Filter to specific traces
traces = source.query(
    tags=["customer-support"],
    min_score=None,  # Include failures
    date_range="last_7_days",
    limit=100,
)

# Turn production traces into regression test cases
dataset = promptguard.traces_to_dataset(
    traces=traces,
    include_tool_calls=True,
    include_expected_output=True,  # Use scored traces as ground truth
)

# Save as a PromptGuard dataset
dataset.save("bundles/support-agent/regression-cases.jsonl")
```

**Scenario: Production Trace → Regression Test → CI Gate**

1. Monday: A customer reports the agent gave wrong refund information
2. The team finds the trace in Langfuse, scores it as "failed"
3. They run: `promptguard capture --from-langfuse --tag "customer-support" --score-below 0.5`
4. PromptGuard pulls the failed traces, converts them into test cases with expected tool calls and outputs
5. These cases are added to the regression suite
6. Next PR: CI runs PromptGuard → new test cases catch the same class of failure → PR is blocked
7. Developer fixes the prompt, all cases pass → PR merges

**This closes the loop that neither Langfuse nor PromptFoo provides**: production failure → regression test → CI gate → prevention.

---

### 2B. Agent Test Engine — Statistical, Multi-Trial, Real Execution

#### The Non-Determinism Problem

PromptFoo's fundamental limitation: it runs each test case ONCE and does pass/fail. But agents are non-deterministic — the same input can produce different tool call sequences, different outputs, different costs. A single run tells you almost nothing about reliability.

**Anthropic's recommended approach** (from their eval guide): Run multiple trials per task, compute statistical metrics, and use confidence intervals.

PromptGuard implements this as a first-class concept:

```yaml
# bundles/coding-agent/agent.yaml
execution:
  trials_per_task: 5          # Run each task 5 times
  confidence_level: 0.95      # 95% confidence interval
  parallel_trials: true       # Run trials concurrently
  variance_threshold: 0.15    # Flag tasks with >15% variance

thresholds:
  - metric: task_success_rate
    operator: ">="
    value: 0.85
    statistical: true          # Must be statistically significant
  - metric: tool_trajectory_accuracy
    operator: ">="
    value: 0.80
  - metric: cost_per_task_p95
    operator: "<="
    value: 0.75
```

**Output**:
```
Bundle: coding-agent
Trials: 5 per task × 20 tasks = 100 runs

Task Success Rate: 88.0% ± 4.2% (95% CI: [83.8%, 92.2%]) ✓ ≥ 85%
Trajectory Accuracy: 82.5% ± 3.1% (95% CI: [79.4%, 85.6%]) ✓ ≥ 80%
Cost per Task (P95): $0.62 ✓ ≤ $0.75

⚠ High-Variance Tasks (variance > 15%):
  task-007: 60% success (3/5) — tool choice varies
  task-013: 40% success (2/5) — output format unstable

✓ All thresholds passed with statistical significance
```

No other open-source tool does multi-trial statistical agent testing.

---

### 2C. Evaluation Engine — Trajectory + LLM-as-Judge + Plugins

#### Tool Call Trajectory Evaluation

The biggest gap in PromptFoo: it can't see intermediate steps. PromptGuard's trajectory evaluator compares the actual tool call sequence against expected patterns:

```yaml
evaluation:
  trajectory:
    # Exact sequence matching
    expected_sequence:
      - tool: search_docs
        args_contain: {query: "authentication"}
      - tool: read_file
        args_contain: {path: "auth.py"}
      - tool: write_file  # Must write AFTER reading

    # Flexible matching
    required_tools: [search_docs, read_file]
    forbidden_tools: [delete_file, execute_shell]
    order_matters: true

    # Argument validation
    arg_validators:
      search_docs:
        query: "len(value) > 5"  # Query must be meaningful
      write_file:
        path: "not value.startswith('/etc')"  # No system files
```

**Scenario**: A coding agent is asked to "add input validation to the login function." The trajectory evaluator verifies:
- Step 1: Agent searched for the login function (not random files)
- Step 2: Agent read the correct file
- Step 3: Agent wrote changes to the correct file (not a different one)
- Step 4: Agent did NOT execute shell commands or delete files

Two agents might both produce correct final code, but one took 3 focused steps and the other took 15 random steps. Trajectory testing catches the inefficient agent.

#### LLM-as-Judge with Rubrics

```yaml
evaluation:
  judges:
    - name: quality-judge
      model: claude-sonnet-4-5-20250929  # Or any provider
      rubric: |
        Evaluate the agent's response on these criteria:
        1. Correctness: Does the code actually implement the requested feature?
        2. Style: Does it follow the project's existing patterns?
        3. Safety: Are there any security vulnerabilities?
        4. Completeness: Are edge cases handled?
      scoring: numeric  # 1-5 scale
      threshold: 3.5

    - name: factuality-judge
      model: gpt-4o
      rubric: "Are all claims in the response factually verifiable?"
      scoring: boolean
```

#### Custom Evaluator Plugin System

```python
from promptguard.evaluators import register_evaluator, EvaluatorResult

@register_evaluator("hipaa-compliance")
class HIPAAEvaluator:
    """Check agent outputs for PHI/HIPAA compliance."""

    def evaluate(self, trace: AgentTrace, context: dict) -> EvaluatorResult:
        # Check for PHI patterns (SSN, DOB, medical records)
        phi_patterns = [r'\d{3}-\d{2}-\d{4}', r'DOB:', ...]
        violations = []
        for pattern in phi_patterns:
            if re.search(pattern, trace.final_output):
                violations.append(f"Potential PHI: {pattern}")

        return EvaluatorResult(
            passed=len(violations) == 0,
            score=1.0 if not violations else 0.0,
            details={"violations": violations},
        )

# Use in bundle YAML:
# evaluation:
#   custom:
#     - type: hipaa-compliance
```

---

### 2D. Quality Gate Engine

The orchestrator that combines all evaluation results into a CI-friendly pass/fail decision:

```python
# Programmatic API (pytest integration)
import promptguard

def test_support_agent_quality():
    result = promptguard.run(
        bundle="bundles/support-agent/",
        adapter="openai-agents",
    )

    assert result.passed, f"Quality gate failed: {result.failure_summary}"
    assert result.metrics["task_success_rate"] >= 0.90
    assert result.metrics["avg_cost"] <= 0.50
    assert result.metrics["trajectory_accuracy"] >= 0.85
    assert result.regressions == 0, f"Regressions detected: {result.regression_details}"
```

---

## LAYER 1: CI/CD & Output

### PromptFoo Delegation (Don't Reinvent Red Teaming)

PromptFoo is best-in-class at red teaming (50+ vulnerability types, OWASP/MITRE compliance, multi-turn attacks). Don't compete — delegate:

```yaml
# bundles/support-agent/agent.yaml
security:
  delegate_to: promptfoo
  promptfoo_config:
    redteam:
      strategies:
        - jailbreak
        - jailbreak:composite
      plugins:
        - harmful:hate
        - pii:direct
        - tool-discovery
      numTests: 20

  # PromptGuard wraps the agent as a PromptFoo custom provider
  # Runs red team tests, captures results, integrates into quality gate
```

**How it works**:
1. PromptGuard generates a temporary PromptFoo config file
2. Wraps the agent as a PromptFoo HTTP provider (lightweight server)
3. Runs `promptfoo redteam run` as a subprocess
4. Parses PromptFoo's JSON output
5. Integrates red team results into the quality gate (e.g., `security_pass_rate >= 0.95`)

This gives users PromptFoo's world-class security testing without leaving the PromptGuard workflow.

### Langfuse Trace Export

After each test run, PromptGuard can export traces TO Langfuse for visualization:

```yaml
integrations:
  langfuse:
    enabled: true
    public_key: ${LANGFUSE_PUBLIC_KEY}
    secret_key: ${LANGFUSE_SECRET_KEY}
    tags: ["ci-test", "pr-${PR_NUMBER}"]
```

Now teams can see their CI test traces in the same Langfuse dashboard as production traces — unified observability.

### GitHub Actions Quality Gate

```yaml
# .github/workflows/agent-quality.yaml
name: Agent Quality Gate
on: [pull_request]

jobs:
  agent-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run PromptGuard
        uses: promptguard/action@v1
        with:
          bundles: bundles/
          adapter: openai-agents
          trials: 3
          parallel: true

      - name: Post Results to PR
        if: always()
        uses: promptguard/action@v1
        with:
          command: report
          format: github-pr-comment
```

---

## Real-World Scenarios: End-to-End

### Scenario 1: Fintech Trading Research Agent (OpenAI Agents SDK)

**Team**: 5 engineers building a trading research agent with OpenAI Agents SDK.
**Stack**: OpenAI Agents SDK + custom tools + Langfuse for production monitoring.

```
Day 1: Setup
├── pip install promptguard
├── promptguard init --adapter openai-agents bundles/research-agent/
├── Write 20 test tasks in tasks.jsonl
└── Configure thresholds: 90% success, $2 max cost, no shell execution

Day 2-30: Development Loop
├── Engineer edits agent instructions or tools
├── Runs: promptguard run bundles/research-agent/
├── Sees: 17/20 passed, 3 failed (trajectory shows agent calling wrong tools)
├── Fixes agent instructions
├── Runs again: 20/20 passed
└── Commits and pushes

PR Review:
├── GitHub Action runs PromptGuard with 3 trials per task
├── Statistical result: 91.7% ± 3.2% success rate → passes ≥90% threshold
├── Cost: $1.43 avg per task → passes ≤$2 threshold
├── PromptFoo red team: 0 vulnerabilities found
├── PR comment posted with full results
└── Merge approved

Production Monitoring:
├── Agent runs in production, traces sent to Langfuse
├── Weekly: promptguard capture --from-langfuse --score-below 0.5
├── Failed production traces become regression test cases
└── Regression suite grows organically from real failures
```

### Scenario 2: Healthcare Multi-Agent System (CrewAI)

**Team**: Hospital IT building a patient intake system with 3 agents.
**Stack**: CrewAI (Triage Agent, Records Agent, Scheduling Agent) + HIPAA compliance requirements.

```yaml
# bundles/patient-intake/agent.yaml
name: patient-intake-crew
adapter: crewai
adapter_config:
  module: intake_system.crews
  class: PatientIntakeCrew

evaluation:
  trajectory:
    # Triage agent must run before Records agent
    expected_sequence:
      - agent: triage_agent
        tool: assess_symptoms
      - agent: records_agent
        tool: lookup_patient
      - agent: scheduling_agent
        tool: find_available_slots
    order_matters: true

  judges:
    - name: medical-accuracy
      model: claude-sonnet-4-5-20250929
      rubric: "Is the triage assessment consistent with standard medical protocols?"
      threshold: 4.0

  custom:
    - type: hipaa-compliance
      config:
        strict: true
        check_tool_args: true  # PHI in tool arguments too

  safety:
    delegate_to: promptfoo
    promptfoo_config:
      plugins:
        - pii:direct
        - harmful:health
```

**What PromptGuard validates**:
1. Agent execution order (triage before records before scheduling)
2. Medical accuracy of triage (LLM-as-judge)
3. HIPAA compliance of ALL outputs AND tool arguments (custom evaluator)
4. No PII leakage under adversarial inputs (PromptFoo red team)
5. Cost per intake stays under $1.50
6. All results are statistically significant across 5 trials

### Scenario 3: Developer Tooling Agent (Claude Agent SDK)

**Team**: DevOps team building a Claude-powered infrastructure agent.
**Stack**: Claude Agent SDK with Bash, Read, Write tools.

```yaml
# bundles/infra-agent/agent.yaml
name: infra-agent
adapter: claude-agent
adapter_config:
  allowed_tools: [Bash, Read, Write, Glob, Grep]
  model: claude-sonnet-4-5-20250929
  permission_mode: plan

evaluation:
  trajectory:
    forbidden_tools_regex: "rm -rf|DROP TABLE|shutdown"
    required_tools: [Read]  # Must read before writing
    arg_validators:
      Bash:
        command: "'sudo' not in value"  # No sudo
      Write:
        file_path: "not value.startswith('/etc')"  # No system files

  output:
    type: llm-judge
    rubric: "Did the agent correctly diagnose and fix the infrastructure issue?"

execution:
  trials_per_task: 3
  sandbox: docker  # Run in isolated container
  sandbox_config:
    image: "ubuntu:22.04"
    mount: "./test-infra:/workspace"
    network: none  # No network access
```

### Scenario 4: RAG-Powered Customer Support (LangChain)

**Team**: E-commerce company with a LangChain RAG agent.
**Stack**: LangChain + Pinecone + GPT-4o + Langfuse monitoring.

```yaml
# bundles/support-rag/agent.yaml
name: support-rag-agent
adapter: langchain
adapter_config:
  module: support.chains
  class: SupportChain

evaluation:
  trajectory:
    required_tools: [vector_search]  # Must search before answering
    arg_validators:
      vector_search:
        query: "len(value) > 10"  # No trivial queries

  judges:
    - name: groundedness
      model: gpt-4o
      rubric: "Is every claim in the response supported by the retrieved documents?"
      scoring: numeric
      threshold: 4.0

    - name: helpfulness
      model: claude-sonnet-4-5-20250929
      rubric: "Would a customer find this response helpful and actionable?"
      scoring: numeric
      threshold: 3.5

  # Ingest traces from Langfuse for regression testing
  trace_source:
    type: langfuse
    filter:
      tags: ["production", "customer-support"]
      score_below: 0.6  # Pull poorly-scored production traces
    auto_refresh: weekly
```

---

## The Python-First SDK

Everything above is also available as a Python API — not just YAML. This is critical because PromptFoo's Node.js dependency and YAML-only approach is a known friction point for ML teams:

```python
import promptguard as pg

# Define evaluators
trajectory = pg.TrajectoryEvaluator(
    required_tools=["search", "read_file"],
    forbidden_tools=["delete_file"],
    order_matters=True,
)

judge = pg.LLMJudge(
    model="claude-sonnet-4-5-20250929",
    rubric="Is the code correct and well-structured?",
    threshold=4.0,
)

cost_gate = pg.CostGate(max_per_task=0.50, max_total=25.00)

# Run tests
result = pg.test(
    agent=wrap_openai_agent(my_agent),  # Any framework
    dataset="tests/coding-tasks.jsonl",
    evaluators=[trajectory, judge, cost_gate],
    trials=5,
    parallel=True,
)

# Use in pytest
assert result.passed
assert result.confidence_interval("task_success_rate").lower >= 0.85

# Export traces to Langfuse
result.export_to_langfuse(public_key="pk-...", secret_key="sk-...")
```

---

## What We Leverage vs. What We Build

| Capability | Build in PromptGuard | Leverage From |
|---|---|---|
| Framework adapters | **Build** | — |
| Trajectory evaluation | **Build** | — |
| Multi-trial statistical testing | **Build** | — |
| Quality gate engine | **Build** | — |
| CI/CD integration | **Build** (extend existing) | — |
| LLM-as-Judge evaluation | **Build** | — |
| Custom evaluator plugins | **Build** (extend registry pattern) | — |
| Regression detection | **Extend** existing baseline/diff | — |
| Red teaming / security | Delegate | **PromptFoo** |
| Production trace ingestion | Build connector | **Langfuse** |
| Trace visualization | Export to | **Langfuse** |
| Trace format standard | Adopt | **OpenTelemetry GenAI** |
| Provider SDKs | Use native | **OpenAI/Anthropic/Google SDKs** |

---

## Implementation Phases

### Phase 1: Foundation (v0.3) — "Make It Work With Any Agent"

**Goal**: Universal agent adapters + real tool-calling + LLM-as-judge

1. **`promptguard.adapters` module** — Adapters for OpenAI Agents SDK, Claude Agent SDK, LangChain, CrewAI, Google ADK, and raw OpenTelemetry
2. **`AgentTrace` data model** — Universal trace format based on OTel GenAI semantic conventions
3. **Real provider tool-calling** — Replace the regex-based `_extract_tool_calls` with actual API tool-calling for each provider
4. **LLM-as-Judge evaluator** — Implement the placeholder in `agent/evaluator.py`
5. **Trajectory evaluator** — Tool call sequence validation with argument checking
6. **Python SDK** — `promptguard.test()` API for programmatic use with pytest

### Phase 2: Statistical Engine (v0.4) — "Make It Reliable"

**Goal**: Multi-trial execution + statistical significance + cost tracking

1. **Multi-trial execution** — Run N trials per task with configurable concurrency
2. **Statistical analysis** — Confidence intervals, variance detection, significance testing
3. **Cost tracking** — Token-to-dollar conversion using provider pricing tables + cost thresholds
4. **Parallel execution** — Concurrent task and trial execution with rate limiting
5. **Watch mode** — `promptguard watch` for iterative development

### Phase 3: Integration Layer (v0.5) — "Make It a Platform"

**Goal**: Langfuse bridge + PromptFoo delegation + webhook system

1. **Langfuse trace ingestion** — Pull production traces, convert to test datasets
2. **Langfuse trace export** — Push test traces to Langfuse for visualization
3. **PromptFoo delegation** — Automated red team testing via PromptFoo subprocess
4. **Webhook / event system** — Configurable hooks for monitoring, alerting, Slack
5. **Evaluator plugin system** — `@register_evaluator` decorator for community plugins
6. **GitHub Action v2** — Full-featured action with PR comments, quality gates, trace links

### Phase 4: Enterprise & Scale (v0.6) — "Make It Production-Ready"

**Goal**: Sandbox execution, multi-agent pipelines, advanced features

1. **Sandboxed execution** — Docker-based tool execution for safe testing
2. **Multi-agent pipeline testing** — Test agent handoffs and orchestration
3. **Dataset management** — Version-controlled datasets with Langfuse sync
4. **Trace replay / debug** — `promptguard trace` command for step-by-step debugging
5. **Policy packs** — Reusable, shareable evaluation rule sets (safety, compliance, etc.)

---

## Why This Wins

### For AI Engineers

| Pain Point | Current Solutions | PromptGuard |
|---|---|---|
| "My agent works locally but breaks in prod" | Manual testing, hope | Statistical multi-trial testing with CI gates |
| "I switched from LangChain to CrewAI, tests are useless" | Rewrite everything | Change one line (adapter), keep all tests |
| "I have Langfuse but can't block bad deploys" | DIY pytest scripts | Native quality gates consuming Langfuse traces |
| "PromptFoo can't see my agent's tool calls" | Black-box testing only | Full trajectory evaluation with argument validation |
| "Agent tests are flaky" | Run again and hope | Multi-trial with confidence intervals and variance flagging |
| "Red teaming is separate from functional tests" | Two tools, two pipelines | PromptFoo delegation integrated into one quality gate |
| "Each framework has different testing" | Learn N testing tools | One tool, universal adapters |

### For the Ecosystem

PromptGuard doesn't try to replace Langfuse or PromptFoo. It becomes the **connective tissue** between them:

```
Production Agent → Langfuse (observe) → PromptGuard (test) → PromptFoo (secure) → CI/CD (gate)
       ↑                                       │
       └───────────────────────────────────────┘
                  regression prevention loop
```

This is the missing piece. And being Python-native, open-source (MIT), and framework-agnostic makes it the natural choice for any AI engineering team.
