"""Pydantic models for Agent Bundle specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Tool Configuration Models
# ============================================================================


class ToolType(str, Enum):
    """Types of tools available to agents."""

    MCP = "mcp"
    FUNCTION = "function"


class MCPToolConfig(BaseModel):
    """Configuration for an MCP server tool."""

    type: Literal["mcp"] = "mcp"
    server: str = Field(description="MCP server name")
    commands: List[str] = Field(
        default_factory=list,
        description="List of MCP commands available from this server",
    )


class FunctionToolConfig(BaseModel):
    """Configuration for a function/tool definition."""

    type: Literal["function"] = "function"
    name: str = Field(description="Tool/function name")
    description: str = Field(description="Tool description for the agent")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for tool parameters",
    )


ToolConfig = Union[MCPToolConfig, FunctionToolConfig]


# ============================================================================
# Memory Configuration
# ============================================================================


class MemoryType(str, Enum):
    """Types of memory available to agents."""

    CONVERSATION = "conversation"
    VECTOR = "vector"
    HYBRID = "hybrid"


class MemoryConfig(BaseModel):
    """Configuration for agent memory."""

    enabled: bool = Field(default=True, description="Enable memory")
    type: MemoryType = Field(
        default=MemoryType.CONVERSATION,
        description="Type of memory to use",
    )
    max_turns: int = Field(
        default=20,
        ge=1,
        description="Maximum conversation turns to remember",
    )


# ============================================================================
# Execution Configuration
# ============================================================================


class ExecutionConfig(BaseModel):
    """Configuration for agent execution."""

    timeout: float = Field(
        default=60.0,
        gt=0,
        description="Timeout per task in seconds",
    )
    retry_on_failure: bool = Field(
        default=True,
        description="Retry failed tasks",
    )
    record_traces: bool = Field(
        default=True,
        description="Record full execution traces",
    )
    max_steps: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum tool-call iterations per task",
    )


# ============================================================================
# Provider Configuration
# ============================================================================


class ProviderConfig(BaseModel):
    """Configuration for the LLM provider."""

    name: str = Field(default="openai", description="Provider name")
    model: Optional[str] = Field(default=None, description="Model to use")
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate",
    )


# ============================================================================
# Agent Configuration
# ============================================================================


class AgentType(str, Enum):
    """Types of agents."""

    TOOL_CALLING = "tool-calling"
    REASONING = "reasoning"
    HYBRID = "hybrid"


class AgentConfig(BaseModel):
    """Configuration for the agent itself."""

    type: AgentType = Field(
        default=AgentType.TOOL_CALLING,
        description="Agent type",
    )
    model: str = Field(description="Model to use (e.g., gpt-4o)")
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_steps: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum tool-call iterations",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate",
    )


# ============================================================================
# Task Models
# ============================================================================


class TaskExpected(BaseModel):
    """Expected outcomes for a task."""

    tool_calls: Optional[List[str]] = Field(
        default=None,
        description="Expected tool calls (by name)",
    )
    files_created: Optional[List[str]] = Field(
        default=None,
        description="Expected files to be created",
    )
    contains: Optional[str] = Field(
        default=None,
        description="Expected substring in output",
    )
    file_content_contains: Optional[str] = Field(
        default=None,
        description="Expected substring in file content (for file read tasks)",
    )
    custom: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom expected values",
    )


class Task(BaseModel):
    """A single test task for the agent."""

    task_id: str = Field(description="Unique task identifier")
    input: str = Field(description="Task input/prompt")
    expected: Optional[TaskExpected] = Field(
        default=None,
        description="Expected outcomes",
    )


# ============================================================================
# Evaluation Models
# ============================================================================


class EvaluationToolUsage(BaseModel):
    """Evaluation rules for tool usage."""

    required_tools: List[str] = Field(
        default_factory=list,
        description="Tools that must be used",
    )
    forbidden_tools: List[str] = Field(
        default_factory=list,
        description="Tools that must not be used",
    )
    order_matters: bool = Field(
        default=False,
        description="Whether tool call order matters",
    )


class EvaluationOutput(BaseModel):
    """Evaluation rules for output validation."""

    type: Literal["schema", "contains", "regex", "llm-judge"] = Field(
        default="contains",
        description="Type of output validation",
    )
    schema_path: Optional[str] = Field(
        default=None,
        description="Path to JSON schema file",
    )
    contains: Optional[str] = Field(
        default=None,
        description="Substring that must be in output",
    )
    regex: Optional[str] = Field(
        default=None,
        description="Regex pattern output must match",
    )
    rubric: Optional[str] = Field(
        default=None,
        description="LLM judging rubric",
    )


class EvaluationTaskCompletion(BaseModel):
    """Evaluation rules for task completion."""

    type: Literal["llm-judge", "exact-match"] = Field(
        default="llm-judge",
        description="How to evaluate task completion",
    )
    rubric: Optional[str] = Field(
        default=None,
        description="Rubric for LLM judging (required if type is llm-judge)",
    )


class EvaluationSafety(BaseModel):
    """Safety check evaluation."""

    name: str = Field(description="Safety check name")
    description: Optional[str] = Field(
        default=None,
        description="Description of the safety check",
    )


class InvariantConfig(BaseModel):
    """Custom invariant definition."""

    name: str = Field(description="Invariant name")
    description: str = Field(description="Invariant description")
    check: str = Field(
        description="Python expression to evaluate (has access to 'tool_calls', 'output', 'input', 'context' variables)",
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration for agent runs."""

    tool_usage: Optional[EvaluationToolUsage] = Field(
        default=None,
        description="Tool usage evaluation rules",
    )
    output: Optional[EvaluationOutput] = Field(
        default=None,
        description="Output validation rules",
    )
    task_completion: Optional[EvaluationTaskCompletion] = Field(
        default=None,
        description="Task completion evaluation rules",
    )
    safety: List[EvaluationSafety] = Field(
        default_factory=list,
        description="Safety checks to run",
    )
    invariants: List[InvariantConfig] = Field(
        default_factory=list,
        description="Custom invariants",
    )


# ============================================================================
# Threshold Configuration
# ============================================================================


class ThresholdConfig(BaseModel):
    """Configuration for a threshold check."""

    metric: str = Field(
        description="Metric to evaluate (e.g., 'task_success_rate', 'tool_correctness')",
    )
    operator: Literal[">=", ">", "<=", "<", "=="] = Field(
        default=">=",
        description="Comparison operator",
    )
    value: float = Field(description="Threshold value")

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float) -> float:
        """Ensure threshold value is reasonable."""
        if v < 0:
            raise ValueError("Threshold value must be non-negative")
        return v


# ============================================================================
# Agent Bundle (top-level configuration)
# ============================================================================


class AgentBundle(BaseModel):
    """Complete Agent Bundle configuration."""

    name: str = Field(description="Human-readable bundle name")
    version: str = Field(default="1.0", description="Bundle version")
    description: Optional[str] = Field(
        default=None,
        description="Bundle description",
    )

    # Agent configuration
    agent: AgentConfig = Field(description="Agent configuration")

    # Tools available to agent
    tools: List[ToolConfig] = Field(
        default_factory=list,
        description="Tools available to the agent",
    )

    # Memory configuration
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory configuration",
    )

    # Execution settings
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Execution settings",
    )

    # Provider configuration
    provider: ProviderConfig = Field(
        default_factory=ProviderConfig,
        description="LLM provider configuration",
    )


# ============================================================================
# Execution Result Models
# ============================================================================


@dataclass
class ToolCall:
    """A tool call made by the agent."""

    name: str
    arguments: Dict[str, Any]
    step: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "step": self.step,
        }


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class AgentTurn:
    """A single turn in the agent conversation."""

    step: int
    input_text: str  # What the agent saw (could be task or tool results)
    output_text: str  # Agent's response
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_results": [tr.to_dict() for tr in self.tool_results],
        }


@dataclass
class AgentCaseResult:
    """Result of running a single task."""

    task_id: str
    input: str
    success: bool
    turns: List[AgentTurn] = field(default_factory=list)
    final_output: str = ""
    steps_used: int = 0
    latency_ms: float = 0.0
    token_usage: int = 0
    error: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    evaluation_passed: bool = True
    evaluation_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "input": self.input,
            "success": self.success,
            "turns": [t.to_dict() for t in self.turns],
            "final_output": self.final_output,
            "steps_used": self.steps_used,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "error": self.error,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "evaluation_passed": self.evaluation_passed,
            "evaluation_details": self.evaluation_details,
        }


@dataclass
class AgentRunResult:
    """Result of running all tasks in an agent bundle."""

    bundle_name: str
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    task_results: List[AgentCaseResult] = field(default_factory=list)

    # Computed metrics
    task_success_rate: float = 0.0
    tool_correctness: float = 0.0
    avg_steps_used: float = 0.0
    avg_latency_ms: float = 0.0
    total_token_usage: int = 0
    safety_violations: int = 0

    # Threshold results
    threshold_passed: bool = True
    threshold_results: List[Dict[str, Any]] = field(default_factory=list)

    # Trace data
    traces: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_name": self.bundle_name,
            "total_tasks": self.total_tasks,
            "passed_tasks": self.passed_tasks,
            "failed_tasks": self.failed_tasks,
            "metrics": {
                "task_success_rate": self.task_success_rate,
                "tool_correctness": self.tool_correctness,
                "avg_steps_used": self.avg_steps_used,
                "avg_latency_ms": self.avg_latency_ms,
                "total_token_usage": self.total_token_usage,
                "safety_violations": self.safety_violations,
            },
            "threshold_passed": self.threshold_passed,
            "threshold_results": self.threshold_results,
            "task_results": [tr.to_dict() for tr in self.task_results],
            "traces": self.traces,
        }
