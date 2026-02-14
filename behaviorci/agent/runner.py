"""Agent runner for executing agent tasks with tool calling."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

from behaviorci.agent.models import (
    AgentCaseResult,
    AgentConfig,
    AgentTurn,
    ExecutionConfig,
    MemoryConfig,
    Task,
    ToolCall,
    ToolResult,
)
from behaviorci.providers.base import LLMProvider, ProviderResponse


class ToolExecutor(Protocol):
    """Protocol for tool execution."""

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool and return the result."""
        ...


class SimpleToolExecutor:
    """Simple tool executor for testing and development."""

    def __init__(self, tools: Optional[Dict[str, Any]] = None):
        """Initialize with tool definitions.

        Args:
            tools: Dictionary of tool name -> tool config
        """
        self.tools = tools or {}

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool (simulated).

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            ToolResult with execution outcome
        """
        # Simulate tool execution
        try:
            result = self._simulate_tool(tool_name, arguments)
            return ToolResult(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                error=None,
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                error=str(e),
            )

    def _simulate_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Simulate tool execution based on tool name."""
        tool_name_lower = tool_name.lower()

        if "read" in tool_name_lower or "file" in tool_name_lower:
            # Simulate file reading
            filename = arguments.get("path") or arguments.get("file") or "unknown"
            return f"[Simulated content of {filename}]"

        elif "write" in tool_name_lower or "create" in tool_name_lower:
            # Simulate file writing
            filename = arguments.get("path") or arguments.get("file") or "unknown"
            content = arguments.get("content") or arguments.get("text") or ""
            return f"Successfully wrote to {filename} ({len(content)} chars)"

        elif "list" in tool_name_lower or "directory" in tool_name_lower or "ls" in tool_name_lower:
            # Simulate directory listing
            return ["file1.txt", "file2.py", "file3.md"]

        elif "execute" in tool_name_lower or "command" in tool_name_lower or "run" in tool_name_lower:
            # Simulate command execution
            cmd = arguments.get("command") or arguments.get("cmd") or ""
            return f"[Simulated output of: {cmd}]"

        elif "search" in tool_name_lower or "find" in tool_name_lower:
            # Simulate search
            return ["result1", "result2", "result3"]

        else:
            # Generic simulation
            return f"[Simulated result for {tool_name}]"


class AgentRunner:
    """Runner for executing agent tasks with tool calling support.

    Handles multi-turn conversations, tool execution, and trace recording.
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str,
        agent_config: AgentConfig,
        execution_config: ExecutionConfig,
        memory_config: MemoryConfig,
        tool_executor: Optional[ToolExecutor] = None,
        tools: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize agent runner.

        Args:
            provider: LLM provider for generating responses
            system_prompt: System prompt for the agent
            agent_config: Agent configuration
            execution_config: Execution settings
            memory_config: Memory configuration
            tool_executor: Optional custom tool executor
            tools: Tool definitions available to the agent
        """
        self.provider = provider
        self.system_prompt = system_prompt
        self.agent_config = agent_config
        self.execution_config = execution_config
        self.memory_config = memory_config
        self.tool_executor = tool_executor or SimpleToolExecutor(tools)
        self.tools = tools or {}

    async def execute_task(self, task: Task) -> AgentCaseResult:
        """Execute a single task.

        Args:
            task: Task to execute

        Returns:
            AgentCaseResult with execution details
        """
        start_time = time.perf_counter()
        turns: List[AgentTurn] = []
        all_tool_calls: List[ToolCall] = []

        # Build conversation history
        conversation_history: List[Dict[str, Any]] = []

        # Add system prompt
        if self.system_prompt:
            conversation_history.append({
                "role": "system",
                "content": self.system_prompt,
            })

        # Add task as user message
        conversation_history.append({
            "role": "user",
            "content": task.input,
        })

        current_step = 0
        max_steps = self.execution_config.max_steps or self.agent_config.max_steps

        try:
            while current_step < max_steps:
                # Generate response from provider
                response = await self._generate_with_tools(conversation_history)

                current_step += 1

                # Extract tool calls from response
                tool_calls = self._extract_tool_calls(response.content)

                if not tool_calls:
                    # No more tool calls - agent is done
                    turn = AgentTurn(
                        step=current_step,
                        input_text=self._format_conversation_for_input(conversation_history),
                        output_text=response.content,
                        tool_calls=[],
                        tool_results=[],
                    )
                    turns.append(turn)

                    # Record token usage
                    token_usage = 0
                    if response.usage:
                        token_usage = (
                            response.usage.get("input_tokens", 0) +
                            response.usage.get("output_tokens", 0)
                        )

                    latency_ms = (time.perf_counter() - start_time) * 1000

                    return AgentCaseResult(
                        task_id=task.task_id,
                        input=task.input,
                        success=True,
                        turns=turns,
                        final_output=response.content,
                        steps_used=current_step,
                        latency_ms=latency_ms,
                        token_usage=token_usage,
                        tool_calls=all_tool_calls,
                    )

                # Execute tool calls
                tool_results: List[ToolResult] = []
                for tool_call in tool_calls:
                    all_tool_calls.append(tool_call)

                    result = await self.tool_executor.execute(
                        tool_call.name,
                        tool_call.arguments,
                    )
                    tool_results.append(result)

                    # Add tool result to conversation
                    conversation_history.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tc.to_dict() for tc in [tool_call]],
                    })
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": f"call_{current_step}_{tool_call.name}",
                        "content": json.dumps(result.result) if result.result else result.error,
                    })

                # Record turn
                turn = AgentTurn(
                    step=current_step,
                    input_text=self._format_conversation_for_input(conversation_history[:-len(tool_results)*2]),
                    output_text=response.content,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                )
                turns.append(turn)

            # Max steps reached
            token_usage = 0
            if response and response.usage:
                token_usage = (
                    response.usage.get("input_tokens", 0) +
                    response.usage.get("output_tokens", 0)
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return AgentCaseResult(
                task_id=task.task_id,
                input=task.input,
                success=False,
                turns=turns,
                final_output=response.content if response else "",
                steps_used=current_step,
                latency_ms=latency_ms,
                token_usage=token_usage,
                tool_calls=all_tool_calls,
                error=f"Max steps ({max_steps}) reached",
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            return AgentCaseResult(
                task_id=task.task_id,
                input=task.input,
                success=False,
                turns=turns,
                final_output="",
                steps_used=current_step,
                latency_ms=latency_ms,
                token_usage=0,
                tool_calls=all_tool_calls,
                error=str(e),
            )

    async def _generate_with_tools(
        self,
        conversation_history: List[Dict[str, Any]],
    ) -> ProviderResponse:
        """Generate response with tool support.

        Args:
            conversation_history: Conversation messages

        Returns:
            ProviderResponse with tool calls
        """
        # Format messages for provider
        messages = self._format_messages(conversation_history)

        # Build tools specification
        tools = self._build_tools_spec()

        # For now, use simple text generation
        # In a full implementation, this would use the provider's tool calling API
        prompt = self._history_to_prompt(conversation_history)
        response = await self.provider.generate(prompt)

        return response

    def _format_messages(
        self,
        conversation_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Format conversation history for provider.

        Args:
            conversation_history: Raw conversation history

        Returns:
            Formatted messages
        """
        messages = []
        for msg in conversation_history:
            if msg.get("content") is not None:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
            elif msg.get("tool_calls"):
                # Tool call message
                messages.append({
                    "role": msg["role"],
                    "tool_calls": msg["tool_calls"],
                })
        return messages

    def _history_to_prompt(self, history: List[Dict[str, Any]]) -> str:
        """Convert conversation history to a single prompt string.

        Args:
            history: Conversation history

        Returns:
            Formatted prompt string
        """
        parts = []

        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content")

            if content is not None:
                parts.append(f"{role.upper()}: {content}")
            elif msg.get("tool_calls"):
                tc_list = []
                for tc in msg["tool_calls"]:
                    args = json.dumps(tc.get("arguments", {}))
                    tc_list.append(f"- {tc.get('name')}({args})")
                parts.append(f"ASSISTANT (tool calls):\n" + "\n".join(tc_list))
            elif msg.get("role") == "tool":
                # Tool result
                tc_id = msg.get("tool_call_id", "")
                parts.append(f"TOOL ({tc_id}): {content}")

        return "\n\n".join(parts)

    def _format_conversation_for_input(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation for display/input tracking.

        Args:
            history: Conversation history

        Returns:
            Formatted string
        """
        return self._history_to_prompt(history)

    def _extract_tool_calls(self, content: str) -> List[ToolCall]:
        """Extract tool calls from model response content.

        Args:
            content: Model response content

        Returns:
            List of ToolCall objects
        """
        tool_calls = []

        # Try to find JSON tool call blocks
        # Pattern: ```json\n{...tool_call...}\n```
        json_pattern = r'```json\s*(\{[\s\S]*?"name"\s*:\s*"([^"]+)"[\s\S]*?\})\s*```'
        matches = re.finditer(json_pattern, content, re.IGNORECASE)

        for match in matches:
            try:
                data = json.loads(match.group(1))
                tool_calls.append(ToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                    step=len(tool_calls) + 1,
                ))
            except json.JSONDecodeError:
                pass

        # Also try to find tool calls in plain text format
        # Pattern: tool_name(arg1=value1, arg2=value2)
        plain_pattern = r'(\w+)\s*\(([^)]+)\)'
        plain_matches = re.finditer(plain_pattern, content)

        for match in plain_matches:
            tool_name = match.group(1).strip()
            args_str = match.group(2).strip()

            # Skip if this looks like regular text
            if tool_name.lower() in ["read", "write", "list", "search", "execute"]:
                # Try to parse arguments
                args = self._parse_simple_args(args_str)
                if args:
                    tool_calls.append(ToolCall(
                        name=tool_name,
                        arguments=args,
                        step=len(tool_calls) + 1,
                    ))

        return tool_calls

    def _parse_simple_args(self, args_str: str) -> Dict[str, Any]:
        """Parse simple argument string into dict.

        Args:
            args_str: Argument string like "path=file.txt, content=hello"

        Returns:
            Parsed arguments
        """
        args = {}
        if not args_str:
            return args

        # Split by comma
        pairs = args_str.split(",")
        for pair in pairs:
            if "=" in pair:
                key, value = pair.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")

                # Try to parse as JSON
                try:
                    args[key] = json.loads(value)
                except json.JSONDecodeError:
                    args[key] = value

        return args

    def _build_tools_spec(self) -> List[Dict[str, Any]]:
        """Build tools specification for the model.

        Returns:
            List of tool definitions
        """
        # For now, return empty - tool calling is in text format
        # In a full implementation, this would return OpenAI tool specs
        return []
