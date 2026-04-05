"""tau-agents: Multi-agent orchestration extension for tau.

Provides tools for spawning sub-agents, sending messages between agents,
and managing background tasks. Uses tau's ExtensionContext.create_sub_session()
to spawn child agents that inherit the parent's provider/model/workspace.

Tools registered:
  - agent        : Spawn a sub-agent with a task and optional persona
  - send_message : Send a follow-up message to the parent's steering queue
  - task_create  : Create a tracked background task
  - task_stop    : Stop a running background task

Slash commands:
  /agents        : List available built-in agent personas
  /tasks         : Show running/completed background tasks
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tau.core.extension import Extension, ExtensionContext
from tau.core.types import (
    ExtensionManifest,
    SlashCommand,
    ToolDefinition,
    ToolParameter,
    TextDelta,
    TurnComplete,
    ErrorEvent,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task registry — lightweight in-memory task tracker
# ---------------------------------------------------------------------------

@dataclass
class TaskEntry:
    id: str
    name: str
    status: str = "pending"        # pending | running | completed | failed | stopped
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result[:200] if self.result else None,
            "error": self.error,
        }


class TaskRegistry:
    """Thread-safe in-memory task registry."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskEntry] = {}
        self._lock = threading.Lock()

    def create(self, name: str) -> TaskEntry:
        task = TaskEntry(id=str(uuid.uuid4())[:8], name=name)
        with self._lock:
            self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> TaskEntry | None:
        with self._lock:
            return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs: Any) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            for k, v in kwargs.items():
                setattr(task, k, v)
            return True

    def stop(self, task_id: str) -> bool:
        return self.update(task_id, status="stopped", completed_at=time.time())

    def list_all(self) -> list[TaskEntry]:
        with self._lock:
            return list(self._tasks.values())


# ---------------------------------------------------------------------------
# Built-in agent discovery
# ---------------------------------------------------------------------------

@dataclass
class AgentPersona:
    """A named agent persona loaded from a skill .md file."""
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] | None = None   # None = all tools
    max_turns: int = 10


def _load_built_in_agents() -> dict[str, AgentPersona]:
    """Discover agent personas from skills/built-in-agents/*.md files."""
    agents: dict[str, AgentPersona] = {}
    skills_dir = Path(__file__).resolve().parent.parent.parent / "skills" / "built-in-agents"
    if not skills_dir.is_dir():
        return agents

    for md_file in sorted(skills_dir.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
            # Parse simple YAML-ish frontmatter
            name = md_file.stem
            description = ""
            max_turns = 10
            body_lines: list[str] = []
            in_frontmatter = False
            past_frontmatter = False

            for line in text.splitlines():
                if line.strip() == "---" and not past_frontmatter:
                    if in_frontmatter:
                        past_frontmatter = True
                    in_frontmatter = not in_frontmatter
                    continue
                if in_frontmatter:
                    if line.startswith("description:"):
                        description = line.split(":", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("max_turns:"):
                        try:
                            max_turns = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                else:
                    body_lines.append(line)

            system_prompt = "\n".join(body_lines).strip()
            if system_prompt:
                agents[name] = AgentPersona(
                    name=name,
                    description=description or name,
                    system_prompt=system_prompt,
                    max_turns=max_turns,
                )
        except Exception as e:
            logger.warning("Failed to load agent persona %s: %s", md_file.name, e)

    return agents


# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------

class AgentToolExtension(Extension):
    manifest = ExtensionManifest(
        name="agent_tool",
        version="0.1.0",
        description="Multi-agent orchestration: spawn sub-agents, manage tasks, coordinate workflows.",
        author="datctbk",
    )

    def __init__(self) -> None:
        self._ext_context: ExtensionContext | None = None
        self._task_registry = TaskRegistry()
        self._personas: dict[str, AgentPersona] = {}

    def on_load(self, context: ExtensionContext) -> None:
        self._ext_context = context
        self._personas = _load_built_in_agents()
        if self._personas:
            logger.debug("Loaded %d agent personas: %s", len(self._personas), list(self._personas.keys()))

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="agent",
                description=(
                    "Spawn a sub-agent to perform a task. The sub-agent runs in its own "
                    "session with its own context, inheriting the parent's provider/model/workspace. "
                    "Use this to delegate focused work: research, planning, verification, etc. "
                    "The sub-agent has access to the same file system tools. "
                    "Returns the sub-agent's final text response.\n\n"
                    "Available personas: " + ", ".join(self._personas.keys()) if self._personas
                    else "Spawn a sub-agent to perform a task."
                ),
                parameters={
                    "task": ToolParameter(
                        type="string",
                        description="The task to delegate to the sub-agent. Be specific and detailed.",
                    ),
                    "persona": ToolParameter(
                        type="string",
                        description=(
                            "Optional named persona (e.g. 'explore', 'plan', 'verify'). "
                            "Each persona has a specialized system prompt. "
                            "If omitted, uses a generic assistant prompt."
                        ),
                        required=False,
                    ),
                    "system_prompt": ToolParameter(
                        type="string",
                        description="Optional custom system prompt. Overrides persona if both are given.",
                        required=False,
                    ),
                    "max_turns": ToolParameter(
                        type="integer",
                        description="Maximum number of agent turns (default: 10).",
                        required=False,
                    ),
                },
                handler=self._handle_agent,
            ),
            ToolDefinition(
                name="send_message",
                description=(
                    "Send a follow-up message to the parent agent's queue. "
                    "Use this when you need to inject information or instructions "
                    "back into the main conversation flow."
                ),
                parameters={
                    "message": ToolParameter(
                        type="string",
                        description="The message to enqueue.",
                    ),
                },
                handler=self._handle_send_message,
            ),
            ToolDefinition(
                name="task_create",
                description="Create a tracked task entry for monitoring purposes.",
                parameters={
                    "name": ToolParameter(
                        type="string",
                        description="Short name/description of the task.",
                    ),
                },
                handler=self._handle_task_create,
            ),
            ToolDefinition(
                name="task_stop",
                description="Stop/cancel a tracked task by its ID.",
                parameters={
                    "task_id": ToolParameter(
                        type="string",
                        description="The task ID to stop.",
                    ),
                },
                handler=self._handle_task_stop,
            ),
        ]

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def slash_commands(self) -> list[SlashCommand]:
        return [
            SlashCommand(
                name="agents",
                description="List available agent personas.",
                usage="/agents",
            ),
            SlashCommand(
                name="tasks",
                description="Show tracked background tasks.",
                usage="/tasks",
            ),
        ]

    def handle_slash(self, command: str, args: str, context: ExtensionContext) -> bool:
        if command == "agents":
            self._show_agents(context)
            return True
        if command == "tasks":
            self._show_tasks(context)
            return True
        return False

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _handle_agent(
        self,
        task: str,
        persona: str | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
    ) -> str:
        if self._ext_context is None:
            return "Error: AgentTool extension context not initialized."

        # Resolve system prompt
        resolved_prompt = system_prompt
        resolved_max_turns = max_turns or 10

        if resolved_prompt is None and persona:
            agent_persona = self._personas.get(persona)
            if agent_persona is None:
                available = ", ".join(self._personas.keys()) or "(none loaded)"
                return f"Error: Unknown persona '{persona}'. Available: {available}"
            resolved_prompt = agent_persona.system_prompt
            if max_turns is None:
                resolved_max_turns = agent_persona.max_turns

        if resolved_prompt is None:
            resolved_prompt = (
                "You are a helpful sub-agent. Complete the assigned task thoroughly "
                "and return a clear, concise result. Use the available tools as needed. "
                "Focus only on the specific task given."
            )

        # Spawn the sub-agent
        try:
            sub = self._ext_context.create_sub_session(
                system_prompt=resolved_prompt,
                max_turns=resolved_max_turns,
                session_name=f"sub-agent:{persona or 'default'}",
            )
        except Exception as e:
            return f"Error spawning sub-agent: {e}"

        # Run the sub-agent and collect its response
        try:
            with sub:
                events = sub.prompt_sync(task)

            # Extract assistant text from events
            text_parts: list[str] = []
            for event in events:
                if isinstance(event, TextDelta) and not getattr(event, "is_thinking", False):
                    text_parts.append(event.text)
                elif isinstance(event, ErrorEvent):
                    return f"Sub-agent error: {event.message}"

            result = "".join(text_parts).strip()
            if not result:
                return "(Sub-agent returned no text output.)"

            # Truncate very long results to avoid context explosion
            if len(result) > 8000:
                result = result[:8000] + "\n\n... (truncated — full output was {:,} chars)".format(len(result))

            return result

        except Exception as e:
            return f"Sub-agent execution failed: {e}"

    def _handle_send_message(self, message: str) -> str:
        if self._ext_context is None:
            return "Error: Extension context not initialized."
        self._ext_context.enqueue(message)
        return f"Message enqueued: {message[:100]}..."

    def _handle_task_create(self, name: str) -> str:
        task = self._task_registry.create(name)
        return json.dumps(task.to_dict(), indent=2)

    def _handle_task_stop(self, task_id: str) -> str:
        if self._task_registry.stop(task_id):
            task = self._task_registry.get(task_id)
            return f"Task {task_id} stopped." + (f" ({task.name})" if task else "")
        return f"Error: Task '{task_id}' not found."

    # ------------------------------------------------------------------
    # Slash command display
    # ------------------------------------------------------------------

    def _show_agents(self, context: ExtensionContext) -> None:
        if not self._personas:
            context.print("[dim]No agent personas loaded.[/dim]")
            return
        lines = ["[bold cyan]Available Agent Personas[/bold cyan]", ""]
        for name, persona in self._personas.items():
            lines.append(f"  [bold]{name}[/bold] — {persona.description}")
            lines.append(f"    [dim]max_turns: {persona.max_turns}[/dim]")
        lines.append("")
        lines.append('[dim]Usage: agent tool with persona="name"[/dim]')
        context.print("\n".join(lines))

    def _show_tasks(self, context: ExtensionContext) -> None:
        tasks = self._task_registry.list_all()
        if not tasks:
            context.print("[dim]No tasks tracked.[/dim]")
            return
        lines = ["[bold cyan]Tracked Tasks[/bold cyan]", ""]
        for t in tasks:
            icon = {"pending": "○", "running": "◉", "completed": "✓", "failed": "✗", "stopped": "■"}.get(t.status, "?")
            color = {"pending": "dim", "running": "cyan", "completed": "green", "failed": "red", "stopped": "yellow"}.get(t.status, "white")
            lines.append(f"  [{color}]{icon} {t.id}[/{color}] {t.name} [{color}]({t.status})[/{color}]")
        context.print("\n".join(lines))


# Module-level instance — required by tau's extension loader
EXTENSION = AgentToolExtension()
