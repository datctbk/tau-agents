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
import os
import queue
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
from tau.core.assistant_events import append_assistant_event, make_assistant_event

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_RETRYABLE_SUBAGENT_PATTERNS = (
    "rate limit",
    "too many requests",
    "429",
    "overloaded",
    "503",
    "502",
    "500",
    "timeout",
    "timed out",
    "connection error",
    "connection refused",
    "network error",
    "fetch failed",
    "temporarily unavailable",
)


# ---------------------------------------------------------------------------
# Task registry — lightweight in-memory task tracker
# ---------------------------------------------------------------------------

@dataclass
class TaskEntry:
    id: str
    name: str
    status: str = "pending"        # pending | running | completed | failed | stopped
    phase: str = "queued"
    progress: int = 0
    retries: int = 0
    max_retries: int = 0
    parent_task_id: str | None = None
    child_task_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: str | None = None
    error: str | None = None

    def to_storage_dict(self) -> dict[str, Any]:
        """Full-fidelity dict for persistence (no truncation)."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "phase": self.phase,
            "progress": self.progress,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "parent_task_id": self.parent_task_id,
            "child_task_ids": self.child_task_ids,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_storage_dict(cls, d: dict[str, Any]) -> "TaskEntry":
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            status=str(d.get("status", "pending")),
            phase=str(d.get("phase", "queued")),
            progress=int(d.get("progress", 0)),
            retries=int(d.get("retries", 0)),
            max_retries=int(d.get("max_retries", 0)),
            parent_task_id=d.get("parent_task_id"),
            child_task_ids=list(d.get("child_task_ids", [])),
            created_at=float(d.get("created_at", time.time())),
            completed_at=d.get("completed_at"),
            result=d.get("result"),
            error=d.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "phase": self.phase,
            "progress": self.progress,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "parent_task_id": self.parent_task_id,
            "child_task_ids": self.child_task_ids,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result[:200] if self.result else None,
            "error": self.error,
        }


class TaskRegistry:
    """Thread-safe in-memory task registry."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self._tasks: dict[str, TaskEntry] = {}
        self._lock = threading.Lock()
        self._storage_path: Path | None = storage_path
        if self._storage_path is not None:
            self._load_from_disk()

    def set_storage_path(self, storage_path: Path) -> None:
        """Configure persistent storage path and load existing tasks."""
        self._storage_path = storage_path
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self._storage_path is None or not self._storage_path.is_file():
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
            items = raw.get("tasks", []) if isinstance(raw, dict) else []
            loaded: dict[str, TaskEntry] = {}
            for it in items:
                if not isinstance(it, dict):
                    continue
                entry = TaskEntry.from_storage_dict(it)
                if entry.id:
                    loaded[entry.id] = entry
            with self._lock:
                self._tasks = loaded
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load task registry from %s: %s", self._storage_path, e)

    def _save_to_disk(self) -> None:
        if self._storage_path is None:
            return
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {
                    "version": 1,
                    "tasks": [t.to_storage_dict() for t in self._tasks.values()],
                }
            tmp = self._storage_path.with_suffix(self._storage_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._storage_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to persist task registry to %s: %s", self._storage_path, e)

    def create(self, name: str) -> TaskEntry:
        task = TaskEntry(id=str(uuid.uuid4())[:8], name=name)
        with self._lock:
            self._tasks[task.id] = task
        self._save_to_disk()
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
        self._save_to_disk()
        return True

    def stop(self, task_id: str) -> bool:
        return self.update(task_id, status="stopped", completed_at=time.time())

    def list_all(self) -> list[TaskEntry]:
        with self._lock:
            return list(self._tasks.values())


@dataclass
class TaskEventEntry:
    seq: int
    type: str
    task_id: str
    parent_task_id: str | None
    status: str
    phase: str
    progress: int
    retries: int
    timestamp: float
    message: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "type": self.type,
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "status": self.status,
            "phase": self.phase,
            "progress": self.progress,
            "retries": self.retries,
            "timestamp": self.timestamp,
            "message": self.message,
            "error": self.error,
        }


class TaskEventStream:
    """Thread-safe in-memory event stream for task lifecycle updates."""

    def __init__(self, max_events: int = 1000) -> None:
        self._lock = threading.Lock()
        self._events: list[TaskEventEntry] = []
        self._seq = 0
        self._max_events = max(100, max_events)

    def emit(
        self,
        *,
        event_type: str,
        task: TaskEntry,
        message: str | None = None,
        error: str | None = None,
    ) -> TaskEventEntry:
        with self._lock:
            self._seq += 1
            ev = TaskEventEntry(
                seq=self._seq,
                type=event_type,
                task_id=task.id,
                parent_task_id=task.parent_task_id,
                status=task.status,
                phase=task.phase,
                progress=task.progress,
                retries=task.retries,
                timestamp=time.time(),
                message=message,
                error=error,
            )
            self._events.append(ev)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]
            return ev

    def list(self, since_seq: int = 0, limit: int = 200) -> list[TaskEventEntry]:
        lim = max(1, min(limit, 2000))
        with self._lock:
            out = [e for e in self._events if e.seq > since_seq]
            return out[:lim]

    def clear(self, before_seq: int | None = None) -> int:
        with self._lock:
            if before_seq is None:
                n = len(self._events)
                self._events.clear()
                return n
            old_len = len(self._events)
            self._events = [e for e in self._events if e.seq > before_seq]
            return old_len - len(self._events)


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
    max_tool_result_chars: int = 0            # 0 = unlimited


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
            allowed_tools: list[str] | None = None
            max_tool_result_chars = 0
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
                    elif line.startswith("allowed_tools:"):
                        raw = line.split(":", 1)[1].strip()
                        if raw.startswith("[") and raw.endswith("]"):
                            allowed_tools = [t.strip().strip('"').strip("'") for t in raw[1:-1].split(",") if t.strip()]
                    elif line.startswith("max_tool_result_chars:"):
                        try:
                            max_tool_result_chars = int(line.split(":", 1)[1].strip())
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
                    allowed_tools=allowed_tools,
                    max_turns=max_turns,
                    max_tool_result_chars=max_tool_result_chars,
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
        self._workspace_root: str = "."
        self._task_registry = TaskRegistry()
        raw_max_events = os.getenv("TAU_BG_EVENT_BUFFER", "1000").strip()
        try:
            max_events = max(100, int(raw_max_events))
        except ValueError:
            max_events = 1000
        self._events = TaskEventStream(max_events=max_events)
        self._personas: dict[str, AgentPersona] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()
        raw_max = os.getenv("TAU_BG_MAX_CONCURRENT", "1").strip()
        try:
            self._bg_max_concurrent = max(1, int(raw_max))
        except ValueError:
            self._bg_max_concurrent = 1
        raw_policy = os.getenv("TAU_BG_QUEUE_POLICY", "fifo").strip().lower()
        self._bg_queue_policy = raw_policy if raw_policy in ("fifo", "lifo") else "fifo"
        raw_retries = os.getenv("TAU_BG_MAX_RETRIES", "0").strip()
        try:
            self._bg_max_retries = max(0, int(raw_retries))
        except ValueError:
            self._bg_max_retries = 0
        raw_retry_delay = os.getenv("TAU_BG_RETRY_DELAY_S", "1.0").strip()
        try:
            self._bg_retry_delay_s = max(0.0, float(raw_retry_delay))
        except ValueError:
            self._bg_retry_delay_s = 1.0
        self._bg_queue: queue.Queue[tuple[str, str, Any, tuple[Any, ...]]] | queue.LifoQueue[tuple[str, str, Any, tuple[Any, ...]]] | None = None
        self._bg_workers_started = False
        self._bg_workers_lock = threading.Lock()

    @staticmethod
    def _is_retryable_subagent_error(message: str) -> bool:
        lower = message.lower()
        return any(p in lower for p in _RETRYABLE_SUBAGENT_PATTERNS)

    def _register_cancel_token(self, task_id: str) -> threading.Event:
        ev = threading.Event()
        with self._cancel_lock:
            self._cancel_events[task_id] = ev
        return ev

    def _get_cancel_token(self, task_id: str) -> threading.Event | None:
        with self._cancel_lock:
            return self._cancel_events.get(task_id)

    def _clear_cancel_token(self, task_id: str) -> None:
        with self._cancel_lock:
            self._cancel_events.pop(task_id, None)

    def _request_cancel(self, task_id: str) -> bool:
        ev = self._get_cancel_token(task_id)
        if ev is None:
            return False
        ev.set()
        return True

    def _ensure_scheduler_started(self) -> None:
        with self._bg_workers_lock:
            if self._bg_workers_started:
                return
            if self._bg_queue_policy == "lifo":
                self._bg_queue = queue.LifoQueue()
            else:
                self._bg_queue = queue.Queue()

            def _worker_loop(worker_idx: int) -> None:
                assert self._bg_queue is not None
                while True:
                    task_id, assigned_task, runner, args = self._bg_queue.get()
                    try:
                        task = self._task_registry.get(task_id)
                        if task is None:
                            continue
                        if task.status == "stopped":
                            continue
                        runner(*args)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Background worker %d failed for task %s: %s", worker_idx, task_id, e)
                    finally:
                        self._bg_queue.task_done()

            for i in range(self._bg_max_concurrent):
                t = threading.Thread(target=_worker_loop, args=(i,), name=f"SubAgentWorker-{i}")
                t.daemon = True
                t.start()
            self._bg_workers_started = True

    def _link_parent_child(self, parent_task_id: str, child_task_id: str) -> bool:
        parent = self._task_registry.get(parent_task_id)
        child = self._task_registry.get(child_task_id)
        if parent is None or child is None:
            return False

        child_ids = list(parent.child_task_ids)
        if child_task_id not in child_ids:
            child_ids.append(child_task_id)

        self._task_registry.update(parent_task_id, child_task_ids=child_ids)
        self._task_registry.update(child_task_id, parent_task_id=parent_task_id)
        child_updated = self._task_registry.get(child_task_id)
        if child_updated is not None:
            self._events.emit(
                event_type="task.linked",
                task=child_updated,
                message=f"Linked child {child_task_id} to parent {parent_task_id}",
            )
        self._refresh_parent_chain(child_task_id)
        return True

    def _refresh_parent_chain(self, task_id: str) -> None:
        current = self._task_registry.get(task_id)
        seen: set[str] = set()
        while current is not None and current.parent_task_id and current.parent_task_id not in seen:
            seen.add(current.parent_task_id)
            self._recompute_parent_status(current.parent_task_id)
            current = self._task_registry.get(current.parent_task_id)

    def _recompute_parent_status(self, parent_task_id: str) -> None:
        parent = self._task_registry.get(parent_task_id)
        if parent is None or not parent.child_task_ids:
            return

        children = [self._task_registry.get(cid) for cid in parent.child_task_ids]
        children = [c for c in children if c is not None]
        if not children:
            return

        avg_progress = int(sum(c.progress for c in children) / len(children))
        statuses = {c.status for c in children}

        if "failed" in statuses:
            status = "failed"
            phase = "failed"
            completed_at = time.time()
        elif statuses.issubset({"completed", "stopped"}):
            if statuses == {"stopped"}:
                status = "stopped"
                phase = "cancelled"
            else:
                status = "completed"
                phase = "completed"
            completed_at = time.time()
        elif "running" in statuses:
            status = "running"
            phase = "running"
            completed_at = None
        else:
            status = "pending"
            phase = "queued"
            completed_at = None

        self._task_registry.update(
            parent_task_id,
            status=status,
            phase=phase,
            progress=avg_progress,
            completed_at=completed_at,
        )

    def _update_task(self, task_id: str, **kwargs: Any) -> bool:
        prev = self._task_registry.get(task_id)
        prev_status = prev.status if prev else None
        prev_progress = prev.progress if prev else None
        prev_retries = prev.retries if prev else None
        ok = self._task_registry.update(task_id, **kwargs)
        if ok:
            cur = self._task_registry.get(task_id)
            if cur is not None:
                event_type = "task.updated"
                if prev_status != cur.status:
                    if cur.status == "pending":
                        event_type = "task.queued"
                    elif cur.status == "running":
                        event_type = "task.started"
                    elif cur.status == "completed":
                        event_type = "task.completed"
                    elif cur.status == "failed":
                        event_type = "task.failed"
                    elif cur.status == "stopped":
                        event_type = "task.cancelled"
                elif prev_progress != cur.progress:
                    event_type = "task.progress"
                elif prev_retries != cur.retries:
                    event_type = "task.retrying"
                self._events.emit(
                    event_type=event_type,
                    task=cur,
                    error=cur.error,
                )
                append_assistant_event(
                    self._workspace_root,
                    make_assistant_event(
                        family="task",
                        name=event_type,
                        payload={
                            "task_id": cur.id,
                            "status": cur.status,
                            "phase": cur.phase,
                            "progress": cur.progress,
                            "parent_task_id": cur.parent_task_id,
                            "error": cur.error,
                        },
                        severity="error" if cur.status == "failed" else "info",
                    ),
                )
            self._refresh_parent_chain(task_id)
        return ok

    def _reconcile_orphaned_tasks_on_startup(self) -> None:
        """Finalize tasks that cannot continue after process restart.

        Background agents run as in-process threads. After tau exits, those
        workers are gone, so persisted pending/running tasks from a prior run
        must be marked interrupted to avoid an infinite "running" state.
        """
        for t in self._task_registry.list_all():
            if t.status in ("pending", "running"):
                self._update_task(
                    t.id,
                    status="failed",
                    phase="interrupted",
                    progress=min(99, max(0, t.progress)),
                    error="Task interrupted: tau process restarted before completion.",
                    completed_at=time.time(),
                )

    def on_load(self, context: ExtensionContext) -> None:
        self._ext_context = context
        workspace = "."
        if hasattr(context, "_agent_config") and context._agent_config:
            workspace = getattr(context._agent_config, "workspace_root", ".") or "."
        self._workspace_root = workspace
        storage_path = Path(workspace) / ".tau" / "agents" / "tasks.json"
        self._task_registry.set_storage_path(storage_path)
        self._reconcile_orphaned_tasks_on_startup()
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
                    "background": ToolParameter(
                        type="boolean",
                        description=(
                            "If true, spawn the agent asynchronously and return a task_id immediately. "
                            "Do NOT auto-poll status. Only use task_get when the USER explicitly asks for status/results. "
                            "When background=true, after the tool call you should only relay the tool result lines and STOP. "
                            "Do not add explanatory narration. "
                            "You can check on the agent later with task_get. "
                            "If false (default), block until the agent completes and return its full text response."
                        ),
                        required=False,
                    ),
                    "parent_task_id": ToolParameter(
                        type="string",
                        description="Optional parent task ID to link this background task as a child.",
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
            ToolDefinition(
                name="task_list",
                description="List all currently tracked background tasks and their statuses.",
                parameters={},
                handler=self._handle_task_list,
            ),
            ToolDefinition(
                name="task_get",
                description=(
                    "Get detailed information about a specific task, including output or errors. "
                    "Use ONLY when the user explicitly asks to check task status/results. "
                    "Do not call this automatically after spawning a background task. "
                    "Do not add narrative text unless the user asks for explanation."
                ),
                parameters={
                    "task_id": ToolParameter(
                        type="string",
                        description="The ID of the task to retrieve.",
                    ),
                },
                handler=self._handle_task_get,
            ),
            ToolDefinition(
                name="task_update",
                description="Update a task's status, result, or error message manually.",
                parameters={
                    "task_id": ToolParameter(
                        type="string",
                        description="The ID of the task to update.",
                    ),
                    "status": ToolParameter(
                        type="string",
                        description="Optional new status (e.g. running, completed, failed, stopped).",
                        required=False,
                    ),
                    "result": ToolParameter(
                        type="string",
                        description="Optional result string to attach to the task output.",
                        required=False,
                    ),
                    "error": ToolParameter(
                        type="string",
                        description="Optional error string to attach if the task failed.",
                        required=False,
                    ),
                },
                handler=self._handle_task_update,
            ),
            ToolDefinition(
                name="task_events",
                description=(
                    "Read structured task lifecycle events (machine-readable). "
                    "Use this for event-driven consumers instead of parsing CLI text."
                ),
                parameters={
                    "since_seq": ToolParameter(
                        type="integer",
                        description="Return events with seq > since_seq (default: 0).",
                        required=False,
                    ),
                    "limit": ToolParameter(
                        type="integer",
                        description="Maximum events to return (default: 200, max: 2000).",
                        required=False,
                    ),
                },
                handler=self._handle_task_events,
            ),
            ToolDefinition(
                name="task_events_clear",
                description="Clear task events buffer (all or up to before_seq).",
                parameters={
                    "before_seq": ToolParameter(
                        type="integer",
                        description="Optional: clear events with seq <= before_seq. If omitted, clear all.",
                        required=False,
                    ),
                },
                handler=self._handle_task_events_clear,
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
        background: bool = False,
        parent_task_id: str | None = None,
    ) -> str:
        if self._ext_context is None:
            return "Error: AgentTool extension context not initialized."

        # Resolve system prompt
        resolved_prompt = system_prompt
        resolved_max_turns = max_turns or 10
        resolved_allowed_tools: list[str] | None = None
        resolved_max_tool_result_chars = 0

        if resolved_prompt is None and persona:
            agent_persona = self._personas.get(persona)
            if agent_persona is None:
                available = ", ".join(self._personas.keys()) or "(none loaded)"
                return f"Error: Unknown persona '{persona}'. Available: {available}"
            resolved_prompt = agent_persona.system_prompt
            if max_turns is None:
                resolved_max_turns = agent_persona.max_turns
            resolved_allowed_tools = agent_persona.allowed_tools
            resolved_max_tool_result_chars = agent_persona.max_tool_result_chars

        if resolved_prompt is None:
            resolved_prompt = (
                "You are a helpful sub-agent. Complete the assigned task thoroughly "
                "and return a clear, concise result. Use the available tools as needed. "
                "Focus only on the specific task given."
            )

        def _create_sub_session():
            # Optional override for background tasks so the main chat can keep
            # using a large model while sub-agents use a lighter one.
            bg_model_override = os.getenv("TAU_BG_MODEL", "").strip() if background else ""
            return self._ext_context.create_sub_session(
                model=(bg_model_override or None),
                system_prompt=resolved_prompt,
                max_turns=resolved_max_turns,
                session_name=f"sub-agent:{persona or 'default'}",
                allowed_tools=resolved_allowed_tools,
                max_tool_result_chars=resolved_max_tool_result_chars,
            )

        def _run_sub_agent(assigned_task: str, target_task_id: str | None) -> str:
            # Unique key so concurrent agents don't clobber each other's status
            agent_key = f"agent:{persona or 'default'}:{target_task_id or id(self)}"
            cancel_ev = self._get_cancel_token(target_task_id) if target_task_id else None
            max_retries = self._bg_max_retries
            attempt = 0

            def _notify_background(status: str, task_id: str, text: str) -> None:
                if self._ext_context is None:
                    return
                preview = " ".join((text or "").strip().split())
                if len(preview) > 220:
                    preview = preview[:220] + "..."
                if not preview:
                    preview = "(no text output)"
                if status == "completed":
                    self._ext_context.print(
                        f"\n✓ Background task {task_id} completed.\n"
                        f"Preview: {preview}\n"
                        f"Hint: run task_get {task_id} for full result.\n"
                    )
                else:
                    self._ext_context.print(
                        f"\n✗ Background task {task_id} failed.\n"
                        f"Error: {preview}\n"
                        f"Hint: run task_get {task_id} for details.\n"
                    )

            try:
                while True:
                    if cancel_ev is not None and cancel_ev.is_set():
                        if target_task_id:
                            self._update_task(
                                target_task_id,
                                status="stopped",
                                phase="cancelled",
                                progress=100,
                                completed_at=time.time(),
                            )
                        self._ext_context.set_spinner("", key=agent_key)
                        return "Sub-agent cancelled."

                    if target_task_id:
                        phase = "retrying" if attempt > 0 else "running"
                        self._update_task(
                            target_task_id,
                            status="running",
                            phase=phase,
                            retries=attempt,
                            progress=5,
                        )

                    sub_name = persona or "default"
                    tool_count = 0
                    turn_count = 1
                    last_tool = ""
                    start_time = time.time()

                    def _progress_msg() -> str:
                        elapsed = int(time.time() - start_time)
                        parts = [f"🤖 {sub_name}"]
                        if turn_count:
                            parts.append(f"turn {turn_count}")
                        if last_tool:
                            parts.append(f"🛠️ {last_tool}")
                        if tool_count:
                            parts.append(f"{tool_count} tools")
                        parts.append(f"{elapsed}s")
                        return " | ".join(parts)

                    self._ext_context.set_spinner(_progress_msg(), key=agent_key)

                    try:
                        sub = _create_sub_session()
                    except Exception as spawn_exc:  # noqa: BLE001
                        raise RuntimeError(f"Error spawning sub-agent: {spawn_exc}") from spawn_exc
                    with sub:
                        events = []
                        for event in sub.prompt(assigned_task):
                            if cancel_ev is not None and cancel_ev.is_set():
                                if target_task_id:
                                    self._update_task(
                                        target_task_id,
                                        status="stopped",
                                        phase="cancelled",
                                        progress=100,
                                        completed_at=time.time(),
                                    )
                                self._ext_context.set_spinner("", key=agent_key)
                                return "Sub-agent cancelled."
                            events.append(event)
                            if type(event).__name__ == "ToolCallEvent":
                                tool_count += 1
                                if hasattr(event, "call"):
                                    last_tool = getattr(event.call, "name", "")
                                self._ext_context.set_spinner(_progress_msg(), key=agent_key)
                                if target_task_id:
                                    prog = min(95, 5 + tool_count * 10)
                                    self._update_task(target_task_id, progress=prog)
                            elif type(event).__name__ == "ToolResultEvent":
                                turn_count += 1
                                last_tool = ""
                                self._ext_context.set_spinner(_progress_msg(), key=agent_key)
                            elif type(event).__name__ == "TurnComplete":
                                self._ext_context.set_spinner(_progress_msg(), key=agent_key)

                    text_parts: list[str] = []
                    last_turn_parts: list[str] = []
                    last_error: str | None = None
                    for event in events:
                        if isinstance(event, TextDelta) and not getattr(event, "is_thinking", False):
                            last_turn_parts.append(event.text)
                        elif type(event).__name__ == "TurnComplete":
                            if last_turn_parts:
                                text_parts = last_turn_parts
                                last_turn_parts = []
                        elif isinstance(event, ErrorEvent):
                            last_error = f"Sub-agent error: {event.message}"

                    if last_turn_parts:
                        text_parts = last_turn_parts

                    if last_error:
                        raise RuntimeError(last_error)

                    result = "".join(text_parts).strip()
                    if not result:
                        result = "(Sub-agent returned no text output.)"

                    if cancel_ev is not None and cancel_ev.is_set():
                        if target_task_id:
                            self._update_task(
                                target_task_id,
                                status="stopped",
                                phase="cancelled",
                                progress=100,
                                completed_at=time.time(),
                            )
                        self._ext_context.set_spinner("", key=agent_key)
                        return "Sub-agent cancelled."

                    if len(result) > 8000:
                        result = result[:8000] + "\n\n... (truncated — full output was {:,} chars)".format(len(result))

                    if target_task_id:
                        self._update_task(
                            target_task_id,
                            status="completed",
                            phase="completed",
                            progress=100,
                            result=result,
                            retries=attempt,
                            completed_at=time.time(),
                        )
                        _notify_background("completed", target_task_id, result)
                    self._ext_context.set_spinner("", key=agent_key)
                    return result
            except Exception as e:
                msg = str(e)
                err_msg = msg if msg.startswith("Error spawning sub-agent:") else f"Sub-agent execution failed: {e}"
                if cancel_ev is not None and cancel_ev.is_set():
                    if target_task_id:
                        self._update_task(
                            target_task_id,
                            status="stopped",
                            phase="cancelled",
                            progress=100,
                            completed_at=time.time(),
                        )
                    self._ext_context.set_spinner("", key=agent_key)
                    return "Sub-agent cancelled."

                if target_task_id:
                    self._update_task(
                        target_task_id,
                        status="failed",
                        phase="failed",
                        retries=attempt,
                        error=err_msg,
                        completed_at=time.time(),
                    )
                    _notify_background("failed", target_task_id, err_msg)
                self._ext_context.set_spinner("", key=agent_key)
                return err_msg
            finally:
                if target_task_id:
                    self._clear_cancel_token(target_task_id)

        if background:
            task_name = f"Sub-agent ({persona or 'default'}): {task[:30]}..."
            tentry = self._task_registry.create(task_name)
            self._update_task(tentry.id, phase="queued", progress=0)
            if parent_task_id:
                if not self._link_parent_child(parent_task_id, tentry.id):
                    return f"Error: Parent task '{parent_task_id}' not found."
            self._register_cancel_token(tentry.id)
            self._ensure_scheduler_started()
            assert self._bg_queue is not None
            self._bg_queue.put((tentry.id, task, _run_sub_agent, (task, tentry.id)))

            return (
                f"Agent spawned in background.\n"
                f"Task ID: {tentry.id}\n"
                f"Name: {tentry.name}\n"
                "Use 'task_get' with this ID to check its status."
            )
        else:
            # Synchronous execution
            return _run_sub_agent(task, None)

    def _handle_send_message(self, message: str) -> str:
        if self._ext_context is None:
            return "Error: Extension context not initialized."
        self._ext_context.enqueue(message)
        return f"Message enqueued: {message[:100]}..."

    def _handle_task_create(self, name: str) -> str:
        task = self._task_registry.create(name)
        self._events.emit(event_type="task.created", task=task)
        append_assistant_event(
            self._workspace_root,
            make_assistant_event(
                family="task",
                name="task.created",
                payload={
                    "task_id": task.id,
                    "status": task.status,
                    "phase": task.phase,
                    "progress": task.progress,
                },
                severity="info",
            ),
        )
        return json.dumps(task.to_dict(), indent=2)

    def _handle_task_stop(self, task_id: str) -> str:
        task = self._task_registry.get(task_id)
        if not task:
            return f"Error: Task '{task_id}' not found."

        self._request_cancel(task_id)
        self._update_task(
            task_id,
            status="stopped",
            phase="cancelled",
            progress=100,
            completed_at=time.time(),
        )
        task = self._task_registry.get(task_id)
        if task is not None:
            return f"Task {task_id} stopped." + (f" ({task.name})" if task else "")
        return f"Error: Task '{task_id}' not found."

    def _handle_task_list(self) -> str:
        tasks = self._task_registry.list_all()
        if not tasks:
            return "No tasks tracked."
        return json.dumps([t.to_dict() for t in tasks], indent=2)

    def _handle_task_get(self, task_id: str) -> str:
        task = self._task_registry.get(task_id)
        if not task:
            return f"Error: Task '{task_id}' not found."
        return json.dumps(task.to_dict(), indent=2)

    def _handle_task_update(
        self,
        task_id: str,
        status: str | None = None,
        result: str | None = None,
        error: str | None = None,
    ) -> str:
        task = self._task_registry.get(task_id)
        if not task:
            return f"Error: Task '{task_id}' not found."
            
        kwargs: dict[str, Any] = {}
        if status is not None:
            kwargs["status"] = status
            if status in ("completed", "failed", "stopped"):
                kwargs["completed_at"] = time.time()
        if result is not None:
            kwargs["result"] = result
        if error is not None:
            kwargs["error"] = error
            
        if not kwargs:
            return f"No fields updated for task '{task_id}'."
            
        self._update_task(task_id, **kwargs)
        # re-fetch to get updated state
        updated = self._task_registry.get(task_id)
        return f"Task {task_id} updated:\n" + json.dumps(getattr(updated, "to_dict", lambda: {})(), indent=2)

    def _handle_task_events(self, since_seq: int | None = None, limit: int | None = None) -> str:
        events = self._events.list(since_seq=since_seq or 0, limit=limit or 200)
        return json.dumps([e.to_dict() for e in events], indent=2)

    def _handle_task_events_clear(self, before_seq: int | None = None) -> str:
        cleared = self._events.clear(before_seq=before_seq)
        return json.dumps({"cleared": cleared}, indent=2)

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
