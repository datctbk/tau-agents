"""Tests for the AgentToolExtension — tools, slash commands, and handlers."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure tau and the extension are importable
ROOT = Path(__file__).resolve().parent.parent
TAU_ROOT = ROOT.parent / "tau"
sys.path.insert(0, str(TAU_ROOT))

import importlib.util

_mod_name = "_tau_ext_agent_tool_ext"
_spec = importlib.util.spec_from_file_location(
    _mod_name,
    str(ROOT / "extensions" / "agent_tool" / "extension.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_mod_name] = _mod
_spec.loader.exec_module(_mod)

AgentToolExtension = _mod.AgentToolExtension
AgentPersona = _mod.AgentPersona

from tau.core.types import TextDelta, ErrorEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ext():
    """Fresh extension instance with mocked context."""
    e = AgentToolExtension()
    ctx = MagicMock()
    ctx.print = MagicMock()
    ctx.enqueue = MagicMock()
    ctx.create_sub_session = MagicMock()
    e._ext_context = ctx
    e._personas = {
        "explore": AgentPersona(
            name="explore",
            description="Read-only explorer",
            system_prompt="You are a research agent.",
            max_turns=15,
        ),
        "plan": AgentPersona(
            name="plan",
            description="Planning agent",
            system_prompt="You are a planning agent.",
            max_turns=8,
        ),
    }
    return e


@pytest.fixture
def ctx_mock():
    """Standalone mocked ExtensionContext."""
    ctx = MagicMock()
    ctx.print = MagicMock()
    ctx.enqueue = MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# Extension metadata
# ---------------------------------------------------------------------------

class TestExtensionManifest:
    def test_manifest_name(self):
        ext = AgentToolExtension()
        assert ext.manifest.name == "agent_tool"

    def test_manifest_version(self):
        ext = AgentToolExtension()
        assert ext.manifest.version == "0.1.0"


# ---------------------------------------------------------------------------
# Tools registration
# ---------------------------------------------------------------------------

class TestToolsRegistration:
    def test_registers_tools(self, ext):
        tools = ext.tools()
        assert len(tools) == 9

    def test_tool_names(self, ext):
        names = {t.name for t in ext.tools()}
        assert names == {
            "agent", "send_message", "task_create", "task_stop", "task_list",
            "task_get", "task_update", "task_events", "task_events_clear",
        }

    def test_agent_tool_has_required_params(self, ext):
        agent_tool = next(t for t in ext.tools() if t.name == "agent")
        assert "task" in agent_tool.parameters
        assert agent_tool.parameters["task"].required is True

    def test_agent_tool_has_optional_params(self, ext):
        agent_tool = next(t for t in ext.tools() if t.name == "agent")
        assert agent_tool.parameters["persona"].required is False
        assert agent_tool.parameters["system_prompt"].required is False
        assert agent_tool.parameters["max_turns"].required is False
        assert agent_tool.parameters["background"].required is False

    def test_tools_have_handlers(self, ext):
        for tool in ext.tools():
            assert callable(tool.handler), f"Tool {tool.name} has no callable handler"


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

class TestSlashCommands:
    def test_registers_two_commands(self, ext):
        cmds = ext.slash_commands()
        assert len(cmds) == 2

    def test_command_names(self, ext):
        names = {c.name for c in ext.slash_commands()}
        assert names == {"agents", "tasks"}

    def test_handle_agents_returns_true(self, ext, ctx_mock):
        assert ext.handle_slash("agents", "", ctx_mock) is True

    def test_handle_tasks_returns_true(self, ext, ctx_mock):
        assert ext.handle_slash("tasks", "", ctx_mock) is True

    def test_handle_unknown_returns_false(self, ext, ctx_mock):
        assert ext.handle_slash("unknown", "", ctx_mock) is False

    def test_show_agents_prints_personas(self, ext, ctx_mock):
        ext.handle_slash("agents", "", ctx_mock)
        output = ctx_mock.print.call_args[0][0]
        assert "explore" in output
        assert "plan" in output

    def test_show_tasks_empty(self, ext, ctx_mock):
        ext.handle_slash("tasks", "", ctx_mock)
        output = ctx_mock.print.call_args[0][0]
        assert "No tasks" in output


# ---------------------------------------------------------------------------
# Agent tool handler
# ---------------------------------------------------------------------------

class TestAgentHandler:
    def test_error_when_no_context(self):
        ext = AgentToolExtension()
        ext._ext_context = None
        result = ext._handle_agent(task="test")
        assert "Error" in result
        assert "not initialized" in result

    def test_unknown_persona_error(self, ext):
        result = ext._handle_agent(task="test", persona="nonexistent")
        assert "Error" in result
        assert "Unknown persona" in result
        assert "nonexistent" in result

    def test_persona_resolves_system_prompt(self, ext):
        # Mock the sub-session
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.prompt.return_value = [
            TextDelta(text="Research complete."),
        ]
        ext._ext_context.create_sub_session.return_value = mock_session

        result = ext._handle_agent(task="find todos", persona="explore")

        # Verify create_sub_session was called with explore's prompt
        call_kwargs = ext._ext_context.create_sub_session.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a research agent."
        assert call_kwargs["max_turns"] == 15
        assert "Research complete." in result

    def test_custom_system_prompt_overrides_persona(self, ext):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.prompt.return_value = [TextDelta(text="ok")]
        ext._ext_context.create_sub_session.return_value = mock_session

        ext._handle_agent(
            task="test",
            persona="explore",
            system_prompt="Custom prompt!",
        )

        call_kwargs = ext._ext_context.create_sub_session.call_args[1]
        assert call_kwargs["system_prompt"] == "Custom prompt!"

    def test_default_prompt_when_no_persona(self, ext):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.prompt.return_value = [TextDelta(text="done")]
        ext._ext_context.create_sub_session.return_value = mock_session

        ext._handle_agent(task="generic task")

        call_kwargs = ext._ext_context.create_sub_session.call_args[1]
        assert "sub-agent" in call_kwargs["system_prompt"].lower()

    def test_error_event_from_subagent(self, ext):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.prompt.return_value = [
            ErrorEvent(message="model failed"),
        ]
        ext._ext_context.create_sub_session.return_value = mock_session

        result = ext._handle_agent(task="fail")
        assert "Sub-agent error" in result
        assert "model failed" in result

    def test_empty_response(self, ext):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.prompt.return_value = []
        ext._ext_context.create_sub_session.return_value = mock_session

        result = ext._handle_agent(task="nothing")
        assert "no text output" in result.lower()

    def test_truncation_of_long_result(self, ext):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.prompt.return_value = [
            TextDelta(text="A" * 10000),
        ]
        ext._ext_context.create_sub_session.return_value = mock_session

        result = ext._handle_agent(task="verbose")
        assert len(result) < 10000
        assert "truncated" in result

    def test_spawn_exception_handled(self, ext):
        ext._ext_context.create_sub_session.side_effect = RuntimeError("no provider")
        result = ext._handle_agent(task="test")
        assert "Error spawning sub-agent" in result

    def test_execution_exception_handled(self, ext):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.prompt.side_effect = Exception("timeout")
        ext._ext_context.create_sub_session.return_value = mock_session

        result = ext._handle_agent(task="crash")
        assert "execution failed" in result.lower()

    @patch("threading.Thread")
    def test_background_agent_handled(self, mock_thread, ext):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.prompt.return_value = [TextDelta("ok")]
        ext._ext_context.create_sub_session.return_value = mock_session

        result = ext._handle_agent(task="test", background=True)
        assert "spawned in background" in result
        assert "Task ID:" in result
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    @patch("threading.Thread")
    def test_background_agent_registers_cancel_token(self, mock_thread, ext):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.prompt.return_value = [TextDelta("ok")]
        ext._ext_context.create_sub_session.return_value = mock_session

        result = ext._handle_agent(task="test", background=True)
        task_id_line = next(line for line in result.splitlines() if line.startswith("Task ID:"))
        task_id = task_id_line.split(":", 1)[1].strip()

        ev = ext._get_cancel_token(task_id)
        assert ev is not None
        assert ev.is_set() is False


# ---------------------------------------------------------------------------
# SendMessage handler
# ---------------------------------------------------------------------------

class TestSendMessageHandler:
    def test_enqueues_message(self, ext):
        result = ext._handle_send_message(message="hello world")
        ext._ext_context.enqueue.assert_called_once_with("hello world")
        assert "enqueued" in result.lower()

    def test_error_without_context(self):
        ext = AgentToolExtension()
        ext._ext_context = None
        result = ext._handle_send_message(message="test")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------

class TestTaskHandlers:
    def test_task_create_returns_json(self, ext):
        result = ext._handle_task_create(name="my task")
        data = json.loads(result)
        assert data["name"] == "my task"
        assert data["status"] == "pending"
        assert "id" in data

    def test_task_stop_existing(self, ext):
        create_result = ext._handle_task_create(name="stoppable")
        task_id = json.loads(create_result)["id"]
        result = ext._handle_task_stop(task_id=task_id)
        assert "stopped" in result.lower()

    @patch("threading.Thread")
    def test_task_stop_signals_background_cancel(self, mock_thread, ext):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.prompt.return_value = [TextDelta("ok")]
        ext._ext_context.create_sub_session.return_value = mock_session

        spawn_result = ext._handle_agent(task="test", background=True)
        task_id_line = next(line for line in spawn_result.splitlines() if line.startswith("Task ID:"))
        task_id = task_id_line.split(":", 1)[1].strip()

        stop_result = ext._handle_task_stop(task_id)
        assert "stopped" in stop_result.lower()
        ev = ext._get_cancel_token(task_id)
        assert ev is not None
        assert ev.is_set() is True

    def test_task_stop_nonexistent(self, ext):
        result = ext._handle_task_stop(task_id="fake-id")
        assert "Error" in result
        assert "not found" in result.lower()

    def test_task_lifecycle(self, ext):
        """Create → stop → verify status in /tasks output."""
        create_result = ext._handle_task_create(name="lifecycle test")
        task_id = json.loads(create_result)["id"]

        # Task should appear in listing
        tasks_before = ext._task_registry.list_all()
        assert any(t.id == task_id for t in tasks_before)

        # Stop it
        ext._handle_task_stop(task_id=task_id)

        # Verify state
        task = ext._task_registry.get(task_id)
        assert task.status == "stopped"
        assert task.completed_at is not None

    def test_task_list_empty(self, ext):
        assert "No tasks" in ext._handle_task_list()

    def test_task_list_filled(self, ext):
        ext._handle_task_create(name="foo")
        ext._handle_task_create(name="bar")
        result = ext._handle_task_list()
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]["name"] == "foo"

    def test_task_get_success(self, ext):
        task_id = json.loads(ext._handle_task_create("foo"))["id"]
        data = json.loads(ext._handle_task_get(task_id))
        assert data["id"] == task_id
        assert data["name"] == "foo"

    def test_task_get_fail(self, ext):
        assert "Error" in ext._handle_task_get("bad-id")

    def test_task_update(self, ext):
        task_id = json.loads(ext._handle_task_create("foo"))["id"]
        
        # update status
        result = ext._handle_task_update(task_id, status="running")
        assert "updated" in result.lower()
        t = ext._task_registry.get(task_id)
        assert t.status == "running"
        assert t.completed_at is None
        
        # update result
        ext._handle_task_update(task_id, result="done stuff", status="completed")
        t = ext._task_registry.get(task_id)
        assert t.status == "completed"
        assert t.result == "done stuff"
        assert t.completed_at is not None

    def test_task_events_stream_and_clear(self, ext):
        task_id = json.loads(ext._handle_task_create("evt"))["id"]
        ext._handle_task_update(task_id, status="running")
        ext._handle_task_update(task_id, status="completed")

        events = json.loads(ext._handle_task_events())
        assert len(events) >= 3
        assert all("seq" in e for e in events)
        assert any(e["type"] == "task.created" for e in events)
        assert any(e["type"] in ("task.started", "task.updated") for e in events)
        assert any(e["type"] == "task.completed" for e in events)

        last_seq = events[-1]["seq"]
        delta = json.loads(ext._handle_task_events(since_seq=last_seq))
        assert delta == []

        cleared = json.loads(ext._handle_task_events_clear())
        assert cleared["cleared"] >= 1
        assert json.loads(ext._handle_task_events()) == []

    @patch("threading.Thread")
    def test_background_child_links_to_parent(self, mock_thread, ext):
        parent_id = json.loads(ext._handle_task_create("parent"))["id"]

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.prompt.return_value = [TextDelta("ok")]
        ext._ext_context.create_sub_session.return_value = mock_session

        spawn_result = ext._handle_agent(
            task="child task",
            background=True,
            parent_task_id=parent_id,
        )
        child_id_line = next(line for line in spawn_result.splitlines() if line.startswith("Task ID:"))
        child_id = child_id_line.split(":", 1)[1].strip()

        parent = ext._task_registry.get(parent_id)
        child = ext._task_registry.get(child_id)
        assert parent is not None
        assert child is not None
        assert child.parent_task_id == parent_id
        assert child_id in parent.child_task_ids

    def test_parent_aggregates_child_progress_and_status(self, ext):
        parent_id = json.loads(ext._handle_task_create("parent"))["id"]
        c1 = json.loads(ext._handle_task_create("child1"))["id"]
        c2 = json.loads(ext._handle_task_create("child2"))["id"]

        assert ext._link_parent_child(parent_id, c1) is True
        assert ext._link_parent_child(parent_id, c2) is True

        ext._update_task(c1, status="running", phase="running", progress=40)
        ext._update_task(c2, status="pending", phase="queued", progress=0)

        parent = ext._task_registry.get(parent_id)
        assert parent is not None
        assert parent.status == "running"
        assert parent.progress == 20

        ext._update_task(c1, status="completed", phase="completed", progress=100, completed_at=1.0)
        ext._update_task(c2, status="completed", phase="completed", progress=100, completed_at=1.0)

        parent = ext._task_registry.get(parent_id)
        assert parent is not None
        assert parent.status == "completed"
        assert parent.progress == 100


# ---------------------------------------------------------------------------
# on_load
# ---------------------------------------------------------------------------

class TestOnLoad:
    def test_on_load_stores_context(self):
        ext = AgentToolExtension()
        ctx = MagicMock()
        ext.on_load(ctx)
        assert ext._ext_context is ctx

    def test_on_load_loads_personas(self):
        ext = AgentToolExtension()
        ctx = MagicMock()
        ext.on_load(ctx)
        # Should have loaded at least the 3 built-in personas
        assert len(ext._personas) >= 3
        assert "explore" in ext._personas

    def test_on_load_wires_task_storage(self, tmp_path):
        ext1 = AgentToolExtension()
        ctx1 = MagicMock()
        ctx1._agent_config = MagicMock()
        ctx1._agent_config.workspace_root = str(tmp_path)
        ext1.on_load(ctx1)
        created_json = ext1._handle_task_create(name="persisted task")
        task_id = json.loads(created_json)["id"]

        ext2 = AgentToolExtension()
        ctx2 = MagicMock()
        ctx2._agent_config = MagicMock()
        ctx2._agent_config.workspace_root = str(tmp_path)
        ext2.on_load(ctx2)

        loaded = ext2._task_registry.get(task_id)
        assert loaded is not None
        assert loaded.name == "persisted task"


class TestSchedulerControls:
    def test_scheduler_env_config(self):
        with patch.dict(os.environ, {"TAU_BG_MAX_CONCURRENT": "3", "TAU_BG_QUEUE_POLICY": "lifo"}, clear=False):
            ext = AgentToolExtension()
            assert ext._bg_max_concurrent == 3
            assert ext._bg_queue_policy == "lifo"

    def test_scheduler_env_defaults_on_invalid(self):
        with patch.dict(os.environ, {"TAU_BG_MAX_CONCURRENT": "bad", "TAU_BG_QUEUE_POLICY": "weird"}, clear=False):
            ext = AgentToolExtension()
            assert ext._bg_max_concurrent == 1
            assert ext._bg_queue_policy == "fifo"
