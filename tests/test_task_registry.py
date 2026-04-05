"""Tests for the in-memory TaskRegistry."""

import sys
import threading
from pathlib import Path

import pytest

# Ensure tau and the extension are importable
ROOT = Path(__file__).resolve().parent.parent
TAU_ROOT = ROOT.parent / "tau"
sys.path.insert(0, str(TAU_ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "agent_tool"))

# Register module name before import (Python 3.12 dataclass compat)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_tau_ext_agent_tool",
    str(ROOT / "extensions" / "agent_tool" / "extension.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_tau_ext_agent_tool"] = _mod
_spec.loader.exec_module(_mod)

TaskRegistry = _mod.TaskRegistry
TaskEntry = _mod.TaskEntry


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestTaskRegistryCreate:
    def test_create_returns_task_with_id(self):
        reg = TaskRegistry()
        task = reg.create("test task")
        assert task.id
        assert task.name == "test task"
        assert task.status == "pending"

    def test_create_generates_unique_ids(self):
        reg = TaskRegistry()
        ids = {reg.create(f"task-{i}").id for i in range(20)}
        assert len(ids) == 20

    def test_created_task_has_timestamp(self):
        reg = TaskRegistry()
        task = reg.create("timed")
        assert task.created_at > 0


class TestTaskRegistryGet:
    def test_get_existing(self):
        reg = TaskRegistry()
        task = reg.create("find me")
        found = reg.get(task.id)
        assert found is not None
        assert found.name == "find me"

    def test_get_nonexistent(self):
        reg = TaskRegistry()
        assert reg.get("no-such-id") is None


class TestTaskRegistryUpdate:
    def test_update_existing(self):
        reg = TaskRegistry()
        task = reg.create("updatable")
        assert reg.update(task.id, status="running") is True
        assert reg.get(task.id).status == "running"

    def test_update_nonexistent(self):
        reg = TaskRegistry()
        assert reg.update("ghost", status="running") is False

    def test_update_multiple_fields(self):
        reg = TaskRegistry()
        task = reg.create("multi")
        reg.update(task.id, status="completed", result="done!")
        t = reg.get(task.id)
        assert t.status == "completed"
        assert t.result == "done!"


class TestTaskRegistryStop:
    def test_stop_existing(self):
        reg = TaskRegistry()
        task = reg.create("stoppable")
        assert reg.stop(task.id) is True
        t = reg.get(task.id)
        assert t.status == "stopped"
        assert t.completed_at is not None

    def test_stop_nonexistent(self):
        reg = TaskRegistry()
        assert reg.stop("nope") is False


class TestTaskRegistryListAll:
    def test_list_empty(self):
        reg = TaskRegistry()
        assert reg.list_all() == []

    def test_list_returns_all(self):
        reg = TaskRegistry()
        reg.create("a")
        reg.create("b")
        reg.create("c")
        assert len(reg.list_all()) == 3

    def test_list_preserves_order(self):
        reg = TaskRegistry()
        for name in ["first", "second", "third"]:
            reg.create(name)
        names = [t.name for t in reg.list_all()]
        assert names == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# TaskEntry serialization
# ---------------------------------------------------------------------------

class TestTaskEntryToDict:
    def test_to_dict_includes_all_fields(self):
        reg = TaskRegistry()
        task = reg.create("serialize me")
        d = task.to_dict()
        assert set(d.keys()) == {"id", "name", "status", "created_at", "completed_at", "result", "error"}

    def test_to_dict_truncates_long_result(self):
        entry = TaskEntry(id="x", name="big", result="A" * 500)
        d = entry.to_dict()
        assert len(d["result"]) == 200


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestTaskRegistryThreadSafety:
    def test_concurrent_creates(self):
        reg = TaskRegistry()
        errors = []

        def create_batch(start: int):
            try:
                for i in range(50):
                    reg.create(f"task-{start + i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_batch, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(reg.list_all()) == 200

    def test_concurrent_stop_and_list(self):
        reg = TaskRegistry()
        tasks = [reg.create(f"t-{i}") for i in range(20)]
        errors = []

        def stop_all():
            try:
                for t in tasks:
                    reg.stop(t.id)
            except Exception as e:
                errors.append(e)

        def list_loop():
            try:
                for _ in range(50):
                    reg.list_all()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=stop_all)
        t2 = threading.Thread(target=list_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        assert all(reg.get(t.id).status == "stopped" for t in tasks)
