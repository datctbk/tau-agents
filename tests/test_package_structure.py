"""Tests for tau.json and overall package structure."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


class TestPackageStructure:
    def test_tau_json_exists(self):
        assert (ROOT / "tau.json").is_file()

    def test_tau_json_valid(self):
        data = json.loads((ROOT / "tau.json").read_text())
        assert data["name"] == "tau-agents"
        assert "version" in data

    def test_tau_json_has_extensions(self):
        data = json.loads((ROOT / "tau.json").read_text())
        assert "extensions" in data
        assert "extensions/agent_tool" in data["extensions"]

    def test_tau_json_has_skills(self):
        data = json.loads((ROOT / "tau.json").read_text())
        assert "skills" in data
        assert "skills/built-in-agents" in data["skills"]

    def test_extension_dir_exists(self):
        assert (ROOT / "extensions" / "agent_tool").is_dir()

    def test_extension_py_exists(self):
        assert (ROOT / "extensions" / "agent_tool" / "extension.py").is_file()

    def test_skills_dir_exists(self):
        assert (ROOT / "skills" / "built-in-agents").is_dir()

    def test_skill_files_exist(self):
        skills = ROOT / "skills" / "built-in-agents"
        assert (skills / "explore.md").is_file()
        assert (skills / "plan.md").is_file()
        assert (skills / "verify.md").is_file()

    def test_readme_exists(self):
        assert (ROOT / "README.md").is_file()

    def test_extension_paths_resolve(self):
        """tau.json extension paths point to real directories."""
        data = json.loads((ROOT / "tau.json").read_text())
        for ext_path in data.get("extensions", []):
            assert (ROOT / ext_path).is_dir(), f"Extension path {ext_path} does not exist"

    def test_skill_paths_resolve(self):
        """tau.json skill paths point to real directories."""
        data = json.loads((ROOT / "tau.json").read_text())
        for skill_path in data.get("skills", []):
            assert (ROOT / skill_path).is_dir(), f"Skill path {skill_path} does not exist"


class TestExtensionModule:
    """Verify the extension module loads and exposes EXTENSION."""

    def test_module_loads(self):
        import importlib.util

        mod_name = "_tau_ext_agent_tool_pkg"
        TAU_ROOT = ROOT.parent / "tau"
        sys.path.insert(0, str(TAU_ROOT))

        spec = importlib.util.spec_from_file_location(
            mod_name,
            str(ROOT / "extensions" / "agent_tool" / "extension.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

        assert hasattr(mod, "EXTENSION")
        assert mod.EXTENSION.manifest.name == "agent_tool"

    def test_extension_is_extension_subclass(self):
        import importlib.util

        mod_name = "_tau_ext_agent_tool_pkg2"
        TAU_ROOT = ROOT.parent / "tau"
        sys.path.insert(0, str(TAU_ROOT))

        spec = importlib.util.spec_from_file_location(
            mod_name,
            str(ROOT / "extensions" / "agent_tool" / "extension.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

        from tau.core.extension import Extension
        assert isinstance(mod.EXTENSION, Extension)
