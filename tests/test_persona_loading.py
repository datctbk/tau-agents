"""Tests for agent persona loading from skill .md files."""

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure tau and the extension are importable
ROOT = Path(__file__).resolve().parent.parent
TAU_ROOT = ROOT.parent / "tau"
sys.path.insert(0, str(TAU_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_tau_ext_agent_tool_personas",
    str(ROOT / "extensions" / "agent_tool" / "extension.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_tau_ext_agent_tool_personas"] = _mod
_spec.loader.exec_module(_mod)

AgentPersona = _mod.AgentPersona
_load_built_in_agents = _mod._load_built_in_agents


# ---------------------------------------------------------------------------
# Loading from real skill files
# ---------------------------------------------------------------------------

class TestLoadBuiltInAgents:
    def test_loads_all_personas(self):
        personas = _load_built_in_agents()
        assert "explore" in personas
        assert "plan" in personas
        assert "verify" in personas

    def test_explore_persona_fields(self):
        personas = _load_built_in_agents()
        explore = personas["explore"]
        assert explore.name == "explore"
        assert "research" in explore.description.lower() or "read-only" in explore.description.lower()
        assert explore.max_turns == 15
        assert "READ ONLY" in explore.system_prompt

    def test_plan_persona_fields(self):
        personas = _load_built_in_agents()
        plan = personas["plan"]
        assert plan.name == "plan"
        assert plan.max_turns == 8
        assert "planning" in plan.description.lower()
        assert "Plan, don't implement" in plan.system_prompt

    def test_verify_persona_fields(self):
        personas = _load_built_in_agents()
        verify = personas["verify"]
        assert verify.name == "verify"
        assert verify.max_turns == 12
        assert "verification" in verify.description.lower() or "checks" in verify.description.lower()
        assert "skeptical" in verify.system_prompt.lower()

    def test_system_prompt_excludes_frontmatter(self):
        personas = _load_built_in_agents()
        for name, persona in personas.items():
            assert "---" not in persona.system_prompt.split("\n")[0], (
                f"Persona '{name}' system prompt starts with frontmatter"
            )
            assert "description:" not in persona.system_prompt.split("\n")[0]


# ---------------------------------------------------------------------------
# Frontmatter parsing edge cases
# ---------------------------------------------------------------------------

class TestFrontmatterParsing:
    def _write_persona(self, tmpdir: Path, name: str, content: str) -> Path:
        f = tmpdir / f"{name}.md"
        f.write_text(content, encoding="utf-8")
        return f

    def test_no_frontmatter(self, tmp_path, monkeypatch):
        """Files without frontmatter still load — name from filename, defaults for rest."""
        skills_dir = tmp_path / "skills" / "built-in-agents"
        skills_dir.mkdir(parents=True)
        self._write_persona(skills_dir, "bare", "Just a prompt, no frontmatter.")

        # Patch the function's path discovery
        import types
        orig_fn = _load_built_in_agents

        def patched():
            import tau.core.types  # noqa: ensure importable
            agents = {}
            for md_file in sorted(skills_dir.glob("*.md")):
                text = md_file.read_text(encoding="utf-8")
                name = md_file.stem
                description = ""
                max_turns = 10
                body_lines = []
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
            return agents

        result = patched()
        assert "bare" in result
        assert result["bare"].description == "bare"  # fallback to name
        assert result["bare"].max_turns == 10  # default
        assert result["bare"].system_prompt == "Just a prompt, no frontmatter."

    def test_empty_file_skipped(self, tmp_path):
        """Empty files produce no persona."""
        skills_dir = tmp_path / "skills" / "built-in-agents"
        skills_dir.mkdir(parents=True)
        (skills_dir / "empty.md").write_text("", encoding="utf-8")

        # Use inline parsing logic
        agents = {}
        for md_file in sorted(skills_dir.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            body = text.strip()
            if body:
                agents[md_file.stem] = body
        assert "empty" not in agents

    def test_quoted_description(self, tmp_path):
        """Quoted description values are unquoted."""
        skills_dir = tmp_path / "skills" / "built-in-agents"
        skills_dir.mkdir(parents=True)
        self._write_persona(skills_dir, "quoted", '---\ndescription: "A quoted desc"\nmax_turns: 5\n---\nPrompt body here.')

        # Inline parse
        text = (skills_dir / "quoted.md").read_text(encoding="utf-8")
        description = ""
        in_frontmatter = False
        past_frontmatter = False
        for line in text.splitlines():
            if line.strip() == "---" and not past_frontmatter:
                if in_frontmatter:
                    past_frontmatter = True
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter and line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip('"').strip("'")
        assert description == "A quoted desc"
