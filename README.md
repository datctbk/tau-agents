# tau-agents

Multi-agent orchestration extension for [tau](https://github.com/datctbk/tau).

Use `tau-agents` when you want bounded delegation and background task tracking without increasing tau core complexity.

## Install

```bash
tau install git:github.com/datctbk/tau-agents
```

Or for local development:

```bash
tau install ./path/to/tau-agents
```

## What's Included

### Extension: `agent_tool`

Registers four tools:

| Tool | Description |
|------|-------------|
| `agent` | Spawn a sub-agent with a task and optional persona |
| `send_message` | Inject a message into the parent's steering queue |
| `task_create` | Create a tracked background task |
| `task_stop` | Stop a running background task |

### Slash Commands

| Command | Description |
|---------|-------------|
| `/agents` | List available agent personas |
| `/tasks` | Show tracked background tasks |

### Built-in Agent Personas

| Persona | Description |
|---------|-------------|
| `explore` | Read-only research agent — explores code, reads files, searches patterns |
| `plan` | Planning agent — analyzes tasks and produces structured implementation plans |
| `verify` | Verification agent — checks changes are correct and nothing is broken |

## Usage

The `agent` tool is available to the LLM automatically. The model can delegate work:

```
User: Refactor the authentication module
```

## Notes

- Keep delegated tasks concrete and bounded.
- Prefer read-only personas (`explore`) for investigation and `verify` for checks.
- Use parent session task tracking (`task_create`/`task_stop`) for visibility.
