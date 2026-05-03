# tau-agents — Design Document

## 1. Purpose

`tau-agents` provides multi-agent orchestration as an extension package so Tau core stays minimal.

Goals:
- Delegate bounded subtasks to sub-agents
- Track background task lifecycle
- Support lightweight persona-based specialization

## 2. Architecture

```
Tau Core Agent Loop
   ├─ ToolRegistry
   ├─ ExtensionRegistry
   │    └─ tau-agents (agent_tool extension)
   │         ├─ agent
   │         ├─ send_message
   │         ├─ task_create
   │         └─ task_stop
   └─ SteeringChannel
```

Key boundary:
- Tau core owns execution/runtime/policy boundaries.
- `tau-agents` owns orchestration semantics and persona/task UX.

## 3. Tool Surface

- `agent`: spawn/delegate sub-agent task
- `send_message`: enqueue steering/follow-up message
- `task_create`: create tracked background task metadata
- `task_stop`: stop/cancel tracked task

Slash commands:
- `/agents`
- `/tasks`

## 4. Persona Model

Built-in personas are intentionally simple:
- `explore`: read/search-focused
- `plan`: planning and decomposition
- `verify`: validation and regression checks

Personas influence prompt framing, not a separate security boundary.

## 5. State & Lifecycle

Task state is extension-scoped and ephemeral unless explicitly persisted by host workflows.

Typical lifecycle:
1. Parent turn identifies parallelizable subtask
2. `agent` spawns worker with bounded instruction
3. Worker outputs result back to parent context
4. Parent integrates and updates task status

## 6. Non-goals

- Not a distributed scheduler
- Not a replacement for Tau core session/state management
- Not a security sandbox by itself

## 7. Evolution Path

Near-term:
- richer result schemas
- explicit ownership metadata
- tighter integration with plan/task primitives in Tau core

Long-term:
- team-level orchestration patterns
- optional persisted task timelines
