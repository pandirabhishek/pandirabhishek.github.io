---
title: "Simple-Agents"
description: "Extensible agentic framework with multi-agent orchestration, execution graphs, shared context, and real-time logging."
dateString: "2026"
draft: false
tags: ["Agentic AI", "Multi-Agent", "Orchestration", "APIs", "Observability"]
weight: 402
showToc: false
---

### Goal

Build a lightweight, extensible framework to orchestrate multiple specialized agents (parallel or sequential), share context, and expose progress/results for frontend visualization.

### Key features

- **Multi-agent orchestration**: run agents in parallel or sequence depending on task dependencies.
- **Extensible agent registry**: add new agents by subclassing a base `Agent` class and registering them.
- **Execution graph**: configurable dependency graph to control execution order and concurrency.
- **Context management**: shared context + aggregated results across agents.
- **Real-time logging**: structured logs accessible via API for UI/debugging.

### Execution model

```mermaid
flowchart TD
  RQ[Request] --> C[Context Init]
  C --> G[Execution Graph Planner]
  G -->|parallel| A[Agent A]
  G -->|parallel| B[Agent B]
  G -->|sequential| D[Agent D (depends on A,B)]
  A --> AGG[Result Aggregator]
  B --> AGG
  D --> AGG
  AGG --> API[API: status / logs / results]
  API --> UI[Frontend Visualization]
```

### Notes

This project is intentionally “simple”: the focus is on predictable orchestration primitives (graph + context + logs) that can be reused across problem domains.

