---
title: "SBOM Agentic Framework"
description: "Distributed agentic framework to identify and remediate security vulnerabilities across Docker images and GitHub repositories."
dateString: "2026"
draft: false
tags: ["Agentic AI", "Security", "SBOM", "RAG", "RLHF", "Docker", "GitHub", "LLM"]
weight: 401
showToc: false
---

### Problem

Security issues in software supply chains (Docker images, dependencies, repos) are hard to track and fix continuously, especially at scale.

### What I built

- A **distributed agentic framework** for scalable vulnerability discovery and remediation.
- AI agents that can:
  - scan **Docker images** and **GitHub repositories**
  - identify vulnerable packages / misconfigurations
  - propose fixes and generate patches
  - continuously monitor for regressions
- A **query agent** that translates English questions to **LQL** using **RAG** (few-shot) with guardrails.
- A feedback loop (**RLHF-based**) over agent executions to improve an in-house LLM and the agents’ prompts over time.

### Architecture (high-level)

```mermaid
flowchart LR
  U[User / CI Trigger] --> O[Orchestrator]
  O -->|dispatch| A1[SBOM + Dependency Agent]
  O -->|dispatch| A2[Docker Image Scanner Agent]
  O -->|dispatch| A3[Repo Scanner Agent]
  O -->|dispatch| A4[Auto-Patch / Remediation Agent]
  O -->|dispatch| A5[English → LQL Agent (RAG + Guardrails)]

  A1 --> R[(Findings Store)]
  A2 --> R
  A3 --> R
  A4 --> R
  A5 --> R

  R --> F[Feedback + Reward Modeling (RLHF)]
  F --> M[(In-house LLM / Prompt Updates)]
  M --> O
```

### Results / Impact

- Automated vulnerability detection + patching to improve **code integrity** and reduce manual security workload.
- Continuous monitoring + feedback-driven improvement loop for more reliable agent behavior.

