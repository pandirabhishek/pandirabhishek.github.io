---
title: "Controlling Foundation Models via Internal Interventions: A Survey"
description: "Survey of methods to steer and control foundation models by intervening on internal representations, activations, and inference-time mechanisms."
dateString: "2026"
draft: false
tags: ["LLM", "Foundation Models", "Mechanistic Interpretability", "Model Steering", "Inference-Time Intervention", "Survey", "NLP"]
showToc: false
weight: 201
paperUrl: "https://openreview.net/forum?id=NIwL22qGYy"
venue: "Under peer review (OpenReview)"
role: "Co-author"
---

### Overview

Survey paper reviewing **internal intervention** methods for controlling foundation models — steering model behavior by modifying internal computations (activations, representations, or inference-time dynamics) rather than only changing prompts or fine-tuning weights.

**Paper link:** [OpenReview](https://openreview.net/forum?id=NIwL22qGYy)

**Status:** Under peer review  
**Role:** Co-author

---

### What the survey covers

- **Inference-time control** — activation steering, representation editing, and intervention during generation
- **Mechanistic interpretability → control** — connecting localized model components to actionable steering
- **Tradeoffs** — controllability vs fluency, generalization, and deployment cost
- **Taxonomy** — organizing intervention methods by where and how they act inside the model stack

---

### Why it matters (interview angle)

This work sits at the intersection of **LLM reliability**, **interpretability**, and **production control** — relevant to:
- reducing hallucinations / unsafe generations without full retraining
- understanding when prompt-only control is insufficient
- designing safer enterprise LLM systems with explicit intervention layers

---

### Related work in my portfolio

- [LLM inference optimization](https://pandirabhishek.github.io/posts/llmquantization/) — cost/latency tradeoffs when adding control layers
- [Prompt compression](https://pandirabhishek.github.io/posts/prompt-compression/) — complementary approach to behavior control via context
- [Sirion RAG engineering](https://pandirabhishek.github.io/experience/sirion-mle/) — production retrieval + evaluation for grounded generation
