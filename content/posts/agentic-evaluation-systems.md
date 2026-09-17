---

title: "Beyond Accuracy: Evaluating Agent Behavior in Production AI Systems" 
description: "A practical outline for evaluating agents across final answers, tool use, HITL decisions, workflow paths, and failure buckets." 
dateString: Sep 2026 
lastmod: 2026-09-17
tags: ["Agentic AI", "Evaluation", "LLM-as-Judge", "Benchmarking", "RAG", "AI Observability"] 
weight: 108   
showToc: true

I used to think evaluation was mostly about one question: did the model get the right answer?

That framing breaks quickly once you move from a single LLM call to a production agent.

In an agentic system, a final answer can look correct while the underlying behavior is unsafe, brittle, or impossible to trust. The agent may skip a required human review, call the wrong tool, use the right tool with the wrong arguments, retrieve the wrong evidence, or reach the answer through a path that should not be allowed in production.

That is the uncomfortable part of agent evaluation: correctness is not just about the answer. It is also about the route.

## The Problem With Final-Answer Evals

Most early evaluation setups are built around input-output pairs:

- User query
- Expected answer
- Model answer
- Pass or fail

This works for simple tasks. It is also easy to explain, easy to score, and easy to put into a dashboard.

But once the system has retrieval, extraction, tools, routing, policy checks, human-in-the-loop escalation, and multi-step workflows, final-answer scoring starts hiding the real failure modes.

A production agent can fail in several ways before the final response is even generated:

- It retrieves the wrong contract clause or misses the relevant document section.
- It extracts the wrong field, date, party, obligation, or condition.
- It routes the query to the wrong skill or pipeline.
- It calls a tool that should not have been used for that request.
- It avoids HITL when the task is ambiguous or risky.
- It triggers HITL too often and makes the workflow unusable.
- It produces the right-looking answer from the wrong evidence.
- It gets lucky on one example but follows a path that will not generalize.

The final answer can be correct and the agent can still be wrong.

## What I Learned Building Agentic Eval

In my work on agentic evaluation for legal AI systems, the goal was not only to decide whether an output was correct. The goal was to make failures useful.

That changed the design of the evaluation system.

Instead of treating every failed row as a generic miss, the eval had to explain what kind of miss it was:

- Was retrieval weak?
- Was extraction wrong?
- Did the agent choose the wrong tool?
- Did it use the right tool incorrectly?
- Did it skip a required human review?
- Did it take a random path that happened to produce a plausible answer?
- Did it refuse correctly because the request was outside supported behavior?
- Did the eval itself fail because the benchmark expected the wrong thing?

This is where agentic eval becomes less like a leaderboard and more like engineering infrastructure.

## Two Agents, One Product (Example)

To make the rest of this post concrete, imagine a contract intelligence product with two specialized agents behind the same metadata catalog.

**Query agent.** A user asks: *"Which active MSAs with Acme expire in the next ninety days?"* The agent retrieves relevant fields, may ask the user to disambiguate a party name, and produces **SQL** that filters the contract database. Success means the right columns, filters, and scope — not a polished paragraph.

**Analytics agent.** A user asks: *"Break down contract count by region and supplier."* The agent plans a chart-style answer and emits a **structured aggregation request**: what to count, how to group, which filters apply. Success means the right metric, dimensions, and product-supported fields — often with a short summary tied to the returned data.

Same platform. Same field catalog. Different artifacts, different failure modes, different evaluators. A dashboard that labels both rows "agentic" hides that distinction until you inspect the trace.

## Outcome Correctness Is Only One Layer

The first layer is still outcome correctness. You need to know whether the final answer matches user intent.

But in production, this layer should not stand alone.

For a legal or contract AI system, the answer has to be grounded in the right document, the right clause, the right metadata, and the right workflow constraints. If the model answers from incomplete evidence, the answer is not reliable even when it sounds fluent.

This is why I think of final-answer evaluation as the last check, not the whole check.

## Tool-Use Correctness

Agents often look intelligent because they can call tools. But tool use creates a new evaluation surface.

The eval should ask:

- Did the agent call the right tool?
- Did it call the tool at the right time?
- Were the arguments valid?
- Did the tool output support the next step?
- Did the agent ignore a tool result it should have used?
- Did it call unnecessary tools to arrive at the answer?

This matters because tool misuse is often invisible in the final response. A user sees a clean answer. The system trace shows whether that answer was produced through a dependable workflow.

## HITL Is Also Something To Evaluate

Human-in-the-loop is often discussed as a safety feature, but in practice it has to be evaluated like any other agent behavior.

An agent should escalate when the situation requires human judgment. It should not escalate just because the prompt is slightly unfamiliar.

So the eval needs to capture both sides:

- Did the agent trigger HITL when uncertainty, ambiguity, risk, or policy required it?
- Did the agent avoid unnecessary HITL when the task could be safely handled automatically?

This is a subtle metric because it is not just accuracy. It is judgment under constraints.

For enterprise systems, especially in legal, finance, insurance, healthcare, or compliance-heavy workflows, this can be as important as the final answer itself.

## Path Correctness: The Part Most Evals Miss

One of the most important lessons for me was that a correct answer from the wrong path is not always acceptable.

In production, agents do not just need to answer. They need to follow an allowed workflow.

For example:

- A document question should use the relevant evidence, not a memorized or guessed answer.
- A structured query should use supported metadata fields, not hallucinated schema.
- An analytics-style breakdown should only proceed if the field supports that operation in the product.
- A risky or ambiguous task should pass through the right review step.
- A refusal should be counted as correct when the system cannot safely fulfill the request.

This means evaluation needs to inspect the trajectory, not only the final message.

The question becomes: did the agent arrive at the answer through a path we would trust if this happened thousands of times in production?

**Different agents, different artifacts.** The query agent and analytics agent from the example above look equally "agentic" in a run list, but you evaluate different outputs: SQL against a schema versus aggregation requests with metrics, group-by dimensions, and filters. Each evaluator needs the vocabulary the product actually uses for that path. A trace timeline — retrieve, plan, delegate, output — lets you score path and outcome separately. Stale reasoning with a correct final payload is a different failure from a wrong payload with polished prose.

## Valid Refusals Should Be First-Class Outcomes

Not every successful agent interaction ends with a direct answer.

Sometimes the correct behavior is refusal:

- The requested field is unsupported.
- The document does not contain enough evidence.
- The requested operation is not allowed.
- The query is outside the product boundary.
- The agent needs human review before continuing.

If the benchmark treats every refusal as a failure, it teaches the wrong behavior. The agent learns to always answer, even when answering is unsafe.

Good evaluation should separate bad refusal from valid refusal.

**Refusal is not the same as empty output.** A valid refusal — unsupported field, dimension that cannot be charted, out-of-scope request — is success under product rules. An analytics agent that finishes cleanly but sends no aggregation request and no usable summary is an execution failure, not a refusal. Bucketing both as "failed" pushes unsafe over-answering and hides real bugs. When refusal rules improve, you should be able to re-judge stored traces without re-running the agent.

## Stage-Level Observability

Once an agent has multiple stages, the eval system needs observability at every stage.

For an extraction or document intelligence pipeline, I would want to inspect:

- Input quality
- Retrieval results
- Reranking or filtering
- Extracted entities and fields
- Tool calls
- Intermediate reasoning or planning
- HITL decision
- Final response
- Judge result
- Failure category

Without this, every failed benchmark row becomes a debugging exercise from scratch.

With this, failures become searchable, bucketed, and actionable.

## LLM-as-Judge Is Useful, But Not Enough

LLM judges are helpful when the output is open-ended, when gold labels are incomplete, or when the expected behavior is too nuanced for exact matching.

But I would not make the judge the only source of truth.

A stronger setup combines:

- Deterministic checks for schema, metadata, tool arguments, and exact constraints
- Gold labels where they exist
- Structural alignment when output format matters
- LLM-as-judge for semantic quality and trajectory reasoning
- Human override or review for ambiguous cases
- Re-judging stored runs when the evaluation rules improve

This hybrid approach keeps evaluation flexible without turning it into vibes.

## Gold-Free Structural Alignment

Most real benchmark rows are question-only: no hand-written expected SQL for the query agent, no expected aggregation payload for the analytics agent. Judge-only scoring is slow, noisy, and hard to regression-test.

A pattern that worked well: extract a **blind intent spec** — filters, aggregation, group-by — from the question and field catalog alone, without seeing the agent output. Then **align** that spec structurally against what the agent produced. Pass or fail only when the check is certain; otherwise abstain and escalate to the jury.

Three design choices matter:

- **Precision over coverage.** A confident wrong spec creates false failures. When the question is genuinely ambiguous, mark it ambiguous and let the jury decide — do not guess.
- **Same concept, different fields.** Catalogs repeat ideas under different labels ("region" vs a more specific geographic field). Alignment must use labels and descriptions, not exact string match on one canonical name.
- **Tag the evidence.** When alignment settles a row, record that separately from a jury verdict so teams know which scores are near-deterministic and which are model judgment.

```mermaid
flowchart LR
  Pre[Pre-checks] --> Align[Alignment pass / fail / abstain]
  Align -->|settled| Done[Verdict + evidence tag]
  Align -->|abstain| Jury[LLM jury]
  Jury --> Reconcile[Reconcile rules]
  Reconcile --> Done
```



## Failure Bucketing Is The Feedback Loop

The most useful eval system is not the one that says "failed."

It is the one that says:

- Failed because retrieval missed the supporting evidence.
- Failed because the agent used a hallucinated field.
- Failed because HITL was skipped.
- Failed because the tool call was correct but the synthesis was wrong.
- Failed because the benchmark expected an invalid behavior.
- Failed because this was actually a valid refusal.

This is what makes the system useful for developers.

The output of evaluation should become the input to improvement:

- New golden dataset examples
- New edge cases
- Better rubrics
- Better tool specs
- Better retrieval training data
- Better regression gates
- Better product constraints

## Who Decided This Row Passed?

A single pass bit hides **how** the verdict was reached: pre-check, structural alignment, unanimous jury, split jury, human override.

Split juries should often become **review signals**, not automatic hard fails. Low agreement means "we are not sure," not necessarily "the agent is wrong."

Pass rate on a run mixes agent behavior, judge strictness, and spec quality. Tightening or fixing eval rules can move the score without the agent changing at all. Treat the headline number as one signal; show the **evidence mix** — how many rows settled by alignment versus jury versus deterministic checks — so nobody over-trusts a dashboard percentage.

## Intent Spec Ambiguity Is a Feature

When several field interpretations are defensible, the intent extractor should say so. Alignment abstains; the jury or a human picks up the row.

When variants are the **same concept at different granularity** — multiple date or region fields that product rules treat as interchangeable — list them as alternatives, not ambiguity. Punishing the agent for any valid variant teaches the wrong lesson and mirrors how the product itself is supposed to behave.

## How I Would Design A Production Agentic Eval Stack

At a high level, my preferred architecture looks like this:

```mermaid
flowchart TD
  GDS[Golden dataset] --> Run[Agent execution]
  Run --> Trace[Trace capture]
  Trace --> Checks[Deterministic checks]
  Checks --> Judge[LLM judge]
  Judge --> Buckets[Failure buckets]
  Buckets --> Dashboard[Observability dashboard]
  Dashboard --> Dev[Developer fixes]
  Dashboard --> Data[Dataset improvements]
  Dev --> Regression[Regression gates]
  Data --> GDS
  Regression --> Run
```



The core loop is simple: run the agent, capture the behavior, score the outcome and the path, bucket the failure, and feed that back into datasets and engineering fixes.

## Benchmarks Are Not The Product

Public benchmarks are useful because they give us shared language.

AgentBench, WebArena, SWE-bench, OSWorld, GAIA, ToolBench, and tau-bench all show different ways to measure agents beyond static question answering.

But for a real product, the most important benchmark is usually private:

- Your users
- Your workflows
- Your tools
- Your policies
- Your failure modes
- Your acceptable risk boundary

That is why benchmark design is product work as much as ML work.

**The eval harness is also software.** Version alignment rules, pre-checks, and jury prompts like application code. Support re-judging stored runs when rules fix false failures — otherwise you cannot tell "agent regressed" from "we fixed the judge." Small curated scenario probes with expected failure categories belong in CI: not as proof of judge accuracy, but as a guardrail against prompt drift.

## Further Reading

Four papers that shaped how I think about this stack:

1. **[Zheng et al., 2023 — *Judging LLM-as-a-Judge* (MT-Bench)](https://arxiv.org/abs/2306.05685)** — Foundational work on using LLMs to evaluate open-ended outputs. Useful baseline for why a jury helps, and why it still needs guardrails.
2. **[Norman et al., 2026 — *Reliability without Validity](https://arxiv.org/abs/2606.19544)*** — Large-scale study showing high judge agreement can coexist with systematic bias. Motivates evidence tiers, split-jury review, and calibrating against human labels (e.g. Cohen's κ) before gating releases on pass rate.
3. **[PAJAMA, 2025 — *Programs are the Future of Evaluation](https://arxiv.org/abs/2506.10403)*** — Argues for synthesizing auditable programmatic checkers alongside LLM judges. Aligns with deterministic pre-checks and structural alignment before the jury runs.
4. **[BenchJack, 2026 — *Auditing AI Agent Benchmarks](https://arxiv.org/html/2605.12673)*** — Red-teams evaluation harnesses, not just agents. Supports treating the benchmark itself as software that can be gamed or drift.

## Closing Thought

My current view is that agentic evaluation is not a one-time scoring layer. It is a reliability system.

A good eval stack should tell you whether the agent answered correctly, whether it behaved correctly, whether it followed a trusted path, whether it escalated at the right time, and what needs to improve next.

The hardest part was not building a judge. It was deciding **when not to judge** — when deterministic or structural checks are enough, when to abstain, and when to flag for human review. Mature agentic eval looks like **triangulation**: several independent, imperfect checks that agree, rather than one confident LLM verdict.

For production AI agents, that is the difference between a demo that works and a system you can keep improving.