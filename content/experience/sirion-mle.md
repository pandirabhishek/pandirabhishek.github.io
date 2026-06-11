---
title: "Machine Learning Engineer | Sirion"
description: "Sirion Labs | India — Contract AI platform engineering: async ml-context-service, RAG retrieval research, Auto-Eval platform, and inference optimization for vectorizer and language-detection deployments."
dateString: "September 2025 - Present"
draft: false
tags:
  - LLM
  - RAG
  - NLP
  - MLOps
  - FastAPI
  - Python
  - Qdrant
  - AWS
  - Pulsar
  - Docker
  - Kubernetes
  - Benchmarking
  - PyTorch
  - Hugging Face
  - Prompt Engineering
  - Vector Databases
  - Elasticsearch APM
---

# Machine Learning Engineer

**Sirion** | India

*Sept 2025 – Present*

---

## Description

### Highlights

- Re-architected `ml-context-service` to **FastAPI + async I/O**, improving throughput and eliminating cross-event-loop runtime failures under load.
- Built a **multi-tenant VectorStore proxy** routing by `client_id`, enabling transparent backends (Qdrant ↔ Amazon S3 Vectors) with **zero call-site changes**.
- Shipped **Auto-Eval**, a spec-driven evaluation platform covering **12 ML pipelines**, with regression gates, dataset tooling, and a React + FastAPI UI.
- Led production-aligned **RAG retrieval evaluation**, improving accuracy **~70% → 81%** on a 790-question golden dataset; identified retrieval as the dominant bottleneck and shipped targeted fixes.
- Delivered retrieval/model optimization work (dense fine-tuning, chunking benchmarks, hybrid search) plus infra sizing studies for high-volume document ingestion.

<details>
<summary><strong>Full write-up (details, metrics, stack)</strong></summary>

### Led Async Re-Architecture and Performance Engineering of ml-context-service

Owned the end-to-end modernization of Sirion’s core **ML Context Service** — the production backend for contract Q&A (Talk-to-Document), AE Turbo extraction, summarization, and multi-document retrieval. Migrated the service to **FastAPI + async I/O**, redesigned HTTP and vector-database clients for safe concurrent use across **uvicorn** (API routes) and **Pulsar consumer** event loops, and introduced loop-aware primitives (`LoopBoundSemaphore`, `LoopBoundLock`, `LoopBoundHTTPClient`) to eliminate cross-loop runtime failures under load.

Refactored ingestion, hybrid search (dense + BM25 + RRF), and pipeline orchestration into modular async components with bounded concurrency, APM tracing, and structured logging. Delivered measurable improvements in throughput for vectorization batches, turbo prefetch flows, and parallel embedding calls — reducing blocking on synchronous I/O and improving reliability for high-volume document processing.

---

### Built Amazon S3 Vectors Backend with Transparent Multi-Tenant Routing

Designed and implemented an **Amazon S3 Vectors** alternative to Qdrant for selected enterprise clients, using a **VectorStore protocol + dispatch proxy** pattern so existing application code required **zero call-site changes**. Routed requests per `client_id` to either Qdrant or S3 Vectors transparently via `VectorStoreProxy`.

Mapped Qdrant operations (`search`, `search_batch`, `scroll`, `put_points`, `delete`, `count`) to the S3 Vectors API (`query_vectors`, `list_vectors`, `put_vectors`, `delete_vectors`), including filter translation, metadata sanitization, BM25 sidecar storage on standard S3, lazy per-client index creation, and dimension-mismatch auto-healing. Validated end-to-end on production-like workloads (ingestion, hybrid search, TTD queries) with live AWS infrastructure.

---

### Delivered Auto-Eval — Internal ML Evaluation Platform (12 Pipelines)

Built a **spec-driven ML evaluation platform** that automates API contract validation, quality scoring, and regression gating across **12 ML pipelines** spanning `ml-context-service` (predict, turbo, summarization) and `ml-service-ns` (language detection, classification, extraction, translation, and related Pulsar flows).

**Key components:**
- Python evaluation engine with HTTP and Pulsar transport, 30+ metrics, and configurable regression gates
- Declarative **YAML** specs per flow (contracts, metrics, Excel column mapping, default prompt placeholders)
- **React + FastAPI** web UI: dataset upload, live run progress, case inspector, run history, compare, report management
- **G-Eval** LLM-as-judge integration (correctness, relevance, coherence, groundedness)
- Chained HTTP pipelines (query → predict) and async Pulsar evals with per-run callback topics
- Docker packaging for one-command team deployment; CLI + REST API for CI/CD
- **42 automated tests** covering config, ingest, API, and pipeline behavior

**Impact:** Reduced bespoke test effort, enabled self-service QA on 790+ case golden datasets, and improved pre-release confidence across contract AI and NS ML services.

---

### Drove RAG Retrieval Research and Production-Aligned Evaluation (Graph RAG / TTD)

Led end-to-end evaluation of **Graph-Based Contextual RAG (BookRAG)** on Sirion’s **TTD Golden Dataset** — 790 questions across 20 legal contracts — using production answer prompts and LLM-as-Judge scoring.

**Results:**
| Metric | Outcome |
|--------|---------|
| Final accuracy | **81.1%** (632 / 779 correct) |
| Baseline | 69.8% |
| Net improvement | **+11.3 percentage points** |
| Documents ≥ 90% accuracy | 4 of 19 |
| Documents ≥ 80% accuracy | 10 of 19 |
| Avg latency | ~7.9s per question |

Built remote GPU pipeline (graph build, TEI embeddings, vLLM reranker Qwen3-Reranker-4B, LiteLLM proxy), eval tooling (`run_rag_eval.py`, judge eval, failure-only reruns), and consolidated failure analysis. Shipped targeted retrieval fixes — entity extraction bug, section truncation for 380+ section contracts, depth expansion, fallback embedding retrieval, sibling expansion — without changing the production generation prompt. Established that **retrieval (~73% of failures)** was the primary bottleneck, informing Sirion’s RAG roadmap.

---

### Optimized Embedding Models, Chunking, and Hybrid Retrieval

Conducted deep RAG pipeline research to improve contract Q&A under strict **8k–16k token** context limits:

- **Dense retrieval fine-tuning:** Improved recall from **0.3827 (BGE)** to **0.9146 (Gemma fine-tuned)** — a ~2.4× uplift in context recall
- **Model efficiency:** Reduced embedding model size by **~50%**, improving inference latency and deployment cost
- **Sparse retrieval:** Trained and evaluated Neural Sparse and SPLADE models for hybrid search
- **Chunking strategy benchmark:** Compared fixed, semantic, sentence-based, layout-aware, and recursive approaches; established **layout-recursive chunking** as the most token-efficient with fastest recall saturation
- **Table retrieval study:** Built benchmarking dataset for raw table text vs. table descriptions; validated flexible ingestion strategies with the new embedding model
- **Multi-document TTC:** Implemented multi-document query support in Qdrant using dense + sparse search with improved context organization

---

### Benchmarking and Inference Optimization — Vectorizer, Language Detection, GenAI

Ran systematic **benchmarking and cost analysis** to guide ML infrastructure sizing and deployment decisions at Sirion scale:

- **Vectorizer deployments:** Throughput and batch-size benchmarks; delivered **100K documents/day** infrastructure and cost projections based on vectorizer + GenAI service measurements
- **Language identification (LID):** Prepared internal benchmarking datasets for priority languages; designed structured evaluation framework for translation quality; informed model selection for future NS ML integration
- **Nu-NER:** Concurrency-level scalability benchmarks across deployment configurations
- **GPT-OSS-120B:** Performance and cost benchmarking with documented findings for production GenAI routing
- **Prompt benchmarking:** Validated Markdown and structured-output prompts on **Llama 3 70B** and **GPT-OSS-120B**

These studies directly supported right-sizing GPU/CPU resources, reducing over-provisioning risk, and optimizing inference paths on the critical document-ingestion pipeline.

---

### Platform Stability, Product Features, and Defect Reduction

Hardened production contract-AI services through security, reliability, and feature delivery:

**Stability & security**
- Eliminated vulnerable dependencies; upgraded Python base image for compliance and runtime stability
- Fixed critical **shard-key misconfiguration** causing chunk mismatches in query results
- Resolved duplicate citation issues; enhanced AE Turbo logging for precise citation tracing
- Removed redundant Qdrant `_definitions` collections after dependency analysis

**Product features delivered**
- **AE Turbo multivalue support** — structured multivalue field extraction with prompt redesign for reliable LLM parsing
- **Markdown response support** — config-driven Markdown output for selected clients
- **Long-form generation** — removed token-limit restrictions for extended answers
- **Document reset mechanism** — corrected parsing/vectorization for mis-ingested 2025 documents

**Observability:** Elastic APM tracing across pipelines; structured logging for production debugging and failure-rate analysis.

---

## Selected Technical Stack

| Layer | Technologies |
|-------|----------------|
| **Backend** | Python, FastAPI, asyncio, Pydantic, httpx |
| **ML / NLP** | PyTorch, Hugging Face Transformers, rank-bm25, LoRA/SFT fine-tuning |
| **Vector & search** | Qdrant, Amazon S3 Vectors, hybrid dense + BM25 + RRF |
| **Messaging & infra** | Apache Pulsar, Docker, boto3, Elastic APM |
| **LLM integration** | LiteLLM, vLLM, WatsonX, G-Eval LLM-as-judge |
| **Evaluation** | YAML-driven test specs, React UI, Excel golden datasets |

---

## Impact Summary

- **Reliability:** Fixed shard-key and citation defects affecting production contract Q&A accuracy
- **Scalability:** Async re-architecture + S3 Vectors enable higher throughput and flexible per-client storage
- **Quality:** Dense retrieval recall 0.38 → 0.91; Graph RAG accuracy 70% → 81% on TTD golden set
- **Efficiency:** ~50% smaller embeddings; vectorizer benchmarks for 100K docs/day capacity planning
- **Release confidence:** Auto-Eval platform covering 12 ML pipelines with regression gates and self-service UI
- **Research → production:** Chunking, table retrieval, and LID benchmarks translated into actionable RAG and infra decisions

---

## Skills

LLM · RAG · NLP · MLOps · FastAPI · Python · Qdrant · AWS S3 Vectors · Apache Pulsar · Docker · Kubernetes · Benchmarking · PyTorch · Hugging Face · Prompt Engineering · Vector Databases · Elastic APM · Hybrid Search · BM25 · Fine-Tuning · Graph RAG · CI/CD · Async Python

---

</details>
