---
title: "Real-time Regional Interest Analysis"
description: "Python tool to analyze a region’s real-time Twitter feed with sentiment, word cloud, and tweet summarization."
dateString: "2023"
draft: false
tags: ["NLP", "Transformers", "Sentiment Analysis", "APIs", "Hugging Face", "Python"]
weight: 404
showToc: false
---

### Objective

Create a Python tool that ingests a region’s real-time Twitter feed and produces actionable summaries of what people are talking about.

### What I built

- Twitter ingestion via API + region filters
- Text cleaning + preprocessing pipeline
- Transformer-based sentiment model trained on Kaggle data
- Outputs: sentiment dashboard, word cloud, and “top 10 tweets” summarization using Hugging Face tooling

### Pipeline

```mermaid
flowchart LR
  T[Twitter API: region feed] --> C[Clean + Normalize Text]
  C --> S[Transformer Sentiment Model]
  S --> V[Visuals: sentiment + word cloud]
  C --> SUM[Top-10 Tweet Summarization]
  V --> OUT[Report / Dashboard]
  SUM --> OUT
```

### Results

- Achieved **0.96 validation ROC-AUC** on the sentiment model (Kaggle-based dataset).

