---
title: "Credit Risk Analysis (MS4610)"
description: "Explainable fraud risk modeling on AmEx customer data with a reproducible preprocessing pipeline and Random Forest classifier."
dateString: "2023"
draft: false
tags: ["Machine Learning", "Explainability", "Random Forest", "Fraud Detection", "Feature Engineering"]
weight: 403
showToc: false
---

### Objective

Build an explainable model to predict the probability of customer fraud using American Express customer data.

### Approach

- Built a preprocessing pipeline to convert **12 months** of customer history into a training-ready dataset.
- Trained and tuned a **Random Forest** classifier with validation-based selection.
- Focused on interpretability-friendly modeling and repeatable training/validation splits.

```mermaid
flowchart LR
  D[Raw 12-month customer data] --> P[Preprocess + Feature Engineering]
  P --> S[Train/Val Split]
  S --> M[Random Forest Training + Tuning]
  M --> E[Validation Metrics]
  E --> R[Model + Insights]
```

### Results

- Achieved **0.7822 validation ROC-AUC** with tuned Random Forest hyperparameters.

