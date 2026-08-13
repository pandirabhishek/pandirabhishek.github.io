# Week 2 — NLP Fundamentals + Evaluation

**Dates**: ____ → ____  
**Status**: ⬜ Not started | 🟡 In progress | ✅ Complete

---

## Goal

Speak like an NLP engineer/scientist: tokenization, metrics, data quality, embeddings.

---

## Weekly deliverables

| Deliverable | Done | Notes |
|-------------|------|-------|
| Tokenization pitfalls list (10 bullets, Indic) | ⬜ | |
| Evaluation template (classification + slices) | ⬜ | |
| NLP fundamentals packet (2 pages) | ⬜ | |
| Story #2: "Failure + recovery" | ⬜ | |
| Mock: NLP metrics (15m) | ⬜ | |
| Mock: System design mini (20m) | ⬜ | |

---

## Daily schedule

| Day | Coding | Core study | Interview output |
|-----|--------|------------|------------------|
| **Mon** | Intervals | BPE vs Unigram; byte-level tokenization | 10 bullets: tokenization pitfalls |
| **Tue** | Heaps | AUROC vs AUPRC; calibration; imbalance | Eval template + slices |
| **Wed** | Graph BFS/DFS | NLP tasks; eval limits for generation | Rubric: groundedness/correctness |
| **Thu** | Recursion/backtracking | Dedup, leakage, labeling noise | Dataset QA + leakage checklist |
| **Fri** | Binary search variants | Static vs contextual embeddings | Embeddings cheatsheet |
| **Sat** | Timed set (2 mediums) | End-to-end NLP eval plan | Polish Story #2 |
| **Sun** | Re-solve 2 problems | Consolidate Week 2 notes | Finalize NLP packet |


---

## Daily log

| Day | Coding ✅ | Study ✅ | Output ✅ | Hours | Conf (1–5) | Notes |
|-----|-----------|----------|-----------|-------|------------|-------|
| Mon | ⬜ | ⬜ | ⬜ | | | |
| Tue | ⬜ | ⬜ | ⬜ | | | |
| Wed | ⬜ | ⬜ | ⬜ | | | |
| Thu | ⬜ | ⬜ | ⬜ | | | |
| Fri | ⬜ | ⬜ | ⬜ | | | |
| Sat | ⬜ | ⬜ | ⬜ | | | |
| Sun | ⬜ | ⬜ | ⬜ | | | |

---

## Self-test

| # | Question | Score |
|---|----------|-------|
| 1 | BPE vs Unigram — when to use which? | |
| 2 | AUROC vs AUPRC — when is AUPRC better? | |
| 3 | What is calibration and how to measure it? | |
| 4 | Common dataset leakage patterns in NLP? | |
| 5 | Why contextual embeddings beat static for most tasks? | |
| 6 | How does tokenization affect Indic language LLM cost? | |
| 7 | Precision vs recall tradeoff in moderation? | |
| 8 | How to evaluate summarization without ROUGE-only? | |

---

## Mock log

| Date | Type | Topic | Score | Improve |
|------|------|-------|-------|---------|

---

## Week retrospective

**Went well:**

**Revisit:**

**Carry to Week 3:**

---

## Resources


## Day 1 (Mon) — Tokenization

| Resource | Type |
|----------|------|
| [Your blog: Tokenization for Indian Languages](https://pandirabhishek.github.io/posts/tokenization/) | Your post |
| [Hugging Face Tokenizers docs](https://huggingface.co/docs/tokenizers/index) | Docs |
| [SentencePiece paper](https://arxiv.org/abs/1808.06226) | Paper |
| [BPE original paper](https://arxiv.org/abs/1508.07909) | Paper |
| [HF NLP Course — Ch. 6 Tokenizers](https://huggingface.co/learn/nlp-course/chapter6/1) | Course |

**DSA**: [NeetCode — Intervals](https://neetcode.io/roadmap) — Merge Intervals, Insert Interval, Non-overlapping Intervals

---

## Day 2 (Tue) — Classification metrics

| Resource | Type |
|----------|------|
| [Google ML Crash Course — Classification](https://developers.google.com/machine-learning/crash-course/classification) | Course |
| [ROC vs PR curves (scikit-learn)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) | Docs |
| [Calibration in ML (Niculescu-Mizil & Caruana)](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf) | Paper |
| [Imbalanced classification guide](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/) | Blog |

**DSA**: Heaps — Kth Largest, Top K Frequent, Find Median from Data Stream

---

## Day 3 (Wed) — NLP tasks + eval limits

| Resource | Type |
|----------|------|
| [Stanford CS224N](http://web.stanford.edu/class/cs224n/) | Lectures |
| [Hugging Face NLP Course — Ch. 2–4](https://huggingface.co/learn/nlp-course/chapter2/1) | Course |
| [BLEU/ROUGE limitations](https://huggingface.co/spaces/evaluate-metric/rouge) | Docs |
| [RAGAS eval framework docs](https://docs.ragas.io/) | Docs |

**DSA**: Graph BFS/DFS review

---

## Day 4 (Thu) — Data pipelines + leakage

| Resource | Type |
|----------|------|
| [Leakage in ML (Kaggle guide)](https://www.kaggle.com/code/alexisbcook/data-leakage) | Tutorial |
| [Cleanlab — label noise](https://cleanlab.ai/blog/learn/cleanlab-2/) | Blog |
| [Deduplication for LLM training (Dedup blog)](https://huggingface.co/blog/dedup) | Blog |

**DSA**: Recursion/backtracking — Subsets, Combination Sum, Permutations

---

## Day 5 (Fri) — Embeddings

| Resource | Type |
|----------|------|
| [Word2Vec paper](https://arxiv.org/abs/1301.3781) | Paper |
| [SBERT paper](https://arxiv.org/abs/1908.10084) | Paper |
| [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | Benchmark |
| [HF — Sentence Transformers](https://www.sbert.net/) | Docs |

**DSA**: Binary search variants

---

## Day 6–7 — Consolidation

- Re-read your tokenization blog
- Build eval template for moderation (precision/recall/F1 + slices by language/topic)
- Mock: "How would you evaluate a sentiment model in production?"

