# Week 1 — Deep Learning + Transformers Fundamentals

**Dates**: ____ → ____  
**Status**: ⬜ Not started | 🟡 In progress | ✅ Complete

---

## Goal

Explain transformers rigorously; understand training dynamics and inference cost (prefill vs decode, KV cache).

---

## Weekly deliverables

| Deliverable | Done | File / link |
|-------------|------|-------------|
| Cheatsheet: Transformer + KV cache | ⬜ | `notes/transformer-kv-cache.md` |
| Notes: Debugging training instability | ⬜ | `notes/training-debug-checklist.md` |
| Story #1: "Biggest impact" outline | ⬜ | |
| Mock: Explain KV cache (15m) | ⬜ | |
| Mock: 1 behavioral (15m) | ⬜ | |

---

## Daily schedule

| Day | Coding | Core study | Interview output |
|-----|--------|------------|------------------|
| **Mon** | Arrays/strings + hashing | Transformer overview: tokens→embeddings→blocks→logits | 1-page: Transformer block anatomy |
| **Tue** | Two pointers / sliding window | CE loss, backprop, AdamW, LR schedule, weight decay | Checklist: debug training collapse |
| **Wed** | Stack / queue | Attention math: Q,K,V; multi-head | 2-min script: attention + multi-head |
| **Thu** | Binary search | Positional encodings: absolute, relative, RoPE | Cheatsheet: positional encoding |
| **Fri** | BFS/DFS | Prefill vs decode; KV cache memory/latency | 1-page: KV cache + latency tradeoffs |
| **Sat** | Timed set (2 mediums) | Mixed precision, norm, gradient clipping | Polish Story #1 |
| **Sun** | Review weak DSA | Consolidate Transformer + KV cache notes | Finalize cheatsheets |


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

**Total hours**: ____

---

## Self-test (target ≥4/5 on all)

| # | Question | Score |
|---|----------|-------|
| 1 | What are Q, K, V and why scale by √d? | |
| 2 | Encoder vs decoder vs decoder-only transformers? | |
| 3 | Why causal masking in GPT-style models? | |
| 4 | Where is LayerNorm applied in a transformer block? | |
| 5 | Adam vs AdamW — difference? | |
| 6 | What causes training loss NaN? How to debug? | |
| 7 | Prefill vs decode — compute vs memory bound? | |
| 8 | How does KV cache reduce generation cost? | |
| 9 | What is RoPE and why do modern LLMs use it? | |
| 10 | Mixed precision training — benefits and risks? | |

---

## Mock log

| Date | Type | Topic | Score | Improve |
|------|------|-------|-------|---------|
| | Technical | KV cache | | |
| | Behavioral | Biggest impact | | |

---

## Week retrospective

**Went well:**

**Revisit:**

**Carry to Week 2:**

---

## Resources


## Day 1 (Mon) — Transformer overview + DSA (arrays/strings, hashing)

### Core study
| Resource | Type | Why it matters |
|----------|------|----------------|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Blog | Best visual intro to tokens → embeddings → encoder/decoder blocks |
| [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) | Code + notes | Walk through real PyTorch implementation line-by-line |
| [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | Paper | Original Transformer paper — skim §3 (Architecture) |
| [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) (Karpathy) | Video (~2h) | End-to-end decoder-only transformer intuition (highly interview-relevant) |
| [Hugging Face: Transformer models course — Ch. 1](https://huggingface.co/learn/nlp-course/chapter1/1) | Course | Modern framing of transformer families (encoder/decoder/decoder-only) |

### Coding (DSA)
| Resource | Type |
|----------|------|
| [NeetCode 150 — Arrays & Hashing](https://neetcode.io/practice) | Practice |
| [LeetCode Explore — Hash Table](https://leetcode.com/explore/learn/card/hash-table/) | Guided track |
| [Blind 75 list](https://neetcode.io/practice?tab=blind75) | Curated set |

**Suggested problems**: Two Sum, Contains Duplicate, Valid Anagram, Group Anagrams, Top K Frequent Elements.

---

## Day 2 (Tue) — Training basics (CE loss, backprop, AdamW, LR) + DSA (two pointers)

### Core study
| Resource | Type | Why it matters |
|----------|------|----------------|
| [3Blue1Brown — Neural networks (playlist)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6Rfk_7txYBOyY-AztqUx) | Video | Intuition for gradients, backprop, learning |
| [CS231n Notes — Backpropagation](https://cs231n.github.io/optimization-2/) | Notes | Rigorous but readable backprop + gradient checks |
| [Deep Learning Book — Ch. 6 (Deep Feedforward Networks)](https://www.deeplearningbook.org/contents/mlp.html) | Book | Loss functions, optimization foundations |
| [AdamW paper](https://arxiv.org/abs/1711.05101) | Paper | Why AdamW ≠ Adam + weight decay (common interview question) |
| [PyTorch — A Recipe for Training Neural Networks](https://karpathy.medium.com/a-recipe-for-training-neural-networks-5bcea71d7a73) | Blog | Practical training debugging checklist (Karpathy) |
| [fast.ai — Practical Deep Learning (Lesson 1–2)](https://course.fast.ai/Lessons/lesson1.html) | Course | Training loops, overfitting, learning rate in practice |

### Coding (DSA)
| Resource | Type |
|----------|------|
| [NeetCode — Two Pointers](https://neetcode.io/roadmap) | Practice |

**Suggested problems**: Valid Palindrome, Two Sum II, 3Sum, Container With Most Water, Trapping Rain Water.

---

## Day 3 (Wed) — Attention math (Q, K, V, multi-head) + DSA (stack/queue)

### Core study
| Resource | Type | Why it matters |
|----------|------|----------------|
| [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) | Blog | Decoder-only attention + causal mask explained visually |
| [Lilian Weng — Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) | Blog | Attention variants history + math intuition |
| [Stanford CS224N — Lecture 3 (Transformers)](http://web.stanford.edu/class/cs224n/) | Lecture | Academic depth; check current semester for "Transformers" lecture |
| [einsum attention from scratch (mini tutorial)](https://e2eml.school/transformers.html) | Interactive | Build attention with explicit matrix shapes |
| [Multi-Head Attention — HF docs](https://huggingface.co/docs/transformers/en/attention) | Docs | How frameworks implement it in practice |

### Coding (DSA)
| Resource | Type |
|----------|------|
| [NeetCode — Stack](https://neetcode.io/roadmap) | Practice |

**Suggested problems**: Valid Parentheses, Min Stack, Evaluate RPN, Daily Temperatures, Largest Rectangle in Histogram.

---

## Day 4 (Thu) — Positional encodings (absolute, relative, RoPE) + DSA (binary search)

### Core study
| Resource | Type | Why it matters |
|----------|------|----------------|
| [RoFormer / RoPE paper](https://arxiv.org/abs/2104.09864) | Paper | Rotary embeddings used in LLaMA, Mistral, etc. |
| [EleutherAI — Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) | Blog | Clear RoPE intuition without heavy math |
| [ALiBi paper](https://arxiv.org/abs/2108.12409) | Paper | Alternative positional approach for length extrapolation |
| [Transformer Architecture: Positional Encoding variants](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) | Blog | Compares sinusoidal, learned, relative, RoPE |
| [LLaMA explained (pos encoding section)](https://magazine.sebastianraschka.com/p/understanding-llama-adaptations) | Blog | How modern LLMs adapt the base transformer |

### Coding (DSA)
| Resource | Type |
|----------|------|
| [NeetCode — Binary Search](https://neetcode.io/roadmap) | Practice |

**Suggested problems**: Binary Search, Search Insert Position, Find Minimum in Rotated Sorted Array, Koko Eating Bananas, Median of Two Sorted Arrays (stretch).

---

## Day 5 (Fri) — Inference cost: prefill vs decode, KV cache + DSA (BFS/DFS)

### Core study
| Resource | Type | Why it matters |
|----------|------|----------------|
| [vLLM paper (PagedAttention)](https://arxiv.org/abs/2309.06180) | Paper | KV cache memory management — production-grade mental model |
| [Hugging Face — KV cache concept](https://huggingface.co/docs/transformers/en/kv_cache) | Docs | What gets cached and why decode is expensive |
| [Tri Dao — FlashAttention blog](https://tridao.me/publications/flash2/flash2.pdf) | Paper/blog | Why attention is memory-bound; kernel-level intuition |
| [Chip Huyen — Efficient Inference for LLMs](https://huyenchip.com/2023/10/10/llm-inference.html) | Blog | Prefill vs decode, batching, latency vs throughput |
| [Lilian Weng — Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | Blog | Comprehensive inference optimization survey |
| [Your blog: LLM Quantization](https://pandirabhishek.github.io/posts/llmquantization/) | Your post | Connects to Week 6 but good early read for cost intuition |

### Coding (DSA)
| Resource | Type |
|----------|------|
| [NeetCode — Graphs (BFS/DFS)](https://neetcode.io/roadmap) | Practice |

**Suggested problems**: Number of Islands, Clone Graph, Pacific Atlantic Water Flow, Course Schedule, Rotting Oranges.

---

## Day 6 (Sat) — Mixed precision, normalization, gradient clipping + DSA timed set

### Core study
| Resource | Type | Why it matters |
|----------|------|----------------|
| [Mixed Precision Training (Micikevicius et al.)](https://arxiv.org/abs/1710.03740) | Paper | FP16/BF16 training fundamentals |
| [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html) | Docs | How autocast + GradScaler work in practice |
| [Layer Normalization paper](https://arxiv.org/abs/1607.06450) | Paper | Why transformers use LayerNorm not BatchNorm |
| [Batch Norm vs Layer Norm — explained](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739) | Blog | Quick comparison for interviews |
| [Gradient clipping in practice](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) | Docs | Stabilizing RNN/Transformer training |

### Coding (DSA)
| Resource | Type |
|----------|------|
| [LeetCode — Top Interview 150 (timed)](https://leetcode.com/studyplan/top-interview-150/) | Timed practice |

**Goal**: 2 mediums in 60–75 min, explain complexity out loud after each.

---

## Day 7 (Sun) — Consolidation + review

### Consolidation (no new topics)
| Resource | Type | Action |
|----------|------|--------|
| Your Week 1 notes | Self | Merge into 2-page cheatsheet: Transformer + KV cache |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Re-read | Explain each diagram without looking |
| [Karpathy GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Re-watch | Skip to attention + training loop sections |
| [A Recipe for Training Neural Networks](https://karpathy.medium.com/a-recipe-for-training-neural-networks-5bcea71d7a73) | Re-read | Turn into your "debug training instability" checklist |

### Coding (DSA)
- Re-solve 2 problems you got wrong earlier in the week (no hints).

### Mock prompts (15 min each)
1. "Explain self-attention to a senior engineer in 2 minutes."
2. "What is KV cache and why does long context hurt latency?"
3. "Walk me through what happens during prefill vs decode."

---

## Week 1 "must-know" interview questions (self-test)

After finishing the week, you should be able to answer without notes:

1. What are Q, K, V and why do we scale by √d?
2. What is the difference between encoder, decoder, and decoder-only transformers?
3. Why causal masking in GPT-style models?
4. What is LayerNorm and where is it applied in a transformer block?
5. Adam vs AdamW — what's the difference?
6. What causes training loss to NaN? How do you debug?
7. Prefill vs decode — which is more compute-bound vs memory-bound?
8. How does KV cache reduce cost during generation?
9. What is RoPE and why do modern LLMs use it?
10. What is mixed precision training and what can go wrong?

---

## Optional deep dives (if you have extra time)

| Topic | Resource |
|-------|----------|
| FlashAttention | [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691) |
| GPT architecture | [GPT-3 paper](https://arxiv.org/abs/2005.14165) — skim architecture section |
| Training at scale | [Chinchilla paper](https://arxiv.org/abs/2203.15556) — compute-optimal training (useful for scientist interviews) |
| Hands-on coding | [nanoGPT](https://github.com/karpathy/nanoGPT) — train a small GPT yourself |

---

## Suggested daily schedule (example)

| Time | Activity |
|------|----------|
| 0:00–1:15 | DSA (problems + explain solutions) |
| 1:15–2:45 | Core study (1 primary resource + notes) |
| 2:45–3:15 | Interview output (cheatsheet / 2-min script / checklist) |
| 3:15–3:30 | Flashcards (10 cards) |

---

## Link to your portfolio content (connect theory → your experience)

| Your work | Week 1 concept it maps to |
|-----------|---------------------------|
| Sprinklr inference optimization | KV cache, quantization, prefill/decode, batching |
| Sirion async ml-context-service | Throughput vs latency, concurrency (preview Week 7) |
| Your quantization blog post | Mixed precision, memory reduction |
| Fine-tuning LLaMA/Mistral | Transformer blocks, CE loss, AdamW, training stability |
