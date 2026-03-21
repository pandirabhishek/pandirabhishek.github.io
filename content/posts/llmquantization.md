---
title: "Faster and Cheaper LLMs: Quantization of LLMS"
description: "Model Compression using quantization"
dateString: Oct 2025
draft: false
tags: ["ML", "AI", "Python", "quantization", "gptq","awq", "llama3"]
weight: 107
---

Continuing the journey of making LLMs more efficient, this post tackles one of the most impactful techniques in the toolbox — **quantization**. In the [previous post on prompt compression](/posts/prompt-compression/), we looked at how to shrink the *input* to a model. Quantization takes a different angle: it shrinks the *model itself*. By representing model weights (and sometimes activations) in lower-precision numerical formats, we can dramatically reduce memory footprint and speed up inference — often with surprisingly little loss in quality.

---

## Introduction

A typical large language model stores its billions of parameters in **FP16** (16-bit floating point) or **BF16** (Brain Float 16). A 70-billion parameter model, for instance, requires roughly **140 GB** of GPU memory just to load the weights — far beyond what a single consumer GPU can handle. This creates real barriers:

- **Hardware costs:** Multi-GPU setups or cloud A100/H100 instances are expensive. Fitting a model on fewer GPUs directly cuts infrastructure spend.
- **Inference latency:** Memory bandwidth is the bottleneck for autoregressive text generation; larger models transfer more data per token.
- **Deployment reach:** Even a single A100 (80 GB) cannot host a full-precision 70B model. Quantization is what makes single-GPU deployment possible.

Quantization addresses all three by converting high-precision weights into lower-bit representations — **INT8**, **INT4**, or even lower. The core idea is simple: if a weight stored in 16 bits can be adequately represented in 4 bits, you save 4x memory and move 4x less data per inference step. The challenge lies in doing this *without destroying the model's capabilities*.

## How Does Quantization Work?

At a high level, quantization maps continuous floating-point values to a discrete set of lower-precision values. Consider a tensor of FP16 weights. To quantize them to INT8:

1. **Find the range** — identify the minimum and maximum values in the tensor (or a sub-group of it).
2. **Compute a scale and zero-point** — these define the linear mapping from the floating-point range to the integer range (0–255 for unsigned INT8).
3. **Round each weight** to the nearest integer in the target range.
4. **During inference**, the quantized weights are dequantized on-the-fly (multiplied by the scale and shifted by the zero-point) to approximate the original values before computation.

The quality of quantization depends on *how* you choose these groups, scales, and which weights you protect. This is where different methods diverge.

### Granularity Matters

- **Per-tensor quantization:** One scale and zero-point for the entire weight matrix. Fast but coarse — outlier values stretch the range, reducing precision for the majority of weights.
- **Per-channel quantization:** Separate parameters for each output channel. Better accuracy, modest overhead.
- **Group quantization:** The weight matrix is divided into small groups (e.g., 128 elements), each with its own scale. This is the sweet spot used by most modern methods like **GPTQ** and **AWQ**, offering near-lossless INT4 quantization.

## Two Paradigms: PTQ vs QAT

There are two broad approaches to quantizing a model:

### Post-Training Quantization (PTQ)

PTQ quantizes a **pre-trained model** without any further training. You take the final weights, calibrate quantization parameters using a small calibration dataset, and produce a compressed model. This is by far the most popular approach for LLMs because:

- It requires **no GPU-intensive retraining**.
- It can be applied to **any pre-trained checkpoint**.
- Calibration takes minutes to hours, not days.

Most methods discussed below (GPTQ, AWQ, bitsandbytes) fall into this category.

### Quantization-Aware Training (QAT)

QAT simulates quantization **during training** by inserting fake quantization operations into the forward pass. The model learns to be robust to quantization noise. This typically yields better accuracy at very low bit-widths but comes at the cost of a full training run (or at least extensive fine-tuning). For LLMs with billions of parameters, QAT is often impractical, though recent methods like **QLoRA**[1](https://arxiv.org/abs/2305.14314) blend quantization with parameter-efficient fine-tuning to make it feasible.

## Key Quantization Methods

### GPTQ

**GPTQ**[2](https://arxiv.org/abs/2210.17323) is one of the most widely adopted PTQ methods for LLMs. It builds on the Optimal Brain Quantization (OBQ) framework and introduces critical optimizations that make it scalable to models with hundreds of billions of parameters.

**How it works:**

- GPTQ processes one layer at a time, quantizing weights column by column.
- For each column, it uses second-order (Hessian) information — computed from a small calibration set — to find the quantization assignment that minimizes the overall layer output error.
- After quantizing each column, the **remaining unquantized weights are adjusted** to compensate for the quantization error (this is the key insight from OBQ).
- The method supports **group quantization** (e.g., group size 128) for INT4, achieving excellent accuracy.

**Practical strengths:**

- Can quantize a 175B parameter model in approximately **4 GPU hours**.
- Achieves **near-lossless INT4** quantization on most LLMs.
- Well supported by libraries like **AutoGPTQ** and integrated into Hugging Face Transformers.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="c4",
    desc_act=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=gptq_config,
    device_map="auto",
)
```

### AWQ (Activation-Aware Weight Quantization)

**AWQ**[3](https://arxiv.org/abs/2306.00978) takes a different philosophy. Instead of using Hessian information to optimally round each weight, AWQ observes that **not all weights are equally important** — a small fraction of weights (roughly 1%) are critical because they correspond to large activation magnitudes. Quantizing these "salient" weights carelessly causes disproportionate accuracy loss.

**How it works:**

- AWQ identifies salient weight channels by examining **activation distributions** on a calibration set.
- Instead of keeping salient weights in higher precision (mixed-precision), AWQ applies **per-channel scaling** — it multiplies salient channels by a scale factor before quantization to reduce their relative quantization error, then compensates during inference.
- This approach is **hardware-friendly** because all weights remain in the same bit-width (INT4), avoiding the overhead of mixed-precision kernels.

**Practical strengths:**

- Consistently outperforms GPTQ on **instruction-tuned** and **multi-modal** models.
- Faster quantization than GPTQ (no Hessian computation).
- Excellent support through the **autoawq** library and Hugging Face integration.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Meta-Llama-3-8B"
quant_path = "llama3-8b-awq"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### bitsandbytes (LLM.int8() and QLoRA)

The **bitsandbytes**[4](https://arxiv.org/abs/2208.07339) library introduced two influential techniques:

- **LLM.int8():** A mixed-precision decomposition that identifies outlier features in activations (which can be 100x larger than typical values) and processes them in FP16 while quantizing the rest to INT8. This enables **zero-degradation INT8 inference** for models up to 175B parameters.

- **NF4 (4-bit NormalFloat):** Used in **QLoRA**[1](https://arxiv.org/abs/2305.14314), NF4 is an information-theoretically optimal data type for normally distributed weights. Combined with **double quantization** (quantizing the quantization constants themselves), it enables loading a 65B model on a single 48GB GPU for fine-tuning with LoRA adapters.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### Serving with vLLM

While GPTQ, AWQ, and bitsandbytes handle the *quantization* itself, the **serving engine** you use on top matters enormously for throughput. **vLLM**[5](https://github.com/vllm-project/vllm) is the de facto standard for high-throughput GPU inference and has first-class support for all three quantization formats:

- **PagedAttention** — manages KV cache like virtual memory pages, eliminating fragmentation and enabling higher batch sizes.
- **Continuous batching** — new requests are inserted into running batches without waiting for previous ones to finish, maximizing GPU utilization.
- **Fused CUDA kernels** — AWQ and GPTQ models benefit from optimized kernels that pack dequantization into the GEMM operation, which is a major reason they outperform FP16 in tokens/second.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Llama-3-8B-AWQ",
    quantization="awq",
    dtype="float16",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

outputs = llm.generate(
    ["Explain quantization in simple terms"],
    SamplingParams(max_tokens=256, temperature=0.7),
)
```

When deploying quantized models for production serving on NVIDIA GPUs, vLLM (or TensorRT-LLM for maximum performance) should be your default choice over vanilla Transformers inference.

## Choosing the Right Method

The landscape of GPU quantization methods can feel overwhelming, so here's a practical framework for choosing:

---

### Comparison Table

| **Method**     | **Bit-Width** | **Quantization Time** | **Inference Speed** | **Quality**    | **Best For**                         |
| -------------- | ------------- | --------------------- | ------------------- | -------------- | ------------------------------------ |
| GPTQ           | 4-bit         | Slow (Hessian)        | Fastest (fused CUDA)| Excellent      | Production GPU serving               |
| AWQ            | 4-bit         | Fast                  | Very Fast (fused CUDA)| Excellent    | Instruction-tuned / multimodal models|
| bitsandbytes   | 4/8-bit       | Instant (on-load)     | Moderate            | Very Good      | Fine-tuning (QLoRA), prototyping     |

---

### Throughput Benchmarks

To make the comparison concrete, here are real benchmark numbers for **Llama-3.1-8B-Instruct** on an **NVIDIA RTX 4090** (24 GB VRAM). GPTQ and AWQ were served via vLLM; bitsandbytes was served via Transformers (as vLLM does not natively support bitsandbytes for serving).

| **Metric**               | **FP16 (Baseline)** | **bitsandbytes NF4** | **AWQ 4-bit** | **GPTQ 4-bit** |
| ------------------------ | ------------------- | -------------------- | ------------- | --------------- |
| **Model Weight Memory**  | ~16 GB              | ~5 GB                | ~5 GB         | ~5 GB           |
| **VRAM Used (vLLM)**     | 22.6 GB             | 7.7 GB               | 22.7 GB       | 22.7 GB         |
| **Tokens/sec**           | 339.6               | 42.4                 | 579.1         | 598.7           |
| **Speed vs FP16**        | —                   | -87.5%               | +70.5%        | +76.3%          |
| **Load Time**            | 16.1s               | 6.6s                 | 10.4s         | 21.0s           |

*Benchmark source: [ermolushka.github.io](https://ermolushka.github.io/posts/vllm-benchmark-4090/)*

A few things stand out:

- **GPTQ and AWQ are *faster* than FP16**, not just smaller. This is because their fused CUDA kernels pack dequantization into the matrix multiply, reducing memory-bandwidth pressure. On an RTX 4090, GPTQ delivers **598.7 tokens/sec** — a **76% speedup** over FP16.

- **bitsandbytes is much slower for inference** (42.4 t/s) because it was designed for **fine-tuning convenience**, not serving throughput. It lacks the fused kernels that GPTQ/AWQ provide. However, it uses only **7.7 GB VRAM**, freeing memory for longer contexts or batch processing.

- **vLLM pre-allocates KV cache** to fill available GPU memory, which is why AWQ/GPTQ show ~22.7 GB total VRAM usage despite the model weights being only ~5 GB. The remaining ~17 GB is KV cache — meaning quantized models actually support **much longer contexts** or **larger batches** than FP16 at the same VRAM budget.

---

### Decision Guide

- **Deploying on GPU for production serving?** Use **GPTQ** or **AWQ** with **vLLM**. Both produce INT4 models with fused CUDA kernels that are actually *faster* than FP16. GPTQ edges out on raw throughput; AWQ tends to preserve quality better on instruction-tuned and multimodal models.

- **Fine-tuning on limited hardware?** Use **bitsandbytes with QLoRA**. Load the base model in NF4 and train LoRA adapters — you can fine-tune a 70B model on a single A100.

- **Need maximum throughput at scale?** Consider **NVIDIA TensorRT-LLM** with FP8 quantization on H100/H200 GPUs. FP8 (available on Hopper and later architectures) halves memory vs FP16 while leveraging the Transformer Engine for near-lossless accuracy — no calibration dataset needed.

- **Need the absolute best quality?** Stick with **INT8** (either GPTQ 8-bit or LLM.int8()). The accuracy loss is negligible for virtually all tasks.

## Practical Tips

Here are some lessons learned from working with quantized models:

- **Calibration data matters.** For GPTQ and AWQ, the calibration dataset should be representative of your use case. Using C4 (a general web corpus) works well as a default, but domain-specific data can improve results.

- **Group size 128 is the sweet spot.** Smaller groups (e.g., 64) improve accuracy marginally but increase overhead. Larger groups (e.g., 256) save space but may hurt quality. Most practitioners default to 128.

- **Perplexity is a useful but imperfect metric.** Always evaluate quantized models on your **actual downstream task**. A small perplexity increase might not matter for classification but could degrade open-ended generation.

- **Quantized models can still be fine-tuned.** QLoRA demonstrated that 4-bit quantized models can be effectively fine-tuned with LoRA adapters, reaching performance comparable to full 16-bit fine-tuning on many benchmarks.

- **Watch out for outliers.** Some model architectures (especially older ones) have extreme activation outliers that make naive quantization fail. LLM.int8()'s mixed-precision decomposition was specifically designed to handle this.

## Challenges

While quantization has become remarkably effective, some open challenges remain:

- **Sub-4-bit quantization** (2-bit, 3-bit) still incurs noticeable quality degradation on complex reasoning tasks, though methods like **QuIP#**[6](https://arxiv.org/abs/2307.13304) and **AQLM**[7](https://arxiv.org/abs/2401.06118) are pushing the boundaries.

- **Quantizing activations** (not just weights) remains harder because activation distributions change with each input. Weight-only quantization is well-understood; weight-and-activation quantization (W4A4, W8A8) is still an active research area, with **SmoothQuant**[8](https://arxiv.org/abs/2211.10438) being a notable contribution.

- **Task-specific sensitivity** means that a quantized model performing well on perplexity benchmarks might struggle on tasks requiring precise numerical reasoning or code generation. Evaluation beyond perplexity is essential.

- **Tooling fragmentation:** GPTQ, AWQ, and bitsandbytes each have their own ecosystems, model formats, and kernel implementations. Serving engines like vLLM and TensorRT-LLM add another layer of choices. Efforts like Hugging Face's unified `quantization_config` API are helping, but the landscape remains fractured.

---

## Conclusion

Quantization has evolved from a niche optimization technique to a **cornerstone of practical LLM deployment on NVIDIA GPUs**. The ability to run a 70B-parameter model on a single GPU was unthinkable just a couple of years ago. Today, methods like **GPTQ**, **AWQ**, and **bitsandbytes**, paired with serving engines like **vLLM**, make it routine — and often *faster* than full-precision inference.

The key takeaways:

- **INT4 weight-only quantization** (via GPTQ or AWQ) is mature and production-ready. With fused CUDA kernels, quantized models deliver **up to 76% higher throughput** than FP16 on the same GPU.
- **bitsandbytes** excels at **memory efficiency** (66% VRAM reduction) and is the go-to for **QLoRA fine-tuning**, even though its inference speed is lower.
- **QLoRA** bridges quantization and fine-tuning, making it possible to customize large models on a single GPU.
- The frontier is pushing toward **FP8** on Hopper GPUs, **sub-4-bit** methods, and **weight-activation co-quantization**, promising even greater compression without sacrificing quality.

As the next step in this series on efficient LLMs, future posts will explore **knowledge distillation** and **speculative decoding** — techniques that complement quantization to make LLM inference even faster and cheaper.

## References
1. [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314)
2. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
3. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
4. [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
5. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://github.com/vllm-project/vllm)
6. [Memory Optimization Deep Dive: Running 8B Models on a Single 4090 using vLLM — Benchmarks](https://ermolushka.github.io/posts/vllm-benchmark-4090/)
7. [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2307.13304)
8. [AQLM: Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118)
9. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
