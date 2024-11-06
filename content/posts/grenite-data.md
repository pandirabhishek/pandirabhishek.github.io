---
title: "Part 1 Granite 3.0:  The Data Powering Granite 3.0"
description: "Unpacking Granite 3.0’s Data Strategy: The Power of Curated, Public, and Synthetic Datasets in AI Model Development"
dateString: Oct 2024
draft: false
tags: ["ML", "LLM", "data", "LLAMA3", "Instruction", "Finetuning"]
weight: 107
---

As large language models (LLMs) advance, the quality and diversity of their training data remain central to their performance. With Granite 3.0, developers have pushed the envelope by crafting a data ecosystem that includes curated web data, public datasets, and extensive synthetic data to meet a broad range of needs. In this post, we’ll explore Granite 3.0's unique data strategy, its filtering mechanisms, and how these choices affect the model's capabilities.

---

## The Foundations: Curated Web and Public Datasets

**Granite 3.0's training data taps into a vast collection of curated web and public sources, carefully selected to enhance its knowledge base and performance across multiple domains.**

### Curated Web Data
Granite 3.0 relies heavily on curated text and code from various sources, spanning academic articles, financial reports, legal documents, and biomedical papers. To ensure quality, the data is preprocessed rigorously with:

- **Text Extraction and Language Identification** – To capture relevant content and ensure accuracy.
- **Filtering for Permissive Licenses** – Especially critical for code, where licensing restrictions are common.
- **IBM’s URL Blocking** – Excludes data from websites deemed untrustworthy or inappropriate, promoting ethical AI.

### Public Datasets
Granite 3.0 incorporates publicly available datasets with permissive licenses, each selected to broaden the model’s reach across distinct domains:

- **Code Repositories** – GitHub Code Clean, StarCoderdata, Code Contests.
- **Web Data** – FineWeb, DCLM-Baseline for diverse internet text.
- **Multilingual Corpora** – MADLAD-12 for language versatility.
- **Instructional Data** – Code Instructions Alpaca, Glaive Function Calling V2, Self-OSS-Instruct-SC2 for improved contextual understanding.

Together, these sources provide a strong foundation, equipping Granite 3.0 to understand and perform tasks across a variety of fields.

## The Rise of Synthetic Data: Tackling Data Gaps and Specialization

A key advancement in Granite 3.0’s data approach is its reliance on **synthetic data generation (SDG)** to address limitations in publicly available datasets. Synthetic data allows for targeted enhancements in specialized areas, minimizing the need for costly human annotation.

### Synthetic Data Applications
Granite 3.0’s synthetic data efforts address a range of model capabilities:

- **Instruction Following** – Evol-Instruct and MagPie create complex instruction-response pairs.
- **Coding Tasks** – Synthetic data aids in code-related functions like debugging, generating docstrings, and unit testing.
- **Reasoning and Retrieval** – Synthetic data builds the model’s capacity for complex reasoning tasks using algorithmic and knowledge-based data generation.
- **Tool Use** – Designed for complex tool interactions, covering multi-turn and nested calls.
- **Cybersecurity** – A two-step SDG process generates high-quality security instructions and diversifies the dataset.
- **Multilingual and Safety Training** – Focused synthetic data enhances translation quality and model alignment with safe behaviors.

**Quality Control** – Granite 3.0 filters synthetic data using multiple techniques to ensure clarity and relevance. This includes discarding short, unclear, or duplicate instructions and employing LLMs as quality judges.


## Fine-Tuning with Data Mixtures and Hyperparameters

Training Granite 3.0 involves strategic mixing of pre-training and fine-tuning data and careful hyperparameter tuning to optimize performance:

- **Pre-Training Mixture** – A two-stage process optimizes robustness across domains using a mixture search method inspired by distributionally robust language modeling.
- **Post-Training Mixture** – A combination of public and synthetic data enhances the model's ability in specific areas, with a focus on datasets proven to improve results in Granite’s internal benchmarks.

## Rigorous Filtering: Ensuring Quality, Safety, and Relevance

High-quality data is critical for training responsible, high-performing AI systems. Granite 3.0’s approach includes a rigorous filtering process for both open-source and synthetic data:

### Filtering Open-Source Data

For open-source data, the sources describe a multi-faceted filtering process, encompassing several key aspects:

*   **Licensing:** The sources emphasize the importance of using permissively licensed data to ensure the models can be used for both research and commercial purposes. For code data, they describe a process of annotating each code file with license information using GitHub APIs and retaining only those files with permissive licenses.

*   **URL Blocking List:**  The sources mention an IBM URL blocking list used to exclude data from sources deemed unsuitable for training. This list likely includes websites or domains known for containing potentially harmful or inappropriate content.

*   **Hateful, Abusive, and Profane (HAP) Content:** To mitigate the risk of generating harmful language, the sources detail a process for filtering HAP content. For text data, this involves computing HAP scores at the sentence level using an IBM-trained HAP detector and filtering out documents exceeding a specific threshold. For code, a dictionary of HAP keywords is used to annotate documents, and those exceeding a threshold based on distributional analysis and manual inspection are removed.

*   **Malware Detection:** For code data, the sources mention using ClamAV to scan and remove instances of malware, further ensuring the safety and integrity of the training data.

*   **Document Quality Filtering:**  The sources describe a two-pronged approach to filtering low-quality documents:

    *   **Heuristics:** Rules based on the Gopher quality filtering criteria are applied to identify and remove documents with low linguistic value. These rules target characteristics like excessive bullet points, ellipsis lines, and symbol-to-word ratios.
    *   **Classifier-Based Filtering:**  A KenLM linear classifier, pre-trained on high-quality documents like Wikipedia articles, is used to assign perplexity scores to documents. Documents with low scores, indicating low similarity to the training corpus, are filtered out.

*   **Code-Specific Heuristics:**  In addition to general document quality filtering, the sources outline several heuristics specifically for filtering lower-quality code:

    *   **Minimum Alphabetic Characters:** Files with less than 25% alphabetic characters are removed.
    *   **XML Filtering:** Files where the string "<?xml version=" appears within the first 100 characters are filtered out.
    *   **HTML Filtering:** Only HTML files where visible text comprises at least 20% of the code and has a minimum length of 100 characters are kept.
    *   **JSON and YAML Filtering:** Only JSON and YAML files with a character count between 50 and 5000 are retained.

*   **Language Identification:** For web data, fasText is used to identify the dominant language of each document. The sources mention selecting documents primarily in English but also including high-quality documents from eleven other languages to improve the model's multilingual capabilities.

*   **Machine Translation Quality Filtering:** To enhance machine translation quality, the sources describe applying extensive filtering to parallel text datasets like ParaCrawl, WikiMatrix, and NLLB/CCMatrix. This filtering involves language-specific heuristics and model-based scoring.


### Filtering Synthetic Data
Synthetic data undergoes stringent quality checks:
*   **Length and Clarity:** Samples with very short instructions, unclear instructions, or duplicated samples are removed.
*   **LLM-as-Judge:**  A Mistral-7B-Instruct-v0.3 model is used as a judge to annotate and assess the quality and difficulty of instructions and responses.
*   **Minimum Neighbor Distance:**  Embedding-based methods are used to identify near-duplicate samples by calculating the minimum neighbor distance in the embedding space. This process is applied to both single-turn and multi-turn data by considering the full conversations.
*   **Multi-Turn Sample Annotation:** For multi-turn datasets, an LLM is used to assess the overall quality of the conversation between the user and the assistant, including all turns.



---

## Key Takeaways

Granite 3.0’s extensive filtering and diverse data strategy highlight several best practices in AI development:

- **Enhanced Model Performance** – Well-filtered data contributes to cleaner, more consistent training examples.
- **Data Safety and Responsibility** – Careful filtering mitigates risks associated with harmful content.
- **Generalization Across Domains** – With balanced data diversity, Granite 3.0 is adaptable to a wide array of tasks and knowledge areas.

Through its sophisticated approach to data curation and filtering, Granite 3.0 exemplifies the value of a well-rounded, ethical, and high-performance AI model. By balancing public and synthetic data with rigorous filtering, Granite 3.0 sets a new standard in the industry.

---

### Conclusion: Driving the Future of AI with Responsible Data Curation

Granite 3.0 showcases how a thoughtful approach to data, leveraging both real-world and synthetic sources, can drive model quality and specialization. As more organizations look to harness the potential of LLMs, the Granite 3.0 strategy serves as a powerful example of how curated, high-quality data can set the stage for success.
