<<<<<<< HEAD
# ChatAcredita PRO

[![Paper](https://img.shields.io/badge/Paper-ESWA%202026-blue)](https://www.sciencedirect.com/journal/expert-systems-with-applications)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-raulgdp-yellow)](https://huggingface.co/raulgdp)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

**RAG-based QA system for Colombian university accreditation (CNA / Decreto 1330 de 2019)**

> Gutiérrez de Piñerez et al. *"ChatAcredita PRO: Fine-tuned LLMs with RAG-VSD Filtering for Colombian University Accreditation QA"* — Expert Systems with Applications, 2026.

---

## Models

| Model | Base | Parameters | HuggingFace |
|-------|------|-----------|-------------|
| **ChatAcredita PRO** ⭐ | Qwen3-14B | 14B | [raulgdp/qwen3-14b-acredita](https://huggingface.co/raulgdp/qwen3-14b-acredita) |
| DeepSeek-14B-Acredita | DeepSeek-R1-14B | 14B | [raulgdp/deepseek14b-acredita](https://huggingface.co/raulgdp/deepseek14b-acredita) |
| Qwen2.5-7B-Acredita | Qwen2.5-7B | 7B | [raulgdp/qwen2.5-7b-acredita-cna-col](https://huggingface.co/raulgdp/qwen2.5-7b-acredita-cna-col) |

All models are fine-tuned with QLoRA (r=32, α=64) on the RAG-VSD filtered dataset of 9,847 CNA instruction pairs.

---

## Repository Structure

```
chatacredita-pro/
├── data/
│   ├── benchmark_142q.jsonl      ← 142-question CNA evaluation benchmark
│   └── README.md
├── evaluation/
│   ├── ragas_multijuez.py        ← 5-judge RAGAS ensemble evaluation
│   ├── geval_acredita.py         ← G-Eval (GPT-4o Mini judge, 4 dimensions)
│   └── benchmark_3modelos.py     ← Generate answers for all 3 models
├── rag_vsd/
│   ├── 02_filter_rag.py          ← RAG-VSD filtering pipeline
│   ├── config.py                 ← Shared configuration & thresholds
│   └── utils/
│       └── qdrant_client.py      ← RAGVerifier (τ_overlap, τ_query, τ_tech)
└── training/
    └── training_config.yaml      ← QLoRA hyperparameters
```

---

## RAG Pipeline

```
Query
  │
  ▼
BGE-M3 (dense) + BM25 (sparse)
  │   Reciprocal Rank Fusion (RRF)
  ▼
Qdrant Cloud  ──► top-10 chunks
  │
BGE-Reranker-v2-m3
  │
  ▼
top-8 chunks → ChatAcredita PRO (Qwen3-14B-Acredita)
  │
  ▼
Answer
```

**Vector store:** Qdrant Cloud (us-east-1)  
**Embedding:** `BAAI/bge-m3` (1024-dim)  
**Reranker:** `BAAI/bge-reranker-v2-m3`  
**Corpus:** CNA documentation + EISC Universidad del Valle accreditation reports

---

## RAG-VSD Data Filtering

The RAG-VSD (RAG Verification by Semantic Distance) filter accepts a synthetic pair only when **all three** conditions hold:

| Threshold | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Lexical overlap | τ_overlap | 0.30 | ≥ 30 % of response tokens appear in retrieved chunks |
| Semantic similarity | τ_query | 0.55 | cosine(query, top-1 chunk) ≥ 0.55 |
| CNA term density | τ_tech | 0.40 | ≥ 40 % of response tokens are CNA technical terms |

**Result:** 9,847 / 15,000 pairs accepted (65.6%), +29.4 pp faithfulness gain vs. unfiltered.

---

## Evaluation Results

### RAGAS MJ-5 (5-judge ensemble via OpenRouter)

Judges: Gemma 3 27B-IT · Llama-3.3-70B · GPT-4o Mini · GPT-4o · Claude 3.5 Sonnet

| Model | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness |
|-------|-------------|-----------------|-------------------|----------------|-------------------|
| **ChatAcredita PRO** | **0.7991** | **0.8210** | **0.6336** | **0.4683** | **0.5154** |
| DeepSeek-14B-Acredita | 0.7814 | 0.8089 | 0.7183 | 0.4201 | 0.4876 |
| Qwen2.5-7B-Acredita | 0.7342 | 0.7891 | 0.6102 | 0.3987 | 0.4423 |

### G-Eval (GPT-4o Mini, 1–5 scale)

| Model | Faithfulness | Relevance | Completeness | Coherence | **Composite** |
|-------|-------------|-----------|--------------|-----------|--------------|
| **ChatAcredita PRO** | 4.21 | 4.38 | 4.19 | 4.37 | **4.299** |

---

## Quick Start

### 1. Run the 5-judge RAGAS evaluation

```bash
pip install openai qdrant-client sentence-transformers

export HF_TOKEN="hf_..."
export OPENROUTER_API_KEY="sk-or-..."
export QDRANT_URL="https://your-cluster.qdrant.io"
export QDRANT_API_KEY="your-key"

python evaluation/ragas_multijuez.py
# Output: multijuez_results_v2.csv + multijuez_summary_v2.json
```

### 2. Run G-Eval

```bash
pip install openai pandas tqdm

python evaluation/geval_acredita.py \
    --input  data/benchmark_142q.jsonl \
    --output resultados_geval.csv
```

### 3. Apply RAG-VSD filter to your own synthetic pairs

```bash
pip install qdrant-client sentence-transformers tqdm

export QDRANT_URL="..."
export QDRANT_API_KEY="..."

# Place your synthetic pairs in data/synthetic_raw/synthetic_15k.jsonl
# Format: {"input": "<question>", "output": "<answer>"}

cd rag_vsd
python 02_filter_rag.py
# Output: data/synthetic_verified/verified.jsonl
```

### 4. Use ChatAcredita PRO directly

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "raulgdp/qwen3-14b-acredita"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

messages = [
    {"role": "system", "content": "Eres ChatAcredita PRO, experto en acreditación CNA Colombia."},
    {"role": "user", "content": "¿Cuántos factores tiene el modelo de acreditación del CNA?"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

---

## Training

Fine-tuned with QLoRA on a single NVIDIA RTX 4090 (24 GB VRAM) for < USD 90:

| Hyperparameter | Value |
|----------------|-------|
| LoRA rank (r) | 32 |
| LoRA alpha (α) | 64 |
| Trainable params | ~86M (0.6 %) |
| Optimizer | paged_adamw_8bit |
| Learning rate | 2×10⁻⁴ (cosine + 10 % warmup) |
| Epochs | 3 |
| Effective batch size | 16 |
| Max sequence length | 2 048 tokens |

See [`training/training_config.yaml`](training/training_config.yaml) for full configuration.

---

## Citation

```bibtex
@article{gutierrez2026chatacredita,
  title   = {ChatAcredita PRO: Fine-tuned LLMs with RAG-VSD Filtering
             for Colombian University Accreditation QA},
  author  = {Gutiérrez de Piñerez Reyes, Raúl Ernesto and others},
  journal = {Expert Systems with Applications},
  year    = {2026},
  publisher = {Elsevier},
  note    = {Manuscript \#25942}
}
```

---

## Contact

**Raúl Gutiérrez de Piñerez Reyes**  
Escuela de Ingeniería de Sistemas y Computación (EISC)  
Universidad del Valle · Cali, Colombia  
raul.gutierrez@correounivalle.edu.co · [HuggingFace](https://huggingface.co/raulgdp) · [GitHub](https://github.com/raulgdp)
=======
# **Official Repository — ChatAcredita PRO**

**Paper Title**
[**From RAG to ReAct: A Controlled Comparison of LLM Architectures for Low-Resource Institutional Question Answering**](https://arxiv.org/abs/2410.XXXXX)

This repository contains the **source code**, **benchmark**, and **evaluation harnesses** for our paper submitted to TACL 2027. It presents the first controlled comparison of four LLM-RAG architectures for Colombian university accreditation (CNA framework, Decree 1330/2019, EISC — Universidad del Valle), isolating architecture as the sole independent variable under a shared retrieval backbone.

> **Raúl Ernesto Gutierrez de Piñerez Reyes** · EISC, Universidad del Valle, Cali, Colombia
> HuggingFace: [raulgdp](https://huggingface.co/raulgdp)

---

## **Table of Contents**

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Systems](#3-systems)
4. [Benchmark](#4-benchmark)
5. [Evaluation](#5-evaluation)
6. [Results](#6-results)
7. [Dataset](#7-dataset)
8. [Contact](#8-contact)

---

## **1. Overview**

We compare four architectures under identical retrieval conditions (Qdrant Cloud, 79,879 chunks, BGE-M3+BM25+RRF+Reranker):

| System | Architecture | RAGAS avg | Latency (s) | DRE |
|--------|-------------|-----------|-------------|------------|
| **Sys-A** | Sequential RAG (baseline) | 0.785 | **4.1 ± 0.8** |0.832 |
| **Sys-B** | Inference-time RAFT | 0.696 | 38.0 ± 16.3 | 0.696 |
| **Sys-C** | Multi-Agent ReAct ⭐ | **0.838** | 99.8 ± 72.4 | **0.437** |
| **Sys-D** | LangGraph | 0.729 | 29.4 ± 10.4 | 0.390 |

**Key findings:**
- **ReAct orchestration** achieves the highest quality (+0.053 RAGAS over baseline)
- **Inference-time RAFT** without fine-tuning *degrades* quality (−0.089) at 9.3× latency
- **LangGraph** offers 3.4× lower latency than ReAct with 85% lower variance
- **Sequential RAG** remains the cost-efficiency frontier at $0.0004/query

---

## **2. Installation**

```bash
git clone https://github.com/raulgdp/chatacredita-pro.git
cd chatacredita-pro
pip install -r requirements.txt
```

Configure your environment:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required variables:

```env
OPENROUTER_API_KEY=sk-or-v1-...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key
```

---

## **3. Systems**

All systems share: Qdrant Cloud · BGE-M3 1024-dim · BM25 · RRF k=60 · BGE-Reranker-v2-m3

### Sys-A — Sequential RAG
```bash
streamlit run sys_a/app.py
```
512-token structural chunks · `pymupdf4llm` Markdown · Long-Context mode (corpus_lc.pkl)

### Sys-B — Inference-time RAFT
```bash
streamlit run sys_b/app.py
```
Flat 1,800-char chunks · 127,000 chunks · 58,418 QA-pair enrichment · `##begin_quote##` CoT

### Sys-C — Multi-Agent ReAct ⭐
```bash
streamlit run sys_c/app.py
```
Agent 1 (Orchestrator): DeepSeek-R1-0528 · `tool_choice=auto` · ≤3 rounds  
Agent 2 (Retriever): BGE-M3+BM25+Reranker top-5  
Agent 3 (Validator): Gemma-3-12B-IT → `{supported, confidence, issues, needs_more_search}`

### Sys-D — LangGraph
```bash
streamlit run sys_d/app.py
```
5 nodes: `retrieve → grade_docs → (rewrite) → generate → grade_answer → END`  
Typed `GraphState` · Python-only transitions · guaranteed termination

---

## **4. Benchmark**

**CNA-EISC-320** (`benchmark/benchmark_342.json`) — 320 questions across 6 categories:

| Category | n | Description |
|----------|---|-------------|
| Normative | 55 | CNA regulations, decrees, resolutions |
| Statistical | 77 | Numerical data, rates, rankings |
| Process | 54 | Procedures and workflows |
| Comparison | 29 | Multi-document comparisons |
| Synthesis | 52 | Summaries and overviews |
| General | 53 | Other accreditation questions |

142 expert-curated + 178 corpus-derived questions. Each entry: `id`, `category`, `question`, `reference`, `source`, `page`.

---

## **5. Evaluation**

```bash
# Sys-C — full evaluation
python evaluation/eval_sysC.py \
  --benchmark benchmark/benchmark_342.json \
  --delay 3 --output results_sysC.json

# Resume interrupted run
python evaluation/eval_sysC.py --resume --delay 3

# Quick test (10 questions)
python evaluation/eval_sysC.py --limit 10

# Sys-B and Sys-D
python evaluation/eval_sysB_sysD.py \
  --systems B D \
  --benchmark benchmark/benchmark_342.json \
  --delay 3
```

RAGAS metrics: `faithfulness` · `answer_relevance` · `context_precision` · `context_recall`  
Evaluator: `qwen/qwen3-32b` with `/no-think` → fallback `deepseek/deepseek-r1-0528`

---

## **6. Results**

### RAGAS metrics (320-question benchmark)

| Metric | Sys-A | Sys-B | Sys-C | Sys-D |
|--------|-------|-------|-------|-------|
| Faithfulness | 0.832 ± .09 | 0.740 ± .16 | **0.873 ± .10** | 0.779 ± .16 |
| Answer Relevance | 0.851 ± .08 | 0.752 ± .18 | **0.878 ± .12** | 0.798 ± .18 |
| Context Precision | 0.744 ± .12 | 0.669 ± .16 | **0.842 ± .15** | 0.704 ± .15 |
| Context Recall | 0.711 ± .14 | 0.623 ± .21 | **0.757 ± .19** | 0.635 ± .20 |
| **RAGAS avg** | 0.785 | 0.696 | **0.838** | 0.729 |
| DRE | 0.832 | 0.696 | **0.437** | 0.390 |
| Latency (s) | **4.1 ± 0.8** | 38.0 ± 16.3 | 99.8 ± 72.4 | 29.4 ± 10.4 |
| Cost/query | **$0.0004** | $0.0007 | $0.0021 | $0.0012 |

### RAGAS average by category

| Category | Sys-A | Sys-B | Sys-C | Sys-D |
|----------|-------|-------|-------|-------|
| Normative (n=55) | 0.832 | 0.711 | **0.879** | 0.764 |
| Statistical (n=77) | 0.785 | 0.702 | **0.830** | 0.743 |
| Process (n=54) | 0.851 | 0.714 | **0.854** | 0.737 |
| Comparison (n=29) | **0.832** | 0.660 | 0.734 | 0.645 |
| Synthesis (n=52) | 0.851 | 0.710 | **0.828** | 0.702 |
| General (n=53) | 0.785 | 0.712 | **0.856** | 0.735 |

### Decision Round Efficiency (DRE)

| Round | Sys-C | Sys-D |
|-------|-------|-------|
| Round 1 | 0.841 | 0.754 |
| Round 2 | 0.863 | 0.779 |
| Round 3 | **0.873** | — |
| **DRE** | **0.437** | 0.390 |

78% of Sys-C's agentic gain is captured at round 1.

---

## **7. Dataset**

The **CNA-EISC-RAFT-58K** dataset is available on HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("raulgdp/CNA-EISC-RAFT-58K")
# DatasetDict with train (46,734) / validation (5,841) / test (5,843)
```

- 58,418 QA triplets (Q, D\*, D\_k, A\*)
- k=4 distractors per example · 59.9% oracle\_included
- Sources: 60 EISC/CNA accreditation documents

---

## **8. Contact**

For questions or issues, open a [GitHub Issue](https://github.com/raulgdp/chatacredita-pro/issues) or email directly.

If you find this repository helpful, please cite our paper:

```bibtex
@article{garcia2027chatacredita,
  title   = {From {RAG} to {ReAct}: A Controlled Comparison of {LLM} Architectures
             for Low-Resource Institutional Question Answering},
  author  = {Garc{\'\i}a, Ra{\'u}l},
  journal = {Transactions of the Association for Computational Linguistics},
  year    = {2027},
  note    = {EISC, Universidad del Valle, Cali, Colombia.
             Preprint: arXiv:2410.XXXXX}
}
```
>>>>>>> ce5da59298b88a7b111700b1775718eb8fc4bc0d
