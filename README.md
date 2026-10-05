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

| System | Architecture | RAGAS avg | Latency (s) | Cost/query |
|--------|-------------|-----------|-------------|------------|
| **Sys-A** | Sequential RAG (baseline) | 0.785 | **4.1 ± 0.8** | **$0.0004** |
| **Sys-B** | Inference-time RAFT | 0.696 | 38.0 ± 16.3 | $0.0007 |
| **Sys-C** | Multi-Agent ReAct ⭐ | **0.838** | 99.8 ± 72.4 | $0.0021 |
| **Sys-D** | LangGraph | 0.729 | 29.4 ± 10.4 | $0.0012 |

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
