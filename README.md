# ChatAcredita PRO

[![Paper](https://img.shields.io/badge/Paper-IP%26M%202026-blue)](https://www.sciencedirect.com/journal/information-processing-and-management)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-raulgdp-yellow)](https://huggingface.co/raulgdp)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

**Fine-tuned LLMs with RAG-VSD filtering for Colombian university accreditation QA (CNA / Decreto 1330 de 2019)**

> Gutiérrez de Piñerez Reyes, R. E. *"ChatAcredita PRO: Fine-tuned LLMs with RAG-VSD Filtering for Colombian University Accreditation QA"* — Information Processing & Management, 2026.

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

**Vector store:** Qdrant Cloud (AWS us-east-1)  
**Embedding:** `BAAI/bge-m3` (1024-dim)  
**Reranker:** `BAAI/bge-reranker-v2-m3`  
**Corpus:** CNA documentation + EISC Universidad del Valle accreditation reports (79,879 chunks)

---

## RAG-VSD Data Filtering

The RAG-VSD (RAG Verification by Semantic Distance) filter accepts a synthetic pair only when **all three** conditions hold:

| Threshold | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Lexical overlap | τ_overlap | 0.30 | ≥ 30% of response tokens appear in retrieved chunks |
| Semantic similarity | τ_query | 0.55 | cosine(query, top-1 chunk) ≥ 0.55 |
| CNA term density | τ_tech | 0.40 | ≥ 40% of response tokens are CNA technical terms |

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
| Trainable params | ~86M (0.6%) |
| Optimizer | paged_adamw_8bit |
| Learning rate | 2×10⁻⁴ (cosine + 10% warmup) |
| Epochs | 3 |
| Effective batch size | 16 |
| Max sequence length | 2,048 tokens |

See [`training/training_config.yaml`](training/training_config.yaml) for full configuration.

---

## Related Repository

The architecture comparison study (SYS-A / SYS-B / SYS-C / SYS-D) that uses the same retrieval infrastructure is published separately:

> **[raulgdp/from-rag-to-react](https://github.com/raulgdp/from-rag-to-react)** — *From RAG to ReAct: A Controlled Comparison of LLM Architectures for Low-Resource Institutional QA*

---

## Citation

```bibtex
@article{gutierrez2026chatacredita,
  title   = {{ChatAcredita PRO}: Fine-tuned {LLMs} with {RAG-VSD} Filtering
             for {Colombian} University Accreditation {QA}},
  author  = {Gutiérrez de Piñerez Reyes, Raúl Ernesto},
  journal = {Information Processing \& Management},
  year    = {2026},
  note    = {EISC, Universidad del Valle, Cali, Colombia}
}
```

---

## Contact

For questions or issues, open a [GitHub Issue](https://github.com/raulgdp/chatacredita-pro/issues).
