# Sys-B — Inference-time RAFT

**Architecture:** Query → BGE-M3 + BM25 → RRF → Reranker top-8 → RAFT Enrichment → DeepSeek-R1

## Key characteristics
- **Chunking:** flat 1,800 chars, no overlap, no structural parsing
- **Index:** 127,000 chunks in Qdrant Cloud
- **RAFT dataset:** [CNA-EISC-RAFT-58K](https://huggingface.co/datasets/raulgdp/CNA-EISC-RAFT-58K) — 58,418 QA triplets
  - k=4 distractors per example (distractor ratio 40.1%)
  - 59.9% oracle_included = True
  - Split 80/10/10 (seed=42)
- **CoT format:** `##begin_quote##`...`##end_quote##` citation prompt
- **Generator:** DeepSeek-R1-0528 via OpenRouter

## Results (320-question benchmark)

| Metric | Value |
|--------|-------|
| Faithfulness | 0.740 ± 0.16 |
| Answer Relevance | 0.752 ± 0.18 |
| Context Precision | 0.669 ± 0.16 |
| Context Recall | 0.623 ± 0.21 |
| **RAGAS avg** | **0.696** |
| Latency | 38.0 ± 16.3 s |
| Cost/query | $0.0007 |

> ⚠️ **Inference-time RAFT without fine-tuning underperforms Sys-A** by −0.089 RAGAS avg
> at 9.3× the latency. Fine-tuning on the 46,734 training examples is required to realize RAFT gains.

## Run

```bash
# Requires dataset_raft_v4.json (or load from HuggingFace)
RAFT_DATASET=dataset_raft_v4.json streamlit run sys_b/app.py
```

## RAFT Dataset

```python
from datasets import load_dataset
ds = load_dataset("raulgdp/CNA-EISC-RAFT-58K")
print(ds)
# DatasetDict({
#     train: Dataset({features: ['pregunta','respuesta_corta','oracle_chunk','distractor_chunks','oracle_incluido'], num_rows: 46734})
#     validation: Dataset({num_rows: 5841})
#     test: Dataset({num_rows: 5843})
# })
```
