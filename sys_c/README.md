# Sys-C — Multi-Agent ReAct ⭐ Best Quality

**Architecture:**
```
Query
  → Agent 1: Orchestrator (DeepSeek-R1-0528, tool_choice=auto)
       ↗ search()  → Agent 2: Retriever (BGE-M3 + BM25 + Reranker top-5)
       ↙ verdict   ← Agent 3: Validator (Gemma-3-12B-IT → JSON)
  → Answer  (≤ 3 rounds, S_max = 3 searches, forced-answer guard)
```

## Key characteristics
- **Chunking:** structural 512-tok, overlap 64, `pymupdf4llm` Markdown (same index as Sys-A/D)
- **Index:** 79,879 chunks in Qdrant Cloud
- **Orchestrator:** DeepSeek-R1-0528, `tool_choice=auto`, MAX_TOOL_ROUNDS=3
- **Retriever:** BGE-M3 dense + BM25 → RRF (0.6/0.4) → BGE-Reranker top-5
- **Validator output:** `{supported: bool, confidence: float, issues: str, needs_more_search: bool}`
- **Guard:** forced-answer fallback via Qwen3-32B when orchestrator exhausts rounds

## Results (320-question benchmark)

| Metric | Value |
|--------|-------|
| Faithfulness | **0.873 ± 0.10** |
| Answer Relevance | **0.878 ± 0.12** |
| Context Precision | **0.842 ± 0.15** |
| Context Recall | **0.757 ± 0.19** |
| **RAGAS avg** | **0.838** |
| Latency | 99.8 ± 72.4 s |
| Cost/query | $0.0021 |
| Rounds avg | 2.5 |

> ✅ **Best system on all four RAGAS metrics** — leads in 5 of 6 question categories.

### Results by Category

| Category | n | RAGAS avg |
|----------|---|-----------|
| Normative | 55 | **0.879** |
| Process | 54 | **0.854** |
| General | 53 | **0.856** |
| Statistical | 77 | **0.830** |
| Synthesis | 52 | **0.828** |
| Comparison | 29 | 0.734 ← Sys-A leads here |

### Decision Round Efficiency

| Round | Faithfulness |
|-------|-------------|
| 1 | 0.841 |
| 2 | 0.863 |
| 3 | **0.873** |

78% of total gain is captured at Round 1.

## Run

```bash
streamlit run sys_c/app.py
```
