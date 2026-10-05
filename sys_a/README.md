# Sys-A — Semantic RAG (Baseline)

**Architecture:** Query → BGE-M3 dense + BM25 sparse → RRF → BGE-Reranker top-5 → DeepSeek-R1

## Key characteristics
- **Chunking:** 512 tokens, overlap 64, `pymupdf4llm` Markdown (structural — tables + headings preserved)
- **Index:** 79,879 chunks in Qdrant Cloud
- **Long-Context mode:** `corpus_lc.pkl` — full PDF as Markdown up to 80,000 chars (toggle in sidebar)
- **Generator:** DeepSeek-R1-0528 via OpenRouter
- **Retrieval:** Dense only (no BM25 in baseline mode)

## Results (320-question benchmark)

| Metric | Value |
|--------|-------|
| Faithfulness | 0.832 ± 0.09 |
| Answer Relevance | 0.851 ± 0.08 |
| Context Precision | 0.744 ± 0.12 |
| Context Recall | 0.711 ± 0.14 |
| **RAGAS avg** | **0.785** |
| Latency | **4.1 ± 0.8 s** |
| Cost/query | **$0.0004** |

> ✅ **Best cost-efficiency** of all four systems.

## Run

```bash
streamlit run sys_a/app.py
```

## Notes
- The Long-Context mode (`corpus_lc.pkl`) is built on first run from `PDF_DIR` using `pymupdf4llm`.
- Toggle in the sidebar to switch between chunked RAG and full-document mode.
- Leads in **comparison** queries (multi-document aggregation, RAGAS 0.832 vs Sys-C 0.734).
