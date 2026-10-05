# Sys-D — LangGraph (Deterministic Typed State)

**Architecture:**
```
Query → retrieve → grade_docs → generate → grade_answer → Answer
                      ↓ not_relevant              ↓ not_supported
                   rewrite ←──────────── retry (≤ Rmax=3 times)
```

## Key characteristics
- **Chunking:** structural 512-tok, overlap 64 (same Qdrant index as Sys-A/C)
- **Index:** 79,879 chunks
- **State:** typed `GraphState` — all branching via Python conditions, never LLM output distribution
- **Transitions:** guaranteed termination (bounded iterations)

### State transitions

| ID | Transition | Python condition |
|----|-----------|-----------------|
| T1 | `grade_docs → rewrite` | `doc_grade == "not_relevant" and iteration < Rmax` |
| T2 | `grade_docs → generate` | `doc_grade == "relevant"` |
| T3 | `grade_answer → END` | `answer_grade == "supported"` |
| T4 | `grade_answer → rewrite` | `answer_grade != "supported" and iteration < Rmax` |

## Results (320-question benchmark)

| Metric | Value |
|--------|-------|
| Faithfulness | 0.779 ± 0.16 |
| Answer Relevance | 0.798 ± 0.18 |
| Context Precision | 0.704 ± 0.15 |
| Context Recall | 0.635 ± 0.20 |
| **RAGAS avg** | **0.729** |
| Latency | **29.4 ± 10.4 s** |
| Cost/query | $0.0012 |
| Rounds avg | 2.1 |

> ✅ **Best latency variance** (σ = 10.4 s vs 72.4 s for Sys-C) — preferable for production SLAs.
> Achieves 0.729 RAGAS avg at **3.4× lower latency** than Sys-C (99.8 s).

## Run

```bash
# Requires langgraph_acredita.py in the same directory
streamlit run sys_d/app.py
```

## Key advantage over Sys-C
Deterministic Python transitions vs LLM-driven flow control:
- **85% lower latency variance** (10.4 vs 72.4 s σ)
- **Guaranteed termination** at Rmax iterations
- Preferred when latency predictability matters more than peak quality
