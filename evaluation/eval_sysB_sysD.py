#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  eval_sysB_sysD.py — Harness RAGAS para Sys-B y Sys-D                    ║
# ║  Evalúa 6 variantes sobre el benchmark CNA-EISC-142                       ║
# ║                                                                            ║
# ║  Variantes:                                                                ║
# ║    Sys-A  — RAG semántico baseline (rag_acreditacion-1)                   ║
# ║    Sys-B  — RAFT híbrido BM25+Dense (app_raft_lc)                        ║
# ║    Sys-B+ — RAFT + Long-Context (corpus_lc.pkl)                           ║
# ║    Sys-D  — LangGraph (langgraph_acredita)                                ║
# ║    Sys-D+ — LangGraph + Long-Context (corpus_lc.pkl)                      ║
# ║    Sys-E  — RAG Router + LangGraph (lang_routed_lc)                       ║
# ║                                                                            ║
# ║  Uso:                                                                      ║
# ║    python eval_sysB_sysD.py --benchmark benchmark_142.json                ║
# ║    python eval_sysB_sysD.py --systems B D --delay 3                       ║
# ║    python eval_sysB_sysD.py --dry-run                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
from __future__ import annotations
import os, sys, json, time, re, argparse, statistics, pickle
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY",
    "os.getenv("OPENROUTER_API_KEY", "")")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GEN_MODEL       = os.getenv("GEN_MODEL",  "deepseek/deepseek-r1-0528")
EVAL_MODEL      = os.getenv("EVAL_MODEL", "qwen/qwen3-32b")         # juez principal: mejor razonamiento JSON
EVAL_FALLBACK   = os.getenv("EVAL_FALLBACK","deepseek/deepseek-r1-0528")  # fallback robusto
FAST_EVAL       = os.getenv("FAST_EVAL",  "qwen/qwen3-8b:free")    # último fallback gratuito
QDRANT_URL      = os.getenv("QDRANT_URL",
    "os.getenv("QDRANT_URL", "")")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY",
    "os.getenv("QDRANT_API_KEY", "")")
COLLECTION      = "acreditacion"
DATASET_PATH    = os.getenv("RAFT_DATASET", "dataset_raft_v4.json")
LC_CACHE        = "corpus_lc.pkl"
PDF_DIR         = os.getenv("PDF_DIR", "pdfs")
LC_MAX_CHARS    = 80_000
TOP_K_CAND      = 20
TOP_K_FINAL     = 8

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    return OpenAI(api_key=OPENROUTER_KEY, base_url=OPENROUTER_BASE, timeout=90)

def safe_call(model: str, messages: list, temperature: float = 0.1,
              max_tokens: int = 1000) -> str:
    """Llamada LLM segura — nunca devuelve None ni vacío por thinking."""
    try:
        r   = get_client().chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        raw = r.choices[0].message.content or ""
        # Texto fuera del thinking
        outside = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if outside:
            return outside
        # Si solo hay thinking (tokens agotados), extraer últimos 2 párrafos del think
        think = re.search(r"<think>(.*)", raw, re.DOTALL)
        if think:
            parrs = [p.strip() for p in think.group(1).split("\n\n") if p.strip()]
            return "\n\n".join(parrs[-2:]) if parrs else raw.strip()
        return raw.strip()
    except Exception as e:
        return f"ERROR: {e}"

def safe_json(model: str, prompt: str, default: dict) -> dict:
    raw = safe_call(model, [{"role":"user","content":prompt}],
                    temperature=0.0, max_tokens=200)
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group()) if m else default
    except Exception:
        return default

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE RECURSOS (lazy, una sola vez)
# ─────────────────────────────────────────────────────────────────────────────
_resources: dict = {}

def get_embedder():
    if "embedder" not in _resources:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Cargando BGE-M3 en {device}…", end=" ", flush=True)
        _resources["embedder"] = SentenceTransformer("BAAI/bge-m3", device=device)
        print("✅")
    return _resources["embedder"]

def get_reranker():
    if "reranker" not in _resources:
        try:
            from FlagEmbedding import FlagReranker
            import torch
            device = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
            print("  Cargando BGE-Reranker-v2-m3…", end=" ", flush=True)
            _resources["reranker"] = FlagReranker(
                "BAAI/bge-reranker-v2-m3", use_fp16=True, devices=device)
            print("✅")
        except ImportError:
            print("  ⚠️  FlagEmbedding no disponible — sin reranker")
            _resources["reranker"] = None
    return _resources["reranker"]

def get_qdrant():
    if "qdrant" not in _resources:
        from qdrant_client import QdrantClient
        print("  Conectando Qdrant…", end=" ", flush=True)
        _resources["qdrant"] = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        info = _resources["qdrant"].get_collection(COLLECTION)
        print(f"✅ {info.points_count:,} chunks")
    return _resources["qdrant"]

def get_bm25():
    """BM25 sobre colección Qdrant completa (con caché en memoria)."""
    if "bm25" in _resources:
        return _resources["bm25"]

    from rank_bm25 import BM25Okapi
    import unicodedata, numpy as np

    def norm(t):
        t = unicodedata.normalize("NFD", t.lower())
        return "".join(c for c in t if unicodedata.category(c) != "Mn")

    qdrant = get_qdrant()
    texts, ids, sources = [], [], []
    offset = None
    print("  Construyendo índice BM25…", end=" ", flush=True)
    while True:
        res = qdrant.scroll(collection_name=COLLECTION, limit=500,
                            offset=offset, with_payload=True, with_vectors=False)
        for pt in res[0]:
            txt = (pt.payload or {}).get("text", "").strip()
            if txt:
                texts.append(txt)
                ids.append(str(pt.id))
                sources.append((pt.payload or {}).get("source", ""))
        offset = res[1]
        if offset is None:
            break

    bm25 = BM25Okapi([norm(t).split() for t in texts])
    _resources["bm25"] = (bm25, texts, ids, sources, norm)
    print(f"✅ {len(texts):,} docs indexados")
    return _resources["bm25"]

def get_raft_index() -> dict:
    if "raft_idx" not in _resources:
        if not Path(DATASET_PATH).exists():
            print(f"  ⚠️  Dataset RAFT no encontrado: {DATASET_PATH}")
            _resources["raft_idx"] = {}
            return {}
        print(f"  Cargando RAFT index desde {DATASET_PATH}…", end=" ", flush=True)
        data = json.load(open(DATASET_PATH, encoding="utf-8"))
        idx: dict = {}
        for item in data:
            if not item.get("oracle_incluido"):
                continue
            ctx  = (item.get("oracle_chunk") or "").strip()
            preg = (item.get("pregunta") or "").strip()
            resp = (item.get("respuesta_corta") or "").strip()
            if ctx and preg:
                idx.setdefault(ctx, []).append({"pregunta": preg, "respuesta": resp})
        _resources["raft_idx"] = idx
        print(f"✅ {len(idx):,} contextos únicos")
    return _resources["raft_idx"]

def get_lc_corpus() -> list[dict]:
    if "lc_corpus" in _resources:
        return _resources["lc_corpus"]

    if Path(LC_CACHE).exists():
        print(f"  Cargando {LC_CACHE}…", end=" ", flush=True)
        corpus = pickle.load(open(LC_CACHE, "rb"))
        print(f"✅ {len(corpus)} docs")
        _resources["lc_corpus"] = corpus
        return corpus

    print(f"  Construyendo corpus_lc.pkl desde {PDF_DIR}/…")
    try:
        import fitz, pymupdf4llm
    except ImportError:
        print("  ⚠️  pip install pymupdf pymupdf4llm")
        _resources["lc_corpus"] = []
        return []

    pdfs = sorted(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"  ⚠️  Sin PDFs en {PDF_DIR}/")
        _resources["lc_corpus"] = []
        return []

    corpus = []
    for p in pdfs:
        doc, text = fitz.open(p), ""
        for pn in range(len(doc)):
            try:
                md = pymupdf4llm.to_markdown(doc, pages=[pn],
                                              show_progress=False, page_chunks=False)
            except Exception:
                md = doc[pn].get_text()
            text += f"\n--- Pág {pn+1} ---\n{md}"
        doc.close()
        if text.strip():
            corpus.append({"title": p.name, "text": text})

    pickle.dump(corpus, open(LC_CACHE, "wb"))
    print(f"✅ {len(corpus)} docs → {LC_CACHE}")
    _resources["lc_corpus"] = corpus
    return corpus

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_retrieve(query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    """BM25 + BGE-M3 → RRF → rerank → top_k."""
    import numpy as np, concurrent.futures, unicodedata

    embedder = get_embedder()
    qdrant   = get_qdrant()
    reranker = get_reranker()
    bm25, texts, ids, sources, norm = get_bm25()

    # Dense
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    hits  = qdrant.query_points(collection_name=COLLECTION, query=q_emb,
                                 limit=TOP_K_CAND, with_payload=True).points
    dense = [{"id": str(h.id), "text": (h.payload or {}).get("text",""),
              "score": h.score, "source": (h.payload or {}).get("source","")}
             for h in hits]

    # BM25
    bm25_scores = bm25.get_scores(norm(query).split())
    ranked      = np.argsort(bm25_scores)[::-1][:TOP_K_CAND]
    sparse      = [{"id": ids[i], "text": texts[i],
                    "score": float(bm25_scores[i]), "source": sources[i]}
                   for i in ranked if bm25_scores[i] > 0]

    # RRF k=60
    rrf: dict = {}
    id2p: dict = {}
    for rk, r in enumerate(dense):
        rrf[r["id"]] = rrf.get(r["id"], 0.0) + 1.0 / (60 + rk + 1)
        id2p.setdefault(r["id"], r)
    for rk, r in enumerate(sparse):
        rrf[r["id"]] = rrf.get(r["id"], 0.0) + 1.0 / (60 + rk + 1)
        id2p.setdefault(r["id"], r)

    cands = [id2p[pid] for pid, _ in
             sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:TOP_K_CAND]
             if pid in id2p]

    # Reranking
    if reranker and cands:
        pairs  = [[query, c["text"][:1024]] for c in cands]
        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        for c, s in zip(cands, scores):
            c["rerank_score"] = float(s)
            c["final_score"]  = 0.7 * float(s) + 0.3 * rrf.get(c["id"], 0.0)
        cands.sort(key=lambda x: x["final_score"], reverse=True)
    else:
        for c in cands:
            c["final_score"] = rrf.get(c["id"], 0.0)

    return cands[:top_k]

# ─────────────────────────────────────────────────────────────────────────────
# SYS-B — RAFT híbrido
# ─────────────────────────────────────────────────────────────────────────────
RAFT_SYSTEM = (
    "Eres un asistente experto en acreditación universitaria CNA/EISC, Universidad del Valle. "
    "Responde SIEMPRE en español. "
    "Basa tu respuesta ÚNICAMENTE en los documentos del contexto. "
    "Usa ##begin_quote## texto exacto ##end_quote## para citar fuentes. "
    "Si no encuentras la información, dilo explícitamente."
)

def run_sys_b(question: str) -> tuple[str, list[dict], float]:
    """Sys-B: RAFT BM25+Dense+Rerank+QA enrichment."""
    t0      = time.time()
    docs    = hybrid_retrieve(question, top_k=TOP_K_FINAL)
    raft_idx = get_raft_index()
    context  = ""

    for i, d in enumerate(docs, 1):
        txt      = d.get("text", "")
        fuente   = Path(str(d.get("source",""))).name or f"doc_{i}"
        enriched = ""
        for ctx_key, pares in raft_idx.items():
            if ctx_key[:80] in txt[:500] or txt[:80] in ctx_key:
                qa_lines = "\n".join(f"  Q: {p['pregunta']} → A: {p['respuesta'][:120]}"
                                     for p in pares[:2])
                if qa_lines:
                    enriched = f"\n[Pares RAFT de este fragmento:]\n{qa_lines}"
                break
        context += f"[Doc {i} · {fuente}]\n{txt[:800]}{enriched}\n\n"

    prompt = (f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}\n\n"
              "Sigue el método RAFT: razona, cita con ##begin_quote##...##end_quote##, "
              "luego da la respuesta final.")
    answer = safe_call(GEN_MODEL,
                       [{"role":"system","content":RAFT_SYSTEM},
                        {"role":"user","content":prompt}],
                       temperature=0.2, max_tokens=1000)
    return answer, docs, time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# SYS-B+ — RAFT + Long-Context
# ─────────────────────────────────────────────────────────────────────────────
def run_sys_b_lc(question: str) -> tuple[str, list[dict], float]:
    """Sys-B+: Long-Context sobre corpus_lc.pkl."""
    t0     = time.time()
    corpus = get_lc_corpus()
    if not corpus:
        return "⚠️ corpus_lc.pkl no disponible.", [], time.time() - t0

    ctx    = "\n\n===\n\n".join(
                 f"[Fuente: {d['title']}]\n{d['text']}" for d in corpus
             )[:LC_MAX_CHARS]
    prompt = (f"Responde usando ÚNICAMENTE el contexto proporcionado.\n"
              f"Para listas: copia los elementos EXACTAMENTE como aparecen.\n"
              f"Cita la fuente: [Fuente: nombre_documento]\n\n"
              f"CONTEXTO:\n{ctx}\n\nPREGUNTA: {question}\nRESPUESTA EN ESPAÑOL:")
    answer = safe_call(GEN_MODEL,
                       [{"role":"system","content":RAFT_SYSTEM},
                        {"role":"user","content":prompt}],
                       temperature=0.0, max_tokens=1500)
    return answer, [], time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# SYS-D — LangGraph
# ─────────────────────────────────────────────────────────────────────────────
def run_sys_d(question: str) -> tuple[str, list[dict], float]:
    """Sys-D: LangGraph 5-nodos vía langgraph_acredita."""
    t0 = time.time()
    try:
        from langgraph_acredita import answer_with_langgraph
        result = answer_with_langgraph(question)
        return (result.get("answer",""), result.get("docs",[]), time.time()-t0)
    except ImportError:
        # Fallback: simulamos el pipeline con retrieval híbrido + grade
        docs   = hybrid_retrieve(question)
        ctx    = "\n---\n".join(d.get("text","")[:600] for d in docs)
        prompt = (f"Responde en español basándote solo en estos fragmentos:\n{ctx}\n\n"
                  f"Pregunta: {question}")
        answer = safe_call(GEN_MODEL,
                           [{"role":"user","content":prompt}],
                           temperature=0.2, max_tokens=1000)
        return answer, docs, time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# SYS-D+ — LangGraph + Long-Context
# ─────────────────────────────────────────────────────────────────────────────
def run_sys_d_lc(question: str) -> tuple[str, list[dict], float]:
    """Sys-D+: Long-Context con prompt estilo LangGraph."""
    t0     = time.time()
    corpus = get_lc_corpus()
    if not corpus:
        return "⚠️ corpus_lc.pkl no disponible.", [], time.time() - t0

    ctx    = "\n\n===\n\n".join(
                 f"[Fuente: {d['title']}]\n{d['text']}" for d in corpus
             )[:LC_MAX_CHARS]
    system = ("Eres un asistente experto en acreditación CNA/EISC. "
              "Responde ÚNICAMENTE en español. "
              "Basa tu respuesta SOLO en el contexto. "
              "NO inventes datos, nombres ni cifras. "
              "Cita la fuente: [Fuente: nombre_documento]")
    prompt = (f"CONTEXTO:\n{ctx}\n\n"
              f"PREGUNTA: {question}\n\n"
              "Razona brevemente y da la respuesta final en español:")
    answer = safe_call(GEN_MODEL,
                       [{"role":"system","content":system},
                        {"role":"user","content":prompt}],
                       temperature=0.0, max_tokens=1500)
    return answer, [], time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS RAGAS (juez LLM)
# ─────────────────────────────────────────────────────────────────────────────
def parse_ragas_json(raw: str) -> dict | None:
    """Parser robusto para respuestas RAGAS — maneja DeepSeek thinking."""
    keys = ["faithfulness","answer_relevance","context_precision","context_recall"]

    def _extract(text: str) -> dict | None:
        text = re.sub(r"```(?:json)?","",text).strip().rstrip("`").strip()
        m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                if any(k in d for k in keys):
                    return {k: float(d.get(k, 0.5)) for k in keys}
            except Exception:
                pass
            try:
                d = json.loads(m.group().replace("'",'"'))
                if any(k in d for k in keys):
                    return {k: float(d.get(k, 0.5)) for k in keys}
            except Exception:
                pass
        nums = re.findall(r'\b(0\.\d+|1\.0+)\b', text)
        if len(nums) >= 4:
            return {k: float(v) for k, v in zip(keys, nums[:4])}
        return None

    # 1. Fuera del thinking (más fiable)
    outside = re.sub(r"<think>.*?</think>","",raw,flags=re.DOTALL).strip()
    result  = _extract(outside)
    if result:
        return result

    # 2. Dentro del thinking como fallback
    think = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think:
        result = _extract(think.group(1))
        if result:
            return result

    # 3. Raw completo
    return _extract(raw)



def compute_ragas(question: str, answer: str,
                  context: str, reference: str) -> dict:
    """1 sola llamada LLM con las 4 métricas → menos rate limiting."""
    default = {"faithfulness":0.5,"answer_relevance":0.5,
               "context_precision":0.5,"context_recall":0.5}
    if not answer or answer.startswith("ERROR"):
        return default

    system_eval = (
        "/no-think "  # Qwen3: desactiva thinking, respuesta JSON directa\n"
        "Eres un evaluador experto de sistemas RAG en español. "
        "Responde ÚNICAMENTE con un objeto JSON con 4 valores numéricos reales "
        "entre 0.1 y 1.0 según tu evaluación. Nunca copies el ejemplo literal."
    )
    prompt = (
        f"PREGUNTA: {question}\n"
        f"CONTEXTO RECUPERADO:\n{context[:1800]}\n\n"
        f"RESPUESTA GENERADA:\n{answer[:700]}\n\n"
        f"RESPUESTA REFERENCIA:\n{reference[:400]}\n\n"
        "Evalúa y asigna valores REALES (no copies el ejemplo):\n"
        "- faithfulness (0=alucinación total, 1=completamente fiel al contexto)\n"
        "- answer_relevance (0=irrelevante, 1=responde exactamente la pregunta)\n"
        "- context_precision (0=contexto inútil, 1=todo el contexto es relevante)\n"
        "- context_recall (0=falta información, 1=contexto cubre la referencia)\n\n"
        "Formato de respuesta — sustituye X.XX con tu evaluación real:\n"
        "{\"faithfulness\": X.XX, \"answer_relevance\": X.XX, "
        "\"context_precision\": X.XX, \"context_recall\": X.XX}"
    )

    # Cadena: Qwen3-32B → DeepSeek-R1 → Qwen3-8B
    for model, max_tok, delay in [
        (EVAL_MODEL,    800, 0),   # qwen3-32b /no-think: JSON directo ~100 tok
        (EVAL_FALLBACK, 1400, 3),  # deepseek-r1: thinking ~900 + JSON ~80
        (FAST_EVAL,     300, 2),   # qwen3-8b:free sin thinking
    ]:
        time.sleep(delay)
        msgs = [
            {"role": "system", "content": system_eval},
            {"role": "user",   "content": prompt},
        ]
        raw = safe_call(model, msgs, temperature=0.0, max_tokens=max_tok)
        if raw.startswith("ERROR"):
            continue
        parsed = parse_ragas_json(raw)
        # Rechazar si todos son 0.0 (modelo copió el ejemplo)
        if parsed and any(v > 0.05 for v in parsed.values()):
            return parsed

    return default

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────
SYSTEMS = {
    "B":    ("Sys-B  (RAFT híbrido BM25+Dense)",      run_sys_b),
    "B+":   ("Sys-B+ (RAFT + Long-Context LC)",        run_sys_b_lc),
    "D":    ("Sys-D  (LangGraph 5-nodos)",             run_sys_d),
    "D+":   ("Sys-D+ (LangGraph + Long-Context LC)",   run_sys_d_lc),
}

def main():
    parser = argparse.ArgumentParser(
        description="Evaluación RAGAS para Sys-B y Sys-D con variantes Long-Context")
    parser.add_argument("--benchmark",  default="benchmark_142.json")
    parser.add_argument("--systems",    nargs="+", default=["B","B+","D","D+"],
                        choices=["B","B+","D","D+"])
    parser.add_argument("--output",     default="results_sysB_sysD.json")
    parser.add_argument("--delay",      type=float, default=2.5)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--resume",     action="store_true",
                        help="Reanudar desde results_sysB_sysD.json existente")
    args = parser.parse_args()

    # Cargar benchmark
    questions = json.load(open(args.benchmark, encoding="utf-8"))
    print(f"\n{'='*60}")
    print(f"Evaluación RAGAS — {len(questions)} preguntas")
    print(f"Sistemas: {args.systems}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nDRY RUN — sistemas y preguntas:")
        for sid in args.systems:
            name, _ = SYSTEMS[sid]
            print(f"  [{sid}] {name}")
        for q in questions[:5]:
            print(f"  [{q.get('id')}] {q['question'][:70]}")
        print(f"  … {len(questions)} preguntas totales")
        return

    # Reanudar si existe output
    all_results: dict = {}
    if args.resume and Path(args.output).exists():
        all_results = json.load(open(args.output, encoding="utf-8"))
        print(f"Reanudando desde {args.output}")

    # Pre-cargar recursos necesarios
    print("\nCargando recursos…")
    needs_hybrid = any(s in args.systems for s in ["B","D"])
    needs_lc     = any(s in args.systems for s in ["B+","D+"])

    if needs_hybrid:
        get_embedder()
        get_reranker()
        get_qdrant()
        get_bm25()
        if "B" in args.systems:
            get_raft_index()
    if needs_lc:
        get_lc_corpus()

    # Evaluación por sistema
    for sys_id in args.systems:
        sys_name, run_fn = SYSTEMS[sys_id]

        if sys_id in all_results:
            print(f"\n[{sys_id}] Ya evaluado — saltando")
            continue

        print(f"\n{'─'*60}")
        print(f"[{sys_id}] {sys_name}")
        print(f"{'─'*60}")

        rows   = []
        m_acc  = {"faithfulness":[],"answer_relevance":[],
                   "context_precision":[],"context_recall":[]}
        lats   = []

        for qi, q in enumerate(questions):
            qid = q.get("id", f"Q{qi+1:03d}")
            cat = q.get("category","?")
            print(f"\n  [{qi+1:3d}/{len(questions)}] {qid}·{cat} "
                  f"{q['question'][:55]}…")

            # Generar respuesta
            try:
                answer, docs, latency = run_fn(q["question"])
            except Exception as e:
                print(f"    ❌ Error: {e}")
                answer, docs, latency = f"ERROR: {e}", [], 0.0

            # Construir contexto para RAGAS
            if docs:
                context = "\n---\n".join(d.get("text","")[:400] for d in docs[:5])
            elif "lc" in sys_id.lower() or sys_id in ["B+","D+"]:
                # LC: usar primeras 2000 chars del corpus como referencia
                corpus  = get_lc_corpus()
                context = " ".join(d["text"][:500] for d in corpus[:3])
            else:
                context = ""

            print(f"    A: {answer[:80]}…  [{latency:.1f}s]")

            # RAGAS
            time.sleep(args.delay)
            scores = compute_ragas(q["question"], answer,
                                   context, q.get("reference",""))

            avg = sum(scores[m] for m in m_acc) / 4
            print(f"    RAGAS={avg:.3f} | F={scores['faithfulness']:.3f} "
                  f"R={scores['answer_relevance']:.3f} "
                  f"P={scores['context_precision']:.3f} "
                  f"C={scores['context_recall']:.3f}")

            for m in m_acc:
                m_acc[m].append(scores[m])
            lats.append(latency)

            rows.append({
                "id":       qid,
                "category": cat,
                "question": q["question"],
                "answer":   answer[:500],
                "latency_s": round(latency, 2),
                **scores,
                "ragas_avg": round(avg, 4),
            })

        # Resumen por sistema
        print(f"\n  {'─'*40}")
        print(f"  RESUMEN [{sys_id}] {sys_name}")
        summary = {}
        for m, vals in m_acc.items():
            mu  = statistics.mean(vals)
            sig = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {m:22s}: {mu:.3f} ± {sig:.3f}")
            summary[m] = {"mean": round(mu,4), "std": round(sig,4)}

        ragas_avg = sum(summary[m]["mean"] for m in m_acc) / 4
        lat_mu    = statistics.mean(lats)
        lat_sig   = statistics.stdev(lats) if len(lats) > 1 else 0.0
        print(f"  {'RAGAS (avg)':22s}: {ragas_avg:.3f}")
        print(f"  {'Latencia (s)':22s}: {lat_mu:.1f} ± {lat_sig:.1f}")

        # Resumen por categoría
        cat_summary: dict = {}
        for row in rows:
            c = row["category"]
            cat_summary.setdefault(c, []).append(row["ragas_avg"])
        print(f"\n  Por categoría:")
        for c, vals in sorted(cat_summary.items()):
            print(f"    {c:15s}: {statistics.mean(vals):.3f}")

        all_results[sys_id] = {
            "name":      sys_name,
            "summary":   summary,
            "ragas_avg": round(ragas_avg, 4),
            "latency":   {"mean": round(lat_mu,2), "std": round(lat_sig,2)},
            "by_category": {c: round(statistics.mean(v),4)
                            for c,v in cat_summary.items()},
            "n":         len(rows),
            "rows":      rows,
        }

        # Guardar progreso (por si se interrumpe)
        json.dump(all_results, open(args.output,"w",encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\n  💾 Guardado: {args.output}")

    # ── Tabla comparativa final ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TABLA COMPARATIVA FINAL")
    print(f"{'='*60}")
    hdr = f"{'Sistema':<28} {'Faith':>6} {'Relev':>6} {'Prec':>6} {'Rec':>6} {'avg':>6} {'Lat(s)':>7}"
    print(hdr)
    print("─" * len(hdr))

    for sid, data in all_results.items():
        s = data["summary"]
        print(f"{data['name'][:28]:<28} "
              f"{s['faithfulness']['mean']:>6.3f} "
              f"{s['answer_relevance']['mean']:>6.3f} "
              f"{s['context_precision']['mean']:>6.3f} "
              f"{s['context_recall']['mean']:>6.3f} "
              f"{data['ragas_avg']:>6.3f} "
              f"{data['latency']['mean']:>7.1f}")

    print(f"\n✅ Resultados completos: {args.output}")


if __name__ == "__main__":
    main()
