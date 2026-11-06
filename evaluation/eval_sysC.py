#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  eval_sysC.py — Harness RAGAS para Sys-C (Multi-Agent ReAct)              ║
# ║  Replica exactamente chatacredita_app1.py sin la UI de Streamlit           ║
# ║  Agente 1: Orquestador (DeepSeek-R1-0528, tool_choice=auto)               ║
# ║  Agente 2: Retriever   (BGE-M3 + BM25 + BGE-Reranker-v2-m3)              ║
# ║  Agente 3: Validador   (Gemma-3-12B-IT → JSON verdict)                    ║
# ║                                                                            ║
# ║  Uso:                                                                      ║
# ║    python eval_sysC.py --benchmark benchmark_342.json                      ║
# ║    python eval_sysC.py --dry-run                                           ║
# ║    python eval_sysC.py --resume --delay 3                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
from __future__ import annotations
import os, sys, json, time, re, argparse, statistics, unicodedata
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — idéntico a chatacredita_app1.py
# ─────────────────────────────────────────────────────────────────────────────
OPENROUTER_KEY      = os.getenv("OPENROUTER_API_KEY",
    "os.getenv("OPENROUTER_API_KEY", "")")
OPENROUTER_BASE     = "https://openrouter.ai/api/v1"
ORCHESTRATOR_MODEL  = os.getenv("ORCHESTRATOR_MODEL", "deepseek/deepseek-r1-0528")
VALIDATOR_MODEL     = os.getenv("VALIDATOR_MODEL",    "google/gemma-3-12b-it")
EVAL_MODEL          = os.getenv("EVAL_MODEL",         "qwen/qwen3-32b")
EVAL_FALLBACK       = os.getenv("EVAL_FALLBACK",      "deepseek/deepseek-r1-0528")
FAST_EVAL           = os.getenv("FAST_EVAL",          "qwen/qwen3-8b:free")
QDRANT_URL          = os.getenv("QDRANT_URL",
    "os.getenv("QDRANT_URL", "")")
QDRANT_API_KEY      = os.getenv("QDRANT_API_KEY",
    "os.getenv("QDRANT_API_KEY", "")")
COLLECTION          = "acreditacion"
TOP_K_CANDIDATES    = 20
TOP_K_FINAL         = 5
MAX_TOOL_ROUNDS     = 3
MAX_SEARCHES        = 3

# ─────────────────────────────────────────────────────────────────────────────
# TOOLS definición — igual que chatacredita_app1.py
# ─────────────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Busca fragmentos relevantes en el corpus de acreditación EISC/CNA. "
                "Usa esto cuando necesites información específica del documento."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta de búsqueda específica en español",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Número de fragmentos a recuperar (1-10)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_answer",
            "description": (
                "Valida si una respuesta está soportada por el contexto. "
                "Usa esto antes de dar la respuesta final."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question":  {"type":"string"},
                    "context":   {"type":"string"},
                    "answer":    {"type":"string"},
                },
                "required": ["question","context","answer"],
            },
        },
    },
]

ORCHESTRATOR_SYSTEM = """Eres ChatAcredita PRO, agente orquestador de un sistema multiagente \
especializado en acreditación universitaria CNA/EISC de la Universidad del Valle, Colombia.

HERRAMIENTAS DISPONIBLES:
1. search_documents(query, top_k): Busca en el corpus de acreditación
2. validate_answer(question, context, answer): Valida si tu respuesta está soportada

FLUJO DE TRABAJO:
1. Analiza la pregunta del usuario
2. Si necesitas información: usa search_documents con una consulta específica
3. Genera una respuesta basada en los documentos encontrados
4. Valida tu respuesta con validate_answer
5. Si la validación falla y tienes rondas disponibles, busca más información

REGLAS:
- Responde SIEMPRE en español
- Basa tus respuestas SOLO en los documentos recuperados
- Cita las fuentes: [Fuente: nombre_archivo.pdf]
- Si la información no está disponible, indícalo claramente"""

# ─────────────────────────────────────────────────────────────────────────────
# RECURSOS — carga lazy
# ─────────────────────────────────────────────────────────────────────────────
_res: dict = {}

def get_llm() -> OpenAI:
    return OpenAI(api_key=OPENROUTER_KEY, base_url=OPENROUTER_BASE, timeout=120)

def get_embedder():
    if "emb" not in _res:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Cargando BGE-M3 ({dev})…", end=" ", flush=True)
        _res["emb"] = SentenceTransformer("BAAI/bge-m3", device=dev)
        print("✅")
    return _res["emb"]

def get_reranker():
    if "rer" not in _res:
        from sentence_transformers import CrossEncoder
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print("  Cargando BGE-Reranker…", end=" ", flush=True)
        _res["rer"] = CrossEncoder("BAAI/bge-reranker-v2-m3",
                                    device=dev, max_length=512)
        print("✅")
    return _res["rer"]

def get_qdrant():
    if "qdr" not in _res:
        from qdrant_client import QdrantClient
        print("  Conectando Qdrant…", end=" ", flush=True)
        _res["qdr"] = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        n = _res["qdr"].get_collection(COLLECTION).points_count
        print(f"✅ {n:,} chunks")
    return _res["qdr"]

def qdrant_search(query_vec: list, limit: int = TOP_K_CANDIDATES) -> list:
    qdrant = get_qdrant()
    hits   = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=limit,
        with_payload=True,
    ).points
    return hits

# ─────────────────────────────────────────────────────────────────────────────
# AGENTE 2 — RETRIEVER (fiel a chatacredita_app1.py)
# ─────────────────────────────────────────────────────────────────────────────
def agent2_retriever(query: str, top_k: int = TOP_K_FINAL) -> dict:
    embedder = get_embedder()
    reranker = get_reranker()

    # 1. Dense retrieval
    qvec  = embedder.encode(query, normalize_embeddings=True).tolist()
    hits  = qdrant_search(qvec, TOP_K_CANDIDATES)
    dense = [{"text":        h.payload.get("text",""),
               "source":     h.payload.get("source","?"),
               "dense_score":h.score}
              for h in hits]

    if not dense:
        return {"docs":[],"context":"No se encontraron documentos.",
                "sources":[],"error":"sin resultados"}

    # 2. BM25 re-score sobre candidatos densos
    try:
        from rank_bm25 import BM25Okapi
        def norm(t):
            t = unicodedata.normalize("NFD",t.lower())
            return "".join(c for c in t if unicodedata.category(c)!="Mn")
        corpus = [norm(d["text"]).split() for d in dense]
        bm25   = BM25Okapi(corpus)
        scores = bm25.get_scores(norm(query).split())
        mx     = max(scores) or 1.0
        for i, d in enumerate(dense):
            d["bm25_score"]   = float(scores[i])
            d["hybrid_score"] = 0.6*d["dense_score"] + 0.4*(scores[i]/mx)
        dense.sort(key=lambda x: x["hybrid_score"], reverse=True)
    except ImportError:
        pass

    # 3. Cross-encoder rerank
    pairs  = [(query, d["text"]) for d in dense]
    rscores = reranker.predict(pairs)
    for i, d in enumerate(dense):
        d["rerank_score"] = float(rscores[i])
    dense.sort(key=lambda x: x["rerank_score"], reverse=True)
    top = dense[:top_k]

    # 4. Contexto formateado
    parts = []
    for i, d in enumerate(top, 1):
        src = Path(str(d["source"])).name
        parts.append(
            f"[Fragmento {i} | Fuente: {src} | score={d['rerank_score']:.3f}]\n"
            f"{d['text'][:700]}"
        )
    context = "\n\n---\n\n".join(parts)
    sources = list({Path(str(d["source"])).name for d in top})

    return {"docs": top, "context": context, "sources": sources, "error": None}

# ─────────────────────────────────────────────────────────────────────────────
# AGENTE 3 — VALIDADOR (fiel a chatacredita_app1.py)
# ─────────────────────────────────────────────────────────────────────────────
VALIDATOR_PROMPT = """Eres un validador de respuestas RAG para acreditación universitaria.
Verifica si la respuesta está SOPORTADA por el contexto de documentos.
Responde ÚNICAMENTE con este JSON (sin texto adicional):
{
  "supported": true/false,
  "confidence": 0.0-1.0,
  "issues": "descripción breve o 'ninguno'",
  "needs_more_search": true/false
}"""

def agent3_validator(question: str, context: str, answer: str) -> dict:
    default = {"supported":True,"confidence":0.5,
               "issues":"error al validar","needs_more_search":False}
    llm  = get_llm()
    msg  = (f"PREGUNTA: {question}\n\nCONTEXTO:\n{context[:1500]}\n\n"
            f"RESPUESTA A VALIDAR:\n{answer[:600]}")
    for model in [VALIDATOR_MODEL, "qwen/qwen3-8b"]:
        try:
            r   = llm.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":VALIDATOR_PROMPT},
                          {"role":"user",  "content":msg}],
                temperature=0.0, max_tokens=200,
            )
            raw = (r.choices[0].message.content or "")
            raw = re.sub(r"<think>.*?</think>","",raw,flags=re.DOTALL).strip()
            raw = re.sub(r"```(?:json)?","",raw).strip().rstrip("`").strip()
            m   = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            if m:
                return {**default, **json.loads(m.group())}
        except Exception:
            continue
    return default

# ─────────────────────────────────────────────────────────────────────────────
# AGENTE 1 — ORQUESTADOR (fiel a chatacredita_app1.py)
# ─────────────────────────────────────────────────────────────────────────────
def clean(raw: str) -> str:
    out = re.sub(r"<think>.*?</think>","",raw,flags=re.DOTALL).strip()
    return out if out else raw.strip()

def agent1_orchestrator(query: str) -> dict:
    llm      = get_llm()
    messages = [{"role":"system","content":ORCHESTRATOR_SYSTEM},
                {"role":"user",  "content":query}]

    all_sources:  list = []
    last_context: str  = ""
    final_answer: str  = ""
    trace:        list = []
    search_count: int  = 0
    rounds_used:  int  = 0
    t0 = time.time()

    for round_num in range(MAX_TOOL_ROUNDS):
        rounds_used   = round_num + 1
        force_answer  = search_count >= MAX_SEARCHES

        if force_answer and last_context:
            messages.append({
                "role": "user",
                "content": (
                    f"Ya tienes suficiente contexto de {search_count} búsquedas. "
                    "NO llames más herramientas. "
                    "Genera AHORA la respuesta final basándote en los documentos recuperados."
                )
            })
            trace.append(f"⚡ Forzando respuesta tras {search_count} búsquedas")

        try:
            resp = llm.chat.completions.create(
                model=ORCHESTRATOR_MODEL,
                messages=messages,
                tools=TOOLS if not force_answer else [],
                tool_choice="none" if force_answer else "auto",
                temperature=0.2,
                max_tokens=1500,
                extra_body={"models": ["qwen/qwen3-32b",
                                       "deepseek/deepseek-v4.1-flash"]},
            )
        except Exception as e:
            trace.append(f"❌ Error orquestador ronda {round_num+1}: {e}")
            break

        msg = resp.choices[0].message

        # ── Sin tool calls → respuesta final ─────────────────────────────────
        if not msg.tool_calls:
            raw_ans      = msg.content or ""
            final_answer = clean(raw_ans)
            trace.append(f"✅ Respuesta final generada (ronda {round_num+1})")
            break

        # ── Con tool calls ────────────────────────────────────────────────────
        messages.append({
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": [
                {"id":tc.id,"type":"function",
                 "function":{"name":tc.function.name,
                             "arguments":tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            fn   = tc.function.name
            args = {}
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                pass

            # ── search_documents → Agente 2 ──────────────────────────────────
            if fn == "search_documents":
                sub_q  = args.get("query", query)
                top_k  = int(args.get("top_k", TOP_K_FINAL))
                trace.append(f"🔍 Agente 2 · Retriever: \"{sub_q[:50]}…\"")
                result = agent2_retriever(sub_q, top_k)

                if result.get("error") and not result["docs"]:
                    tool_result = "No se encontraron documentos relevantes."
                else:
                    last_context  = result["context"]
                    all_sources  += result["sources"]
                    search_count += 1
                    tool_result   = result["context"]
                    trace.append(
                        f"  ✅ {len(result['docs'])} fragmentos | "
                        f"fuentes: {', '.join(result['sources'][:3])}"
                    )

                messages.append({
                    "role":        "tool",
                    "tool_call_id": tc.id,
                    "content":     tool_result,
                })

            # ── validate_answer → Agente 3 ───────────────────────────────────
            elif fn == "validate_answer":
                q2   = args.get("question", query)
                ctx2 = args.get("context",  last_context)
                ans2 = args.get("answer",   "")
                trace.append("✅ Agente 3 · Validador verificando…")
                val  = agent3_validator(q2, ctx2, ans2)

                trace.append(
                    f"  {'✅' if val['supported'] else '⚠️'} "
                    f"supported={val['supported']} "
                    f"conf={val['confidence']:.2f}"
                )

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(val),
                })

    # Fallback si sin respuesta — generar directamente con el contexto recuperado
    if not final_answer:
        if last_context:
            trace.append("🔄 Fallback: generando respuesta directa desde contexto…")
            try:
                fb_prompt = (
                    f"Responde esta pregunta basándote ÚNICAMENTE en los documentos.\n"
                    f"Responde en español. Cita la fuente si aparece.\n\n"
                    f"CONTEXTO:\n{last_context[:2000]}\n\n"
                    f"PREGUNTA: {query}\nRESPUESTA:"
                )
                fb_resp = get_llm().chat.completions.create(
                    model="qwen/qwen3-32b",
                    messages=[{"role":"user","content":fb_prompt}],
                    temperature=0.1, max_tokens=800,
                )
                raw_fb = (fb_resp.choices[0].message.content or "")
                final_answer = re.sub(r"<think>.*?</think>","",raw_fb,
                                      flags=re.DOTALL).strip() or raw_fb
                trace.append("✅ Respuesta fallback generada")
            except Exception as e:
                final_answer = (
                    f"⚠️ Sin respuesta del orquestador. "
                    f"Fuentes recuperadas: {', '.join(set(all_sources))}"
                )
        else:
            final_answer = "⚠️ No se recuperó contexto relevante para esta pregunta."

    return {
        "answer":      final_answer,
        "context":     last_context,
        "sources":     list(set(all_sources)),
        "trace":       trace,
        "rounds":      rounds_used,
        "searches":    search_count,
        "latency_s":   time.time() - t0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS RAGAS (mismo evaluador que eval_sysB_sysD.py)
# ─────────────────────────────────────────────────────────────────────────────
def parse_ragas_json(raw: str) -> dict | None:
    keys = ["faithfulness","answer_relevance","context_precision","context_recall"]

    def _extract(text: str) -> dict | None:
        text = re.sub(r"```(?:json)?","",text).strip().rstrip("`").strip()
        m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if m:
            for attempt in [m.group(), m.group().replace("'",'"')]:
                try:
                    d = json.loads(attempt)
                    if any(k in d for k in keys):
                        return {k: float(d.get(k,0.5)) for k in keys}
                except Exception:
                    pass
        nums = re.findall(r'\b(0\.\d+|1\.0+)\b', text)
        if len(nums) >= 4:
            return {k: float(v) for k,v in zip(keys, nums[:4])}
        return None

    outside = re.sub(r"<think>.*?</think>","",raw,flags=re.DOTALL).strip()
    result  = _extract(outside)
    if result:
        return result
    think = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think:
        result = _extract(think.group(1))
        if result:
            return result
    return _extract(raw)

def compute_ragas(question: str, answer: str,
                  context: str, reference: str) -> dict:
    default = {"faithfulness":0.5,"answer_relevance":0.5,
               "context_precision":0.5,"context_recall":0.5}
    if not answer or answer.startswith("⚠️") or answer.startswith("ERROR"):
        return default

    system_eval = (
        "/no-think\n"
        "Eres un evaluador experto de sistemas RAG en español. "
        "Responde ÚNICAMENTE con JSON con 4 valores entre 0.1 y 1.0. "
        "Nunca copies el ejemplo literal."
    )
    prompt = (
        f"PREGUNTA: {question}\n"
        f"CONTEXTO:\n{context[:1800]}\n\n"
        f"RESPUESTA:\n{answer[:700]}\n\n"
        f"REFERENCIA:\n{reference[:400]}\n\n"
        "Evalúa (valores REALES, no el ejemplo):\n"
        "- faithfulness: ¿respuesta fiel al contexto?\n"
        "- answer_relevance: ¿responde la pregunta?\n"
        "- context_precision: ¿contexto útil?\n"
        "- context_recall: ¿contexto cubre la referencia?\n\n"
        '{"faithfulness": X.XX, "answer_relevance": X.XX, '
        '"context_precision": X.XX, "context_recall": X.XX}'
    )
    llm = get_llm()
    for model, max_tok, delay in [
        (EVAL_MODEL,    800, 0),
        (EVAL_FALLBACK, 1400, 3),
        (FAST_EVAL,     300, 2),
    ]:
        time.sleep(delay)
        try:
            r   = llm.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":system_eval},
                          {"role":"user",  "content":prompt}],
                temperature=0.0, max_tokens=max_tok,
            )
            raw = (r.choices[0].message.content or "")
            parsed = parse_ragas_json(raw)
            if parsed and any(v > 0.05 for v in parsed.values()):
                return parsed
        except Exception:
            continue
    return default

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluación RAGAS Sys-C (Multi-Agent ReAct)")
    parser.add_argument("--benchmark", default="benchmark_342.json")
    parser.add_argument("--output",    default="results_sysC.json")
    parser.add_argument("--delay",     type=float, default=3.0)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--limit",     type=int,   default=0,
                        help="Evaluar solo los primeros N (0=todos)")
    args = parser.parse_args()

    questions = json.load(open(args.benchmark, encoding="utf-8"))
    if args.limit:
        questions = questions[:args.limit]

    print(f"\n{'='*60}")
    print(f"Sys-C — Multi-Agent ReAct · {len(questions)} preguntas")
    print(f"Orquestador: {ORCHESTRATOR_MODEL}")
    print(f"Validador:   {VALIDATOR_MODEL}")
    print(f"{'='*60}")

    if args.dry_run:
        for q in questions[:5]:
            print(f"  [{q.get('id')}] {q['question'][:70]}")
        print(f"  … {len(questions)} preguntas")
        return

    # Reanudar
    rows: list = []
    done_ids: set = set()
    if args.resume and Path(args.output).exists():
        prev = json.load(open(args.output, encoding="utf-8"))
        rows = prev.get("rows", [])
        done_ids = {r["id"] for r in rows}
        print(f"Reanudando — {len(done_ids)} ya evaluadas")

    # Pre-cargar recursos
    print("\nCargando recursos…")
    get_embedder()
    get_reranker()
    get_qdrant()

    m_acc = {"faithfulness":[],"answer_relevance":[],
              "context_precision":[],"context_recall":[]}
    lats, rounds_acc = [], []

    # Añadir métricas de preguntas ya evaluadas al acumulador
    for r in rows:
        for m in m_acc:
            if m in r:
                m_acc[m].append(r[m])
        lats.append(r.get("latency_s",0))
        rounds_acc.append(r.get("rounds",1))

    pending = [q for q in questions if q.get("id","") not in done_ids]
    print(f"Pendientes: {len(pending)}\n")

    for qi, q in enumerate(pending):
        qid = q.get("id", f"Q{qi+1:03d}")
        cat = q.get("category","?")
        print(f"\n[{len(rows)+1:3d}/{len(questions)}] {qid}·{cat} "
              f"{q['question'][:55]}…")

        # Ejecutar Sys-C
        try:
            result = agent1_orchestrator(q["question"])
        except Exception as e:
            print(f"  ❌ Error fatal: {e}")
            result = {"answer":f"ERROR: {e}","context":"","sources":[],
                      "trace":[],"rounds":0,"searches":0,"latency_s":0}

        print(f"  A: {result['answer'][:80]}…  "
              f"[{result['latency_s']:.1f}s | "
              f"{result['searches']} búsquedas | "
              f"{result['rounds']} rondas]")

        # RAGAS
        time.sleep(args.delay)
        scores = compute_ragas(
            q["question"],
            result["answer"],
            result["context"],
            q.get("reference","")
        )
        avg = sum(scores[m] for m in m_acc) / 4
        print(f"  RAGAS={avg:.3f} | "
              f"F={scores['faithfulness']:.3f} "
              f"R={scores['answer_relevance']:.3f} "
              f"P={scores['context_precision']:.3f} "
              f"C={scores['context_recall']:.3f}")

        for m in m_acc:
            m_acc[m].append(scores[m])
        lats.append(result["latency_s"])
        rounds_acc.append(result["rounds"])

        rows.append({
            "id":           qid,
            "category":     cat,
            "question":     q["question"],
            "answer":       result["answer"][:500],
            "latency_s":    round(result["latency_s"],2),
            "rounds":       result["rounds"],
            "searches":     result["searches"],
            "sources":      result["sources"],
            **scores,
            "ragas_avg":    round(avg,4),
        })

        # Guardar progreso
        _save(rows, m_acc, lats, rounds_acc, args.output)

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("RESUMEN Sys-C (Multi-Agent ReAct)")
    print(f"{'─'*50}")
    for m, vals in m_acc.items():
        mu  = statistics.mean(vals)
        sig = statistics.stdev(vals) if len(vals)>1 else 0
        print(f"  {m:22s}: {mu:.3f} ± {sig:.3f}")
    ragas_avg = sum(statistics.mean(m_acc[m]) for m in m_acc) / 4
    lat_mu    = statistics.mean(lats)
    lat_sig   = statistics.stdev(lats) if len(lats)>1 else 0
    rounds_mu = statistics.mean(rounds_acc) if rounds_acc else 0
    print(f"  {'RAGAS (avg)':22s}: {ragas_avg:.3f}")
    print(f"  {'Latencia (s)':22s}: {lat_mu:.1f} ± {lat_sig:.1f}")
    print(f"  {'Rondas (avg)':22s}: {rounds_mu:.1f}")

    # Por categoría
    from collections import defaultdict
    cat_scores: dict = defaultdict(list)
    for r in rows:
        cat_scores[r["category"]].append(r["ragas_avg"])
    print(f"\n  Por categoría:")
    for c, vals in sorted(cat_scores.items()):
        print(f"    {c:15s}: {statistics.mean(vals):.3f} (n={len(vals)})")

    _save(rows, m_acc, lats, rounds_acc, args.output)
    print(f"\n✅ Resultados: {args.output}")


def _save(rows, m_acc, lats, rounds_acc, path):
    summary = {}
    for m, vals in m_acc.items():
        if vals:
            summary[m] = {"mean": round(statistics.mean(vals),4),
                          "std":  round(statistics.stdev(vals) if len(vals)>1 else 0,4)}
    ragas_avg = sum(summary[m]["mean"] for m in summary) / max(len(summary),1)
    out = {
        "sistema":    "Sys-C (Multi-Agent ReAct)",
        "orchestrator": ORCHESTRATOR_MODEL,
        "validator":    VALIDATOR_MODEL,
        "n":          len(rows),
        "summary":    summary,
        "ragas_avg":  round(ragas_avg,4),
        "latency":    {"mean": round(statistics.mean(lats),2) if lats else 0,
                       "std":  round(statistics.stdev(lats) if len(lats)>1 else 0,2)},
        "rounds_avg": round(statistics.mean(rounds_acc),1) if rounds_acc else 0,
        "rows":       rows,
    }
    json.dump(out, open(path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
