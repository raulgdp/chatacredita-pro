# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ChatAcredita PRO — Arquitectura Multi-Agente                              ║
# ║  Agente 1: Orquestador (Claude Sonnet 4 / Qwen3-30B) con tool-calling     ║
# ║  Agente 2: Retriever  (BGE-M3 + Qdrant + BM25 + Reranker)                ║
# ║  Agente 3: Validador  (LLM pequeño — Qwen3-8B / Gemma)                   ║
# ║  Sin backend separado — todo en Streamlit                                  ║
# ║  EISC — Universidad del Valle, Cali, Colombia                               ║
# ║  Uso: streamlit run chatacredita_app.py                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import os, time, json, datetime, uuid, re
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# ── Cargar .env ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env", override=True)

# ── Modelos ───────────────────────────────────────────────────────────────────
ORCHESTRATOR_MODEL  = os.getenv("ORCHESTRATOR_MODEL",  "deepseek/deepseek-r1-0528")
#ORCHESTRATOR_MODEL  = os.getenv("ORCHESTRATOR_MODEL",  "deepseek/deepseek-r1-0528")
VALIDATOR_MODEL     = os.getenv("VALIDATOR_MODEL",     "google/gemma-3-12b-it:free")

OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY", "os.getenv("OPENROUTER_API_KEY", "")")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
QDRANT_URL      = os.getenv("QDRANT_URL", "os.getenv("QDRANT_URL", "")")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "os.getenv("QDRANT_API_KEY", "")")
COLLECTION      = os.getenv("QDRANT_COLLECTION", "acreditacion")

TOP_K_CANDIDATES = 20
TOP_K_FINAL      = 5
MAX_TOOL_ROUNDS  = 3
EMBED_MODEL      = "BAAI/bge-m3"
RERANK_MODEL     = "BAAI/bge-reranker-v2-m3"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatAcredita PRO · Multi-Agente",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS WhatsApp Dark
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}
.stApp{background:#111b21;}

.wa-header{background:#202c33;border-bottom:1px solid #2a3942;padding:10px 16px;
  display:flex;align-items:center;gap:12px;margin-bottom:8px;border-radius:8px 8px 0 0;}
.wa-avatar{width:40px;height:40px;border-radius:50%;
  background:linear-gradient(135deg,#00a884,#00856f);
  display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;}
.wa-header-name{font-size:1rem;font-weight:600;color:#e9edef;}
.wa-header-sub{font-size:0.72rem;color:#8696a0;margin-top:1px;}
.wa-status-dot{width:8px;height:8px;border-radius:50%;background:#00a884;
  display:inline-block;margin-right:4px;vertical-align:middle;}

.wa-row-user{display:flex;justify-content:flex-end;margin:3px 0;}
.wa-row-bot {display:flex;justify-content:flex-start;margin:3px 0;}

.wa-bubble-user{background:#005c4b;color:#e9edef;border-radius:7.5px 0 7.5px 7.5px;
  padding:7px 10px 5px 10px;max-width:78%;font-size:0.875rem;line-height:1.5;
  word-break:break-word;box-shadow:0 1px 0.5px rgba(11,20,26,.13);position:relative;}
.wa-bubble-user::after{content:'';position:absolute;top:0;right:-8px;
  border-width:8px 0 0 8px;border-style:solid;
  border-color:#005c4b transparent transparent transparent;}

.wa-bubble-bot{background:#202c33;color:#e9edef;border-radius:0 7.5px 7.5px 7.5px;
  padding:7px 10px 5px 10px;max-width:82%;font-size:0.875rem;line-height:1.5;
  word-break:break-word;box-shadow:0 1px 0.5px rgba(11,20,26,.13);position:relative;}
.wa-bubble-bot::after{content:'';position:absolute;top:0;left:-8px;
  border-width:8px 8px 0 0;border-style:solid;
  border-color:transparent #202c33 transparent transparent;}

.wa-sender{font-size:0.72rem;font-weight:600;color:#53bdeb;margin-bottom:2px;}
.wa-meta{display:flex;align-items:center;justify-content:flex-end;gap:4px;margin-top:3px;}
.wa-time{font-size:0.65rem;color:#8696a0;}
.wa-check{font-size:0.65rem;color:#53bdeb;}

.wa-sources{margin-top:5px;padding:4px 8px;background:rgba(0,0,0,0.25);
  border-radius:6px;border-left:2px solid #00a884;
  font-size:0.68rem;color:#8696a0;line-height:1.4;}

.wa-trace{margin-top:6px;padding:6px 8px;background:#182229;
  border-radius:6px;border:1px solid #2a3942;}
.wa-trace-title{font-size:0.68rem;color:#53bdeb;font-weight:600;margin-bottom:3px;}
.wa-trace-row{font-size:0.68rem;color:#8696a0;line-height:1.6;padding-left:4px;}
.wa-trace-row b{color:#e9edef;}

.wa-step{display:inline-block;background:#1a2f3a;border:1px solid #2a4a5a;
  color:#7ec8e3;font-size:0.67rem;padding:2px 7px;border-radius:10px;margin:1px;}
.wa-step.done{border-color:#00a884;color:#00a884;}
.wa-step.error{border-color:#e74c3c;color:#e74c3c;}

/* Informe de agentes */
.wa-report{margin-top:8px;padding:8px 10px;background:#0d1f27;
  border-radius:8px;border:1px solid #1e3a4a;}
.wa-report-title{font-size:0.7rem;color:#53bdeb;font-weight:700;
  margin-bottom:6px;letter-spacing:0.03em;}
.wa-report-row{display:flex;align-items:flex-start;gap:6px;
  margin-bottom:4px;font-size:0.68rem;line-height:1.5;}
.wa-report-icon{font-size:0.85rem;flex-shrink:0;margin-top:1px;}
.wa-report-label{color:#8696a0;min-width:80px;flex-shrink:0;}
.wa-report-val{color:#e9edef;}
.wa-report-badge{display:inline-block;background:#182229;border:1px solid #2a3942;
  color:#7ec8e3;font-size:0.63rem;padding:1px 5px;border-radius:6px;
  margin-left:4px;vertical-align:middle;}
.wa-report-badge.fallback{border-color:#f39c12;color:#f39c12;}
.wa-report-badge.ok{border-color:#00a884;color:#00a884;}
.wa-report-divider{border:none;border-top:1px solid #1e3a4a;margin:4px 0;}

.wa-date-sep{text-align:center;margin:8px 0 4px 0;}
.wa-date-sep span{background:#182229;color:#8696a0;font-size:0.7rem;
  padding:3px 10px;border-radius:8px;font-weight:500;}

section[data-testid="stSidebar"]{background:#111b21 !important;}
section[data-testid="stSidebar"] *{color:#e9edef;}
#MainMenu,footer,header{visibility:hidden;}
.stDeployButton{display:none;}
.stTextInput>div>div>input{background:#2a3942 !important;color:#e9edef !important;
  border:none !important;border-radius:24px !important;padding:10px 16px !important;
  font-size:0.9rem !important;}
.stTextInput>div>div>input::placeholder{color:#8696a0 !important;}
.stTextInput>div>div>input:focus{box-shadow:none !important;}
.stTextInput label{display:none !important;}
div[data-testid="column"]:last-child .stButton>button{
  background:#00a884 !important;color:white !important;border:none !important;
  border-radius:50% !important;width:42px !important;height:42px !important;
  padding:0 !important;font-size:1.1rem !important;}
div[data-testid="column"]:last-child .stButton>button:hover{background:#00856f !important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE RECURSOS (cacheados)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Cargando BGE-M3…")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=dev)

@st.cache_resource(show_spinner="⏳ Cargando reranker…")
def load_reranker():
    from sentence_transformers import CrossEncoder
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(RERANK_MODEL, device=dev, max_length=512)

@st.cache_resource(show_spinner="⏳ Conectando Qdrant…")
def load_qdrant():
    from qdrant_client import QdrantClient
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(":memory:")

@st.cache_resource(show_spinner="⏳ Iniciando OpenRouter…")
def load_llm():
    from openai import OpenAI
    return OpenAI(api_key=OPENROUTER_KEY, base_url=OPENROUTER_BASE,
                  max_retries=2, timeout=90)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: búsqueda compatible con todas las versiones de qdrant-client
# ─────────────────────────────────────────────────────────────────────────────
def qdrant_search(qdrant, collection: str, query_vec: list, limit: int) -> list:
    """
    Wrapper compatible con qdrant-client antiguo (search) y nuevo (query_points).
    Siempre retorna una lista de hits con .payload y .score.
    """
    # Versión nueva ≥ 1.7
    if hasattr(qdrant, "query_points"):
        result = qdrant.query_points(
            collection_name=collection,
            query=query_vec,
            limit=limit,
            with_payload=True,
        )
        return result.points

    # Versión antigua < 1.7
    if hasattr(qdrant, "search"):
        return qdrant.search(
            collection_name=collection,
            query_vector=query_vec,
            limit=limit,
            with_payload=True,
        )

    raise RuntimeError(
        f"QdrantClient no tiene 'search' ni 'query_points'. "
        f"Versión instalada: {getattr(qdrant, '__version__', 'desconocida')}. "
        f"Métodos disponibles: {[m for m in dir(qdrant) if 'search' in m or 'query' in m]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# AGENTE 2 — RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────
def agent2_retriever(query: str, top_k: int = TOP_K_FINAL) -> dict:
    embedder = load_embedder()
    reranker = load_reranker()
    qdrant   = load_qdrant()

    # 1. Dense retrieval — BGE-M3
    qvec = embedder.encode(query, normalize_embeddings=True).tolist()
    try:
        hits = qdrant_search(qdrant, COLLECTION, qvec, TOP_K_CANDIDATES)
        dense_docs = [
            {"text":        h.payload.get("text", ""),
             "source":      h.payload.get("source", "?"),
             "dense_score": h.score}
            for h in hits
        ]
    except Exception as e:
        return {"docs": [], "context": "", "sources": [], "error": str(e)}

    if not dense_docs:
        return {"docs": [], "context": "No se encontraron documentos relevantes.",
                "sources": [], "error": None}

    # 2. BM25 sobre candidatos densos (sparse re-score)
    try:
        from rank_bm25 import BM25Okapi
        corpus_tokens = [d["text"].lower().split() for d in dense_docs]
        bm25   = BM25Okapi(corpus_tokens)
        bm25_s = bm25.get_scores(query.lower().split())
        for i, d in enumerate(dense_docs):
            d["bm25_score"] = float(bm25_s[i])
        bm25_max = max(d["bm25_score"] for d in dense_docs) or 1.0
        for d in dense_docs:
            d["hybrid_score"] = 0.6 * d["dense_score"] + 0.4 * (d["bm25_score"] / bm25_max)
        dense_docs.sort(key=lambda x: x["hybrid_score"], reverse=True)
    except ImportError:
        pass  # BM25 opcional

    # 3. Reranking cruzado
    pairs  = [(query, d["text"]) for d in dense_docs]
    scores = reranker.predict(pairs)
    for i, d in enumerate(dense_docs):
        d["rerank_score"] = float(scores[i])
    dense_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_docs = dense_docs[:top_k]

    # 4. Construir contexto
    parts = []
    for i, d in enumerate(top_docs, 1):
        src = Path(str(d["source"])).name
        parts.append(
            f"[Fragmento {i} | Fuente: {src} | score={d['rerank_score']:.3f}]\n"
            f"{d['text'][:700]}"
        )
    context = "\n\n---\n\n".join(parts)

    return {
        "docs":    top_docs,
        "context": context,
        "sources": list({Path(str(d["source"])).name for d in top_docs}),
        "error":   None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENTE 3 — VALIDADOR
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
    llm = load_llm()
    prompt = (
        f"PREGUNTA: {question}\n\n"
        f"CONTEXTO:\n{context[:3000]}\n\n"
        f"RESPUESTA:\n{answer}\n\n"
        "¿La respuesta está soportada por el contexto?"
    )
    try:
        resp = llm.chat.completions.create(
            model=VALIDATOR_MODEL,             # Principal: gemma-3-12b-it:free
            messages=[
                {"role": "system", "content": VALIDATOR_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            extra_body={
                "models": [
                    "meta-llama/llama-3.1-8b-instruct:free",  # 2do: free, rápido
                    "qwen/qwen3-8b",                           # 3ro: pequeño con thinking
                ]
            },
        )
        raw   = resp.choices[0].message.content or "{}"
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"supported": True, "confidence": 0.5,
                "issues": "No se pudo parsear", "needs_more_search": False}
    except Exception as e:
        return {"supported": True, "confidence": 0.5,
                "issues": str(e), "needs_more_search": False}


# ─────────────────────────────────────────────────────────────────────────────
# AGENTE 1 — ORQUESTADOR
# ─────────────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Busca fragmentos relevantes en la base de conocimiento de acreditación CNA/EISC. "
                "Usa esta herramienta cuando necesites información sobre: factores CNA, "
                "características, indicadores, estadísticas EISC, programas académicos, "
                "investigación, bienestar, extensión, o cualquier dato del proceso de acreditación."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query de búsqueda optimizada para recuperación semántica.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Número de fragmentos a recuperar (3-8, default 5).",
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
                "Valida si una respuesta generada está soportada por los documentos recuperados. "
                "Usa esta herramienta DESPUÉS de generar una respuesta para verificar coherencia."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "La respuesta que se quiere validar.",
                    },
                    "context": {
                        "type": "string",
                        "description": "El contexto de documentos que respalda la respuesta.",
                    },
                },
                "required": ["answer", "context"],
            },
        },
    },
]

ORCHESTRATOR_SYSTEM = """Eres ChatAcredita PRO, agente orquestador de un sistema multiagente \
de acreditación universitaria para el CNA de Colombia, especializado en la EISC \
de la Universidad del Valle.

HERRAMIENTAS DISPONIBLES:
- search_documents: busca en la base de conocimiento (79,879 chunks indexados)
- validate_answer: verifica coherencia de tu respuesta con los documentos

ESTRATEGIA:
1. Analiza la pregunta.
2. Llama a search_documents con una query precisa.
3. Si el resultado es insuficiente, reformula y busca de nuevo (máx 2 búsquedas).
4. Genera la respuesta basándote en los documentos.
5. Opcionalmente valida con validate_answer.
6. Entrega la respuesta final en español, clara y estructurada.

REGLAS:
- Basa tus respuestas EXCLUSIVAMENTE en los documentos recuperados.
- Si no hay información suficiente, dilo claramente.
- Cita fuentes con [Fuente: nombre_documento].
- Responde siempre en español."""


def agent1_orchestrator(query: str, history: list, trace_placeholder) -> dict:
    llm = load_llm()

    messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM}]
    for m in history[-6:]:
        role = "assistant" if m["role"] == "assistant" else "user"
        messages.append({"role": role, "content": m["text"]})
    messages.append({"role": "user", "content": query})

    trace          = []
    all_sources    = []
    last_context   = ""
    final_answer   = ""
    round_num_used = 0
    t0             = time.time()
    # Informe de modelos que actuaron en esta consulta
    agent_report   = {
        "orquestador": {"modelo": ORCHESTRATOR_MODEL, "modelo_real": ORCHESTRATOR_MODEL, "rondas": 0, "tokens_in": 0, "tokens_out": 0},
        "retriever":   {"modelo": f"BGE-M3 + Reranker", "invocaciones": 0, "chunks": 0},
        "validador":   {"modelo": VALIDATOR_MODEL, "modelo_real": None, "invocado": False, "resultado": None},
    }

    def update_trace():
        rows = "".join(f'<div class="wa-trace-row">{r}</div>' for r in trace)
        trace_placeholder.markdown(
            f'<div class="wa-row-bot">'
            f'<div class="wa-bubble-bot" style="min-width:260px;">'
            f'<div class="wa-sender">🎓 ChatAcredita PRO · Multi-Agente</div>'
            f'<div class="wa-trace">'
            f'<div class="wa-trace-title">🔄 Razonamiento en curso…</div>'
            f'{rows}'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

    for round_num in range(MAX_TOOL_ROUNDS):
        trace.append(f'<b>Ronda {round_num+1}/{MAX_TOOL_ROUNDS}</b> — consultando orquestador…')
        update_trace()

        try:
            resp = llm.chat.completions.create(
                model=ORCHESTRATOR_MODEL,       # Principal: deepseek-r1-0528 (thinking + tools)
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1200,
                extra_body={
                    # Fallback automático si el modelo principal falla
                    # (rate-limit, downtime, rechazo de moderación)
                    # OpenRouter prueba en orden hasta que uno responde
                    "models": [
                        "qwen/qwen3-32b",              # 2do: thinking + tool-calling
                        "deepseek/deepseek-v4-flash",   # 3ro: mejor tool-calling del mercado
                        "deepseek/deepseek-v4.1-flash" # 4to: rápido, último recurso
                    ]
                },
            )
        except Exception as e:
            trace.append(f'<span class="wa-step error">❌ Error LLM: {e}</span>')
            update_trace()
            final_answer = f"⚠️ Error del orquestador: {e}"
            break

        msg            = resp.choices[0].message
        round_num_used = round_num
        # Capturar el modelo real usado (puede ser un fallback)
        agent_report["orquestador"]["modelo_real"] = getattr(resp, "model", ORCHESTRATOR_MODEL)
        agent_report["orquestador"]["rondas"]      = round_num + 1
        usage = getattr(resp, "usage", None)
        if usage:
            agent_report["orquestador"]["tokens_in"]  += getattr(usage, "prompt_tokens", 0)
            agent_report["orquestador"]["tokens_out"] += getattr(usage, "completion_tokens", 0)

        if msg.tool_calls:
            messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except Exception:
                    fn_args = {}

                # ── TOOL: search_documents → Agente 2 ────────────────────────
                if fn_name == "search_documents":
                    q2    = fn_args.get("query", query)
                    top_k = fn_args.get("top_k", TOP_K_FINAL)
                    trace.append(
                        f'<span class="wa-step">🔍 Agente 2 · Retriever</span> '
                        f'query: <i>"{q2[:60]}…"</i>'
                    )
                    update_trace()

                    result2 = agent2_retriever(q2, top_k=top_k)

                    if result2["error"]:
                        tool_result = f"Error en búsqueda: {result2['error']}"
                        trace.append(
                            f'<span class="wa-step error">❌ Retriever falló: {result2["error"]}</span>'
                        )
                    else:
                        tool_result  = result2["context"]
                        last_context = result2["context"]
                        n_docs       = len(result2["docs"])
                        all_sources += result2.get("sources", [])
                        agent_report["retriever"]["invocaciones"] += 1
                        agent_report["retriever"]["chunks"]       += n_docs
                        trace.append(
                            f'<span class="wa-step done">✅ Retriever</span> '
                            f'{n_docs} fragmentos · '
                            f'fuentes: {", ".join(result2.get("sources", [])[:3])}'
                        )
                    update_trace()

                # ── TOOL: validate_answer → Agente 3 ─────────────────────────
                elif fn_name == "validate_answer":
                    ans_to_val = fn_args.get("answer", "")
                    ctx_to_val = fn_args.get("context", last_context)
                    trace.append(
                        f'<span class="wa-step">✅ Agente 3 · Validador</span> '
                        f'verificando coherencia…'
                    )
                    update_trace()

                    val         = agent3_validator(query, ctx_to_val, ans_to_val)
                    tool_result = json.dumps(val, ensure_ascii=False)
                    icon        = "✅" if val.get("supported") else "⚠️"
                    conf        = val.get("confidence", 0)
                    issue       = val.get("issues", "")
                    agent_report["validador"]["invocado"]  = True
                    agent_report["validador"]["resultado"] = val
                    trace.append(
                        f'<span class="wa-step done">{icon} Validador</span> '
                        f'soportada={val.get("supported")} · '
                        f'confianza={conf:.0%} · {issue[:60]}'
                    )
                    update_trace()

                else:
                    tool_result = f"Herramienta desconocida: {fn_name}"

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      str(tool_result)[:6000],
                })

        else:
            import re as _re
            raw = msg.content or ""
            # DeepSeek R1 / Qwen3 con thinking meten razonamiento en <think>...</think>
            # La respuesta real viene después del bloque o en reasoning_content
            if not raw.strip():
                raw = getattr(msg, "reasoning_content", "") or ""
            clean = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
            if not clean:
                clean = _re.sub(r"</?think>", "", raw).strip()
            final_answer = clean
            trace.append(
                f'<span class="wa-step done">✍️ Respuesta generada</span> '
                f'rondas: {round_num+1} · {time.time()-t0:.1f}s'
            )
            update_trace()
            break

    if not final_answer:
        final_answer = (
            "He buscado en los documentos disponibles pero no encontré "
            "información suficiente. Verifica que los PDFs estén indexados."
        )
        trace.append('<span class="wa-step error">⚠️ Rondas agotadas sin respuesta final</span>')
        update_trace()

    agent_report["orquestador"]["rondas"] = round_num_used + 1
    return {
        "answer":       final_answer,
        "trace":        trace,
        "sources":      list(set(all_sources)),
        "latency_s":    time.time() - t0,
        "rounds":       round_num_used + 1,
        "context":      last_context,
        "agent_report": agent_report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDEXADO DE PDFs
# ─────────────────────────────────────────────────────────────────────────────
def index_pdf(pdf_bytes: bytes, filename: str) -> int:
    import fitz
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from qdrant_client.models import VectorParams, Distance, PointStruct

    embedder = load_embedder()
    qdrant   = load_qdrant()

    try:
        qdrant.get_collection(COLLECTION)
    except Exception:
        qdrant.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    doc     = fitz.open(stream=pdf_bytes, filetype="pdf")
    text    = "\n".join(page.get_text() for page in doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks  = splitter.split_text(text)

    points = []
    for chunk in chunks:
        vec = embedder.encode(chunk, normalize_embeddings=True).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()), vector=vec,
            payload={"text": chunk, "source": filename},
        ))
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    for k, v in {"messages": [], "pending_query": ""}.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def ts_now() -> str:
    return datetime.datetime.now().strftime("%H:%M")


# ─────────────────────────────────────────────────────────────────────────────
# RENDER BURBUJAS
# ─────────────────────────────────────────────────────────────────────────────
def render_user(msg: dict):
    st.markdown(
        f'<div class="wa-row-user"><div class="wa-bubble-user">{msg["text"]}'
        f'<div class="wa-meta"><span class="wa-time">{msg.get("ts","")}</span>'
        f'<span class="wa-check">✓✓</span></div></div></div>',
        unsafe_allow_html=True,
    )

def render_bot(msg: dict):
    src_html = ""
    if msg.get("sources"):
        srcs = " · ".join(f"📄 {s}" for s in msg["sources"])
        src_html = f'<div class="wa-sources">{srcs}</div>'

    trace_html = ""
    if msg.get("trace"):
        rows = "".join(f'<div class="wa-trace-row">{r}</div>' for r in msg["trace"])
        lat  = msg.get("latency_s", 0)
        rds  = msg.get("rounds", 1)
        trace_html = (
            f'<div class="wa-trace">'
            f'<div class="wa-trace-title">🔄 Traza · {rds} ronda(s) · {lat:.1f}s</div>'
            f'{rows}</div>'
        )

    lat_info = ""
    if msg.get("latency_s"):
        lat_info = f' · {msg["latency_s"]:.1f}s · {msg.get("rounds",1)} ronda(s)'

    # ── Informe de modelos ────────────────────────────────────────────────────
    report_html = ""
    ar = msg.get("agent_report")
    if ar:
        def _badge(texto, tipo=""):
            return f'<span class="wa-report-badge {tipo}">{texto}</span>'

        # Agente 1 — Orquestador
        orch      = ar["orquestador"]
        m_cfg     = orch["modelo"].split("/")[-1]
        m_real    = orch["modelo_real"].split("/")[-1]
        fallback  = m_cfg != m_real
        tok_in    = orch.get("tokens_in", 0)
        tok_out   = orch.get("tokens_out", 0)
        fb_badge  = _badge("fallback activado", "fallback") if fallback else _badge("modelo principal", "ok")
        orch_row  = (
            f'<div class="wa-report-row">'
            f'<span class="wa-report-icon">🧠</span>'
            f'<span class="wa-report-label">Orquestador</span>'
            f'<span class="wa-report-val">{m_real}{fb_badge}</span>'
            f'</div>'
            f'<div class="wa-report-row">'
            f'<span class="wa-report-icon" style="opacity:0">·</span>'
            f'<span class="wa-report-label"></span>'
            f'<span class="wa-report-val" style="color:#8696a0">'
            f'rondas: {orch["rondas"]} · tokens: {tok_in}↑ {tok_out}↓</span>'
            f'</div>'
        )

        # Agente 2 — Retriever
        retr     = ar["retriever"]
        inv      = retr["invocaciones"]
        chunks   = retr["chunks"]
        retr_row = (
            f'<div class="wa-report-row">'
            f'<span class="wa-report-icon">🔍</span>'
            f'<span class="wa-report-label">Retriever</span>'
            f'<span class="wa-report-val">BGE-M3 + BM25 + Reranker</span>'
            f'</div>'
            f'<div class="wa-report-row">'
            f'<span class="wa-report-icon" style="opacity:0">·</span>'
            f'<span class="wa-report-label"></span>'
            f'<span class="wa-report-val" style="color:#8696a0">'
            f'invocaciones: {inv} · chunks recuperados: {chunks}</span>'
            f'</div>'
        ) if inv > 0 else (
            f'<div class="wa-report-row">'
            f'<span class="wa-report-icon">🔍</span>'
            f'<span class="wa-report-label">Retriever</span>'
            f'<span class="wa-report-val" style="color:#8696a0">no invocado</span>'
            f'</div>'
        )

        # Agente 3 — Validador
        vald = ar["validador"]
        if vald["invocado"] and vald["resultado"]:
            val_r   = vald["resultado"]
            sup     = "✅ soportada" if val_r.get("supported") else "⚠️ no soportada"
            conf    = val_r.get("confidence", 0)
            m_val   = vald["modelo"].split("/")[-1]
            val_row = (
                f'<div class="wa-report-row">'
                f'<span class="wa-report-icon">✅</span>'
                f'<span class="wa-report-label">Validador</span>'
                f'<span class="wa-report-val">{m_val}</span>'
                f'</div>'
                f'<div class="wa-report-row">'
                f'<span class="wa-report-icon" style="opacity:0">·</span>'
                f'<span class="wa-report-label"></span>'
                f'<span class="wa-report-val" style="color:#8696a0">'
                f'{sup} · confianza: {conf:.0%}</span>'
                f'</div>'
            )
        else:
            val_row = (
                f'<div class="wa-report-row">'
                f'<span class="wa-report-icon">✅</span>'
                f'<span class="wa-report-label">Validador</span>'
                f'<span class="wa-report-val" style="color:#8696a0">no invocado</span>'
                f'</div>'
            )

        report_html = (
            f'<div class="wa-report">'
            f'<div class="wa-report-title">📊 Informe de agentes</div>'
            f'{orch_row}'
            f'<hr class="wa-report-divider">'
            f'{retr_row}'
            f'<hr class="wa-report-divider">'
            f'{val_row}'
            f'</div>'
        )

    st.markdown(
        f'<div class="wa-row-bot"><div class="wa-bubble-bot">'
        f'<div class="wa-sender">🎓 ChatAcredita PRO · Multi-Agente</div>'
        f'{trace_html}{msg["text"]}{src_html}{report_html}'
        f'<div class="wa-meta"><span class="wa-time">{msg.get("ts","")}{lat_info}</span>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 ChatAcredita PRO")
    st.caption("Arquitectura Multi-Agente · CNA · EISC · Univalle")
    st.divider()

    st.markdown("### 🤖 Agentes activos")
    st.markdown(f"""
<div style="background:#182229;border-radius:8px;padding:10px;font-size:0.75rem;
color:#e9edef;line-height:1.8;">
<b style="color:#53bdeb;">Agente 1 · Orquestador</b><br>
<span style="color:#8696a0;">└ {ORCHESTRATOR_MODEL.split("/")[-1]}</span><br>
<span style="color:#8696a0;">└ tool_calling · max {MAX_TOOL_ROUNDS} rondas</span><br><br>
<b style="color:#53bdeb;">Agente 2 · Retriever</b><br>
<span style="color:#8696a0;">└ BGE-M3 + BM25 + Reranker</span><br>
<span style="color:#8696a0;">└ Top-{TOP_K_CANDIDATES}→rerank→Top-{TOP_K_FINAL}</span><br><br>
<b style="color:#53bdeb;">Agente 3 · Validador</b><br>
<span style="color:#8696a0;">└ {VALIDATOR_MODEL.split("/")[-1]}</span><br>
<span style="color:#8696a0;">└ coherencia respuesta↔docs</span>
</div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔌 Estado")
    if OPENROUTER_KEY:
        st.success("✅ OpenRouter OK")
    else:
        st.error("❌ Falta OPENROUTER_API_KEY")
    if QDRANT_URL:
        st.success("✅ Qdrant Cloud")
    else:
        st.warning("⚠️ Qdrant local")

    if st.button("📊 Stats colección"):
        try:
            info = load_qdrant().get_collection(COLLECTION)
            st.metric("Chunks indexados", f"{info.points_count:,}")
        except Exception as e:
            st.warning(f"Error: {e}")

    st.divider()
    st.markdown("### ⚙️ Modelos")
    ORCHESTRATOR_MODEL = st.selectbox("Orquestador (Agente 1)", [
        "deepseek/deepseek-v4.1-flash",          # rápido, tool-calling nativo
        "deepseek/deepseek-r1-0528",             # con thinking
        "qwen/qwen3-235b-a22b",                  # Qwen3 MoE
        "qwen/qwen3-32b",                        # Qwen3 denso
        "anthropic/claude-sonnet-4",             # Claude con extended thinking
        "google/gemini-2.5-flash-preview",       # Gemini flash
        "meta-llama/llama-3.3-70b-instruct",    # Llama 70B
    ], index=0)
    VALIDATOR_MODEL = st.selectbox("Validador (Agente 3)", [
        "google/gemma-3-12b-it:free",
        "qwen/qwen3-8b",
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-3.1-8b-instruct:free",
    ], index=0)
    MAX_TOOL_ROUNDS = st.slider("Máx. rondas agénticas", 1, 5, 3)

    st.divider()
    st.markdown("### 📄 Indexar documentos")
    pdfs = st.file_uploader("PDFs de acreditación", type=["pdf"],
                            accept_multiple_files=True,
                            label_visibility="collapsed")
    if pdfs and st.button("⬆️ Indexar", type="primary"):
        total = 0
        for pdf in pdfs:
            with st.spinner(f"Indexando {pdf.name}…"):
                try:
                    n = index_pdf(pdf.read(), pdf.name)
                    total += n
                    st.success(f"✅ {pdf.name}: {n} chunks")
                except Exception as e:
                    st.error(f"❌ {pdf.name}: {e}")
        if total:
            st.balloons()

    st.divider()
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.pending_query = ""
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# CHAT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="wa-header">
  <div class="wa-avatar">🎓</div>
  <div style="flex:1">
    <div class="wa-header-name">ChatAcredita PRO · Multi-Agente</div>
    <div class="wa-header-sub">
      <span class="wa-status-dot"></span>
      Orquestador · Retriever · Validador · en línea
    </div>
  </div>
  <div style="color:#8696a0;font-size:0.7rem;text-align:right;">EISC · Univalle · CNA</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f'<div class="wa-date-sep"><span>'
    f'{datetime.date.today().strftime("%d %b %Y")}'
    f'</span></div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        '<div class="wa-row-bot"><div class="wa-bubble-bot">'
        '<div class="wa-sender">🎓 ChatAcredita PRO · Multi-Agente</div>'
        '<div class="wa-trace">'
        '<div class="wa-trace-title">🤖 Sistema Multi-Agente activo</div>'
        '<div class="wa-trace-row">Agente 1 · Orquestador → decide cuándo y qué buscar</div>'
        '<div class="wa-trace-row">Agente 2 · Retriever   → BGE-M3 + BM25 + Reranker</div>'
        '<div class="wa-trace-row">Agente 3 · Validador  → verifica coherencia respuesta↔docs</div>'
        '</div>'
        'Hola 👋 Sube tus PDFs en el panel izquierdo y hazme preguntas sobre CNA · EISC.'
        '<div class="wa-meta"><span class="wa-time">ahora</span></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user(msg)
    else:
        render_bot(msg)

# ── Procesamiento pendiente ───────────────────────────────────────────────────
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = ""

    trace_placeholder = st.empty()
    result = agent1_orchestrator(
        query=query,
        history=st.session_state.messages,
        trace_placeholder=trace_placeholder,
    )
    trace_placeholder.empty()

    st.session_state.messages.append({
        "role":         "assistant",
        "text":         result["answer"],
        "ts":           ts_now(),
        "sources":      result["sources"],
        "trace":        result["trace"],
        "latency_s":    result["latency_s"],
        "rounds":       result["rounds"],
        "agent_report": result.get("agent_report"),
    })
    st.rerun()

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
col_inp, col_btn = st.columns([9, 1])
with col_inp:
    query_input = st.text_input(
        "msg", placeholder="Pregunta sobre acreditación CNA · EISC…",
        label_visibility="collapsed", key="query_input",
    )
with col_btn:
    send = st.button("➤", key="send_btn")

if send and query_input.strip():
    q = query_input.strip()
    st.session_state.messages.append({"role": "user", "text": q, "ts": ts_now()})
    st.session_state.pending_query = q
    st.rerun()
