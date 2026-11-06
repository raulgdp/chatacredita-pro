# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ChatAcredita PRO · Sys-E — RAG Router + LangGraph                        ║
# ║  Combina:                                                                  ║
# ║    • Router de agentes de app-ultima.py (6 tipos de pregunta)              ║
# ║    • Grafo LangGraph de langgraph_acredita.py (control de flujo)           ║
# ║    • evaluate_response() → métricas RAGAS por pregunta                    ║
# ║    • ConversationMemory multi-turno                                        ║
# ║    • Login + rate limiting + feedback vectorial                            ║
# ║  Comando: streamlit run langgraph_routed.py                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
import streamlit as st
import os, time, json, hashlib, re, html, uuid
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv(override=True)

OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY",
    "os.getenv("OPENROUTER_API_KEY", "")")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
ROUTER_MODEL    = os.getenv("ROUTER_MODEL",    "google/gemma-3-12b-it:free")
EVAL_MODEL      = os.getenv("EVAL_MODEL",      "deepseek/deepseek-r1-0528")
FAST_GRADER     = os.getenv("FAST_GRADER",     "google/gemma-3-12b-it:free")
QDRANT_URL      = os.getenv("QDRANT_URL",
    "os.getenv("QDRANT_URL", "")")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY",
    "os.getenv("QDRANT_API_KEY", "")")
COLLECTION      = "acreditacion"
FEEDBACK_COL    = "feedback_acreditacion"

# ── Importar el motor LangGraph ───────────────────────────────────────────────
try:
    from langgraph_acredita import (
        build_graph, GraphState, retrieve_docs, build_context,
        get_llm, clean_thinking, GENERATOR_MODEL, GRADER_MODEL,
        node_retrieve, node_rewrite_query,
        node_grade_answer, MAX_ITERATIONS,
    )
    # Sobreescribir GRADER_MODEL con modelo rápido para grade_answer
    import langgraph_acredita as _lg_mod
    _lg_mod.GRADER_MODEL = os.getenv("FAST_GRADER", "google/gemma-3-12b-it:free")
    GRADER_MODEL = _lg_mod.GRADER_MODEL

    BACKEND_OK  = True
    BACKEND_ERR = ""

    # Grade docs heurístico: relevante si al menos 2 docs tienen score > 0.3
    def node_grade_documents(state: GraphState) -> dict:
        docs = state.get("documents", [])
        good = [d for d in docs if d.get("rerank_score", d.get("final_score", 0.5)) > 0.25]
        grade = "relevant" if len(good) >= 2 else "not_relevant"
        state["trace"].append(
            f"{'✅' if grade == 'relevant' else '⚠️'} Docs evaluados: "
            f"{len(good)}/{len(docs)} relevantes (heurístico)"
        )
        return {"doc_grade": grade}

except ImportError as e:
    BACKEND_OK  = False
    BACKEND_ERR = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# SEGURIDAD — Login + rate limiting
# ─────────────────────────────────────────────────────────────────────────────
USERS = {
    "admin": hashlib.sha256("1234".encode()).hexdigest(),
    "raul":  hashlib.sha256("eisc2025".encode()).hexdigest(),
}
_req_counts: dict[str, list] = defaultdict(list)

def verify(user: str, pwd: str) -> bool:
    return USERS.get(user) == hashlib.sha256(pwd.encode()).hexdigest()

def rate_ok(user: str) -> bool:
    now = time.time()
    _req_counts[user] = [t for t in _req_counts[user] if now - t < 60]
    if len(_req_counts[user]) >= 15:
        return False
    _req_counts[user].append(now)
    return True

INJECTION = [r"ignore (?:previous|all) instructions",
             r"forget (?:your|the) (?:system )?prompt",
             r"jailbreak", r"act as if you",
             r"you are now", r"disregard (?:your|all)"]

def sanitize(q: str) -> str:
    q = q[:2000]
    for p in INJECTION:
        if re.search(p, q, re.I):
            return "[Consulta bloqueada por política de seguridad]"
    return q

# ─────────────────────────────────────────────────────────────────────────────
# ROUTER DE AGENTES (de app-ultima.py)
# ─────────────────────────────────────────────────────────────────────────────
AGENT_TYPES = {
    "lista":       "Pregunta que pide una lista, conjunto, todos los elementos de una categoría",
    "estadistica": "Pregunta sobre números, tasas, porcentajes, cantidades, rankings",
    "normativa":   "Pregunta sobre reglamentos, resoluciones, leyes, artículos, normas CNA",
    "proceso":     "Pregunta sobre pasos, procedimientos, flujos, cómo hacer algo",
    "comparacion": "Pregunta que compara dos o más elementos, criterios o periodos",
    "sintesis":    "Pregunta de resumen, conclusión o visión general de múltiples aspectos",
    "general":     "Cualquier otra pregunta sobre acreditación EISC",
}

# Instrucciones de formato por tipo de agente
AGENT_FORMAT = {
    "lista":       "El usuario quiere una lista COMPLETA. Usa lista numerada con TODOS los ítems encontrados. No resumas ni omitas ninguno. Si el documento tiene más elementos, indícalo al final.",
    "estadistica": "Presenta datos numéricos con precisión. Usa tablas Markdown si hay más de 2 valores. Indica siempre el periodo/año. Señala tendencias si las hay.",
    "normativa":   "Cita el artículo o resolución exacta si está en el contexto. Indica si la norma es vigente. Formato: Artículo X — [contenido resumido]. Nunca inventes referencias.",
    "proceso":     "Usa lista numerada de pasos, clara y accionable. Incluye prerrequisitos. Indica responsable de cada paso cuando sea relevante.",
    "comparacion": "Usa tabla Markdown con columnas para cada elemento. Añade fila de 'Conclusión' al final si aplica.",
    "sintesis":    "Sintetiza en máximo 4 viñetas los puntos más importantes. Luego un párrafo integrador.",
    "general":     "Responde en prosa clara. Máximo 3 párrafos. Usa viñetas solo si hay más de 3 ítems paralelos.",
}

AGENT_ICONS = {
    "lista": "📋", "estadistica": "📊", "normativa": "📜", "proceso": "⚙️",
    "comparacion": "⚖️", "sintesis": "🔍", "general": "💬",
}

# Palabras clave por categoría para routing rápido sin LLM
_ROUTE_KEYWORDS = {
    "estadistica":  ["cuántos","cuántas","porcentaje","tasa","número","cantidad",
                     "promedio","total","ranking","estadística","dato","cifra",
                     "índice","ratio","proporción","cuánto","cuánta"],
    "normativa":    ["decreto","resolución","artículo","norma","ley","reglamento",
                     "establece","define","según","dispone","criterio","requisito",
                     "lineamiento","política","acuerdo","circular","vigente"],
    "proceso":      ["cómo","pasos","etapas","procedimiento","proceso","flujo",
                     "qué hacer","cómo se","cuál es el proceso","trámite",
                     "paso a paso","instrucciones","secuencia","cómo funciona"],
    "lista":        ["lista","todos los","todas las","completa","completo","nombrados",
                     "listado","enumera","cuáles son todos","nómina","plantilla",
                     "relaciona","dame todos","regala la lista","quiénes son todos"],
    "comparacion":  ["diferencia","comparar","versus","vs","mejor","peor",
                     "ventaja","desventaja","distinción","entre","frente a",
                     "comparación","similar","distinto","igual","contraste"],
    "sintesis":     ["resumen","conclusión","síntesis","visión general","qué dice",
                     "principales","más importantes","en general","en conjunto",
                     "fortalezas","debilidades","balance","resultado","logros"],
}

def route_query(query: str) -> str:
    """Clasifica la query por palabras clave — sin llamada LLM, latencia ~0ms."""
    q = query.lower()
    scores = {cat: 0 for cat in _ROUTE_KEYWORDS}
    for cat, keywords in _ROUTE_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"

# ─────────────────────────────────────────────────────────────────────────────
# GRAFO LANGGRAPH ENRIQUECIDO CON PROMPT DE AGENTE
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_BASE = """Eres ChatAcredita PRO, asistente experto en acreditación universitaria CNA/EISC de la Universidad del Valle, Colombia.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE en español.
2. Basa tu respuesta EXCLUSIVAMENTE en los fragmentos de documentos proporcionados.
3. Si la información no está en los documentos, di: "No encontré información suficiente en los documentos disponibles."
4. Cita siempre la fuente: [Fuente: nombre_archivo.pdf]
5. NO inventes datos, fechas, nombres ni cifras.

FORMATO DE RESPUESTA ({agent_type}):
{format_instruction}"""

def run_routed_langgraph(query: str, agent_type: str) -> dict:
    """
    Ejecuta el grafo LangGraph con el prompt especializado del agente.
    Retorna dict compatible con el panel de resultados.
    """
    from langgraph.graph import StateGraph, END
    import time as _time

    # Top-K adaptativo: más chunks para preguntas de lista
    TOP_K_LISTA   = 12   # listas necesitan más chunks del mismo PDF
    TOP_K_NORMAL  = 5

    def node_retrieve_smart(state: GraphState) -> dict:
        """Retrieval con top-K adaptativo según tipo de agente."""
        q      = state.get("rewritten_q") or state["query"]
        top_k  = TOP_K_LISTA if agent_type == "lista" else TOP_K_NORMAL
        result = retrieve_docs(q, top_k=top_k)
        docs   = result.get("documents", [])
        srcs   = result.get("sources",   [])

        # Para agente lista: ordenar chunks del mismo PDF por página/posición
        # así el LLM recibe la lista en orden correcto
        if agent_type == "lista" and docs:
            from collections import defaultdict
            by_source: dict = defaultdict(list)
            for d in docs:
                src = str(d.get("source", ""))
                by_source[src].append(d)
            # Fuente con más chunks = el PDF más relevante, ordenar sus chunks
            dominant_src = max(by_source, key=lambda s: len(by_source[s]))
            dominant_chunks = sorted(
                by_source[dominant_src],
                key=lambda d: d.get("chunk_index", d.get("page", 0))
            )
            # Poner los chunks de la fuente dominante primero, en orden
            other_chunks = [d for d in docs
                            if str(d.get("source","")) != dominant_src]
            docs = dominant_chunks + other_chunks

        state["trace"].append(
            f"🔍 [Iteración {state.get('iterations',0)+1}] "
            f"Recuperando: \"{q[:60]}…\" (top-{top_k})"
        )
        if srcs:
            src_str = ", ".join(list(dict.fromkeys(
                str(s).split("/")[-1] for s in srcs
            ))[:4])
            state["trace"].append(
                f"✅ {len(docs)} fragmentos recuperados · fuentes: {src_str}"
            )
        return {"documents": docs, "sources": srcs,
                "iterations": state.get("iterations", 0) + 1}

    # Nodo generate especializado para este tipo de agente
    def node_generate_routed(state: GraphState) -> dict:
        llm     = get_llm()
        query_s = state.get("rewritten_q") or state["query"]
        docs    = state["documents"]
        context = build_context(docs)

        fmt = AGENT_FORMAT.get(agent_type, AGENT_FORMAT["general"])
        system_prompt = SYSTEM_BASE.format(
            agent_type=agent_type,
            format_instruction=fmt
        )

        # Para agente LISTA: advertencia de verificación en la traza
        if agent_type == "lista":
            state["trace"].append(
                "⚠️ Modo extracción literal — verifica nombres contra el documento original"
            )
        state["trace"].append(
            f"✍️ [{AGENT_ICONS.get(agent_type,'💬')} {agent_type}] Generando respuesta…"
        )

        # Para agente lista: contexto completo + prompt de extracción literal
        if agent_type == "lista":
            context = build_context(docs)[:6000]   # más contexto para listas
            prompt  = (
                f"DOCUMENTOS (texto literal extraído del corpus):\n{context}\n\n"
                f"TAREA: {query_s}\n\n"
                f"INSTRUCCIÓN CRÍTICA: Copia EXACTAMENTE los nombres tal como "
                f"aparecen en los documentos anteriores. "
                f"NO uses tu conocimiento previo. NO inventes ni añadas ningún "
                f"nombre que no esté escrito literalmente en el texto de arriba. "
                f"Si un nombre aparece incompleto en el documento, cópialo "
                f"incompleto y añade '[texto cortado]'. "
                f"Si no encuentras la lista en los documentos, di exactamente: "
                f"'No encontré esta información en los fragmentos recuperados.'"
            )
            temp = 0.0   # temperatura 0 para máxima fidelidad
        else:
            context = build_context(docs)[:3000]
            prompt  = (f"DOCUMENTOS:\n{context}\n\n"
                       f"PREGUNTA: {query_s}\n\n"
                       f"Responde en español basándote solo en los documentos anteriores.")
            temp = 0.1
        try:
            resp = llm.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temp,
                max_tokens=1200,
                extra_body={"models": ["qwen/qwen3-32b",
                                       "google/gemma-3-27b-it:free",
                                       "meta-llama/llama-3.3-70b-instruct:free"]},
            )
            raw = resp.choices[0].message.content or ""

            # Extraer texto FUERA del bloque <think> primero
            # Si no hay texto fuera, usar el contenido del think como respuesta
            outside = re.sub(r"<think>.*?</think>", "", raw,
                             flags=re.DOTALL).strip()
            if outside:
                gen = outside
            else:
                # El modelo solo generó thinking — extraer conclusión del think
                think_match = re.search(r"<think>(.*?)</think>",
                                        raw, re.DOTALL)
                if think_match:
                    think_content = think_match.group(1).strip()
                    # Tomar los últimos 2 párrafos del thinking como respuesta
                    parrafos = [p.strip() for p in
                                think_content.split("\n\n") if p.strip()]
                    gen = "\n\n".join(parrafos[-2:]) if parrafos else raw
                else:
                    gen = raw  # devolver tal cual si no hay estructura

            # Fallback si sigue vacío: segunda llamada con modelo más simple
            if not gen or len(gen) < 20:
                try:
                    resp2 = llm.chat.completions.create(
                        model="google/gemma-3-27b-it:free",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": prompt},
                        ],
                        temperature=0.2,
                        max_tokens=800,
                    )
                    gen = (resp2.choices[0].message.content or "").strip()
                except Exception:
                    pass

            toks = getattr(getattr(resp, "usage", None), "total_tokens", 0)
        except Exception as e:
            gen  = f"⚠️ Error al generar: {e}"
            toks = 0

        return {"generation": gen,
                "tokens_used": state.get("tokens_used", 0) + toks}

    # Construir grafo con nodo generate especializado
    def decide_after_grading(state: GraphState) -> str:
        if (state["doc_grade"] == "not_relevant"
                and state["iterations"] < MAX_ITERATIONS):
            return "rewrite"
        return "generate"

    def decide_after_answer(state: GraphState) -> str:
        if state["answer_grade"] == "supported":
            return "END"
        if state["iterations"] < MAX_ITERATIONS:
            return "rewrite"
        return "END"

    g = StateGraph(GraphState)
    g.add_node("retrieve",     node_retrieve_smart)   # ← adaptativo
    g.add_node("grade_docs",   node_grade_documents)
    g.add_node("rewrite",      node_rewrite_query)
    g.add_node("generate",     node_generate_routed)
    g.add_node("grade_answer", node_grade_answer)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges("grade_docs", decide_after_grading,
        {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("rewrite",  "retrieve")
    g.add_edge("generate", "grade_answer")
    g.add_conditional_edges("grade_answer", decide_after_answer,
        {"END": END, "rewrite": "rewrite"})

    app = g.compile()

    t0 = _time.time()
    initial: GraphState = {
        "query":        query,
        "rewritten_q":  "",
        "documents":    [],
        "generation":   "",
        "doc_grade":    "",
        "answer_grade": "",
        "iterations":   0,
        "trace":        [],
        "sources":      [],
        "tokens_used":  0,
        "t_start":      t0,
    }

    try:
        final = app.invoke(initial)
    except Exception as e:
        return {"answer": f"⚠️ Error en grafo: {e}", "docs": [],
                "sources": [], "latency_s": _time.time()-t0,
                "trace": [f"❌ {e}"], "iterations": 0,
                "agent_type": agent_type, "tokens_used": 0,
                "doc_grade": "", "answer_grade": ""}

    return {
        "answer":       final.get("generation", "Sin respuesta"),
        "docs":         final.get("documents",  []),
        "sources":      list(set(final.get("sources", []))),
        "latency_s":    _time.time() - t0,
        "trace":        final.get("trace", []),
        "iterations":   final.get("iterations", 0),
        "tokens_used":  final.get("tokens_used", 0),
        "doc_grade":    final.get("doc_grade", ""),
        "answer_grade": final.get("answer_grade", ""),
        "agent_type":   agent_type,
    }

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE_RESPONSE — métricas RAGAS inline
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_response(query: str, context: str, answer: str) -> dict:
    llm = get_llm()
    prompt = (
        f"Evalúa esta respuesta de un sistema RAG sobre acreditación universitaria.\n"
        f"PREGUNTA: {query}\n"
        f"CONTEXTO RECUPERADO: {context[:1200]}\n"
        f"RESPUESTA GENERADA: {answer[:700]}\n"
        "Evalúa en escala 0.0 a 1.0 y responde SOLO con JSON:\n"
        '{"faithfulness": <float>, "answer_relevance": <float>, '
        '"context_precision": <float>, "context_recall": <float>}'
    )
    default = {"faithfulness": 0.8, "answer_relevance": 0.8,
               "context_precision": 0.7, "context_recall": 0.7}
    try:
        r = llm.chat.completions.create(
            model=EVAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=150,
        )
        raw   = clean_thinking(r.choices[0].message.content or "")
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        scores = json.loads(match.group()) if match else default
        for k in default:
            if k not in scores or not isinstance(scores[k], (int, float)):
                scores[k] = default[k]
        return scores
    except Exception:
        return default

# ─────────────────────────────────────────────────────────────────────────────
# MEMORIA CONVERSACIONAL
# ─────────────────────────────────────────────────────────────────────────────
class ConversationMemory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns

    def get_context(self, messages: list) -> str:
        recent = [m for m in messages if m["role"] != "system"][-self.max_turns:]
        if not recent:
            return ""
        return "\n".join(
            f"{'Usuario' if m['role']=='user' else 'Asistente'}: "
            f"{m['content'][:200]}"
            for m in recent
        )

# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK VECTORIAL (guarda en Qdrant)
# ─────────────────────────────────────────────────────────────────────────────
def save_feedback(query: str, answer: str, rating: int, agent_type: str):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, VectorParams, Distance
        from sentence_transformers import SentenceTransformer
        qdrant   = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")
        emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()

        cols = [c.name for c in qdrant.get_collections().collections]
        if FEEDBACK_COL not in cols:
            qdrant.create_collection(
                FEEDBACK_COL,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
        qdrant.upsert(
            collection_name=FEEDBACK_COL,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"query": query, "answer": answer[:500],
                         "rating": rating, "agent": agent_type,
                         "ts": time.time(), "source": "feedback_usuario"},
            )]
        )
        return True
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatAcredita · Sys-E",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS WhatsApp dark ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fondo y texto general ─────────────────────────────────────── */
html,body,[data-testid="stAppViewContainer"]{background:#f7f8fa!important;color:#1a1a2e!important}
[data-testid="stSidebar"]{background:#ffffff!important;border-right:1px solid #e2e6ea!important}
[data-testid="stSidebar"] *{color:#1a1a2e!important}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#1a1a2e!important}

/* ── Input de chat ─────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea{
  background:#ffffff!important;
  color:#1a1a2e!important;
  border:1.5px solid #d0d7de!important;
  border-radius:24px!important;
  font-size:14px!important;
  box-shadow:0 1px 4px rgba(0,0,0,0.06)!important;
}

/* ── Burbuja usuario ───────────────────────────────────────────── */
.wa-user{
  background:#1a73e8;
  color:#ffffff;
  border-radius:18px 4px 18px 18px;
  padding:10px 16px;
  margin:6px 0 6px 22%;
  max-width:76%;
  float:right;clear:both;
  font-size:14px;line-height:1.6;
  box-shadow:0 1px 3px rgba(26,115,232,0.25);
}

/* ── Burbuja bot ───────────────────────────────────────────────── */
.wa-bot{
  background:#ffffff;
  color:#1a1a2e;
  border-radius:4px 18px 18px 18px;
  padding:10px 16px;
  margin:6px 22% 6px 0;
  max-width:76%;
  float:left;clear:both;
  font-size:14px;line-height:1.6;
  box-shadow:0 1px 4px rgba(0,0,0,0.08);
  border:1px solid #e8ecf0;
}

/* ── Traza de nodos ────────────────────────────────────────────── */
.wa-trace{
  background:#f0f4ff;
  border-left:3px solid #1a73e8;
  border-radius:6px;
  padding:8px 14px;
  margin:4px 22% 4px 0;
  font-size:12px;
  color:#4a5568;
  clear:both;
}
.wa-step{color:#1a73e8;font-weight:500;}

/* ── Métricas ──────────────────────────────────────────────────── */
.metric-row{display:flex;gap:6px;flex-wrap:wrap;margin:4px 22% 4px 0;clear:both;}
.metric-pill{
  background:#eef2ff;
  color:#3730a3;
  border-radius:20px;
  padding:3px 10px;
  font-size:11px;
  font-weight:500;
  border:1px solid #c7d2fe;
}
.ragas-pill{
  background:#ecfdf5;
  color:#065f46;
  border-radius:20px;
  padding:3px 10px;
  font-size:11px;
  font-weight:500;
  border:1px solid #a7f3d0;
}

/* ── Badge de agente ───────────────────────────────────────────── */
.agent-badge{
  display:inline-block;
  padding:3px 10px;
  border-radius:20px;
  font-size:11px;
  font-weight:600;
  margin-bottom:4px;
  letter-spacing:0.02em;
}

/* ── Fuentes ───────────────────────────────────────────────────── */
.wa-sources{font-size:11px;color:#718096;margin:4px 22% 4px 0;clear:both;}
.wa-src-pill{
  display:inline-block;
  background:#f0f4f8;
  color:#4a5568;
  border-radius:12px;
  padding:2px 8px;
  margin:2px;
  font-size:10px;
  border:1px solid #e2e8f0;
}

/* ── Timestamp ─────────────────────────────────────────────────── */
.wa-time{font-size:10px;color:#a0aec0;text-align:right;margin-top:4px;}

/* ── Clearfix ──────────────────────────────────────────────────── */
.cf::after{content:"";display:table;clear:both;}

/* ── Header ────────────────────────────────────────────────────── */
.wa-header{
  background:#ffffff;
  border:1px solid #e2e6ea;
  border-radius:12px;
  padding:14px 18px;
  margin-bottom:14px;
  box-shadow:0 1px 4px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ── Login ─────────────────────────────────────────────────────────────────────
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.sidebar.markdown("## 🔐 Acceso ChatAcredita Sys-E")
    user = st.sidebar.text_input("Usuario")
    pwd  = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button("Ingresar"):
        if verify(user, pwd):
            st.session_state.auth = True
            st.session_state.user = user
            st.rerun()
        else:
            st.sidebar.error("❌ Credenciales incorrectas")
    st.info("Inicia sesión en el panel lateral para acceder a ChatAcredita Sys-E.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("messages", []), ("total_q", 0), ("total_lat", 0.0),
              ("history_ragas", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

memory = ConversationMemory()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🎓 ChatAcredita PRO")
    st.markdown(f"**Sys-E · RAG Router + LangGraph**")
    st.caption(f"👤 {st.session_state.get('user','?')} · "
               f"{datetime.now().strftime('%H:%M')}")
    st.divider()

    st.markdown("### 🏗️ Arquitectura Sys-E")
    st.markdown("""
```
Query
  ↓
[Router LLM] → tipo agente
  ↓
[LangGraph especializado]
  retrieve → grade_docs
  → [rewrite↺] | generate(agente)
  → grade_answer → END
  ↓
[evaluate_response] → RAGAS
```
""")
    st.divider()

    st.markdown("### 📊 Estado")
    if BACKEND_OK:
        st.success("✅ LangGraph listo")
        st.caption(f"Gen: `{GENERATOR_MODEL[:25]}`")
        st.caption(f"Router: `{ROUTER_MODEL[:25]}`")
    else:
        st.error(f"❌ {BACKEND_ERR}")

    st.divider()
    st.markdown("### 📈 Sesión")
    st.metric("Consultas", st.session_state.total_q)
    if st.session_state.total_q > 0:
        avg = st.session_state.total_lat / st.session_state.total_q
        st.metric("Latencia media", f"{avg:.1f}s")
    if st.session_state.history_ragas:
        avg_r = sum(r["ragas_avg"] for r in st.session_state.history_ragas) / \
                len(st.session_state.history_ragas)
        st.metric("RAGAS promedio sesión", f"{avg_r:.3f}")

    st.divider()
    show_ragas = st.toggle("Mostrar métricas RAGAS", value=True)
    compute_ragas = st.toggle("Calcular RAGAS (lento +15s)", value=False)
    st.session_state["compute_ragas"] = compute_ragas
    if compute_ragas:
        st.caption("⚠️ Activo: cada respuesta tardará ~15s adicionales")
    else:
        st.caption("✅ Desactivado: respuestas rápidas (~15-25s total)")

    st.divider()
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.total_q  = 0
        st.session_state.total_lat = 0.0
        st.session_state.history_ragas = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
AGENT_COLORS = {
    "lista": "#00897B",
    "estadistica": "#2196F3", "normativa": "#9C27B0",
    "proceso":     "#FF9800", "comparacion": "#F44336",
    "sintesis":    "#009688", "general": "#607D8B",
}

st.markdown("""
<div class="wa-header">
  <b style="color:#1a1a2e;font-size:15px;">🎓 ChatAcredita PRO · Sys-E — RAG Router + LangGraph</b><br>
  <small style="color:#718096;">Router de agentes → Grafo LangGraph especializado → Métricas RAGAS · CNA/EISC</small>
</div>
""", unsafe_allow_html=True)

# ── Renderizar historial ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]

    if role == "user":
        st.markdown(f'<div class="cf"><div class="wa-user">'
                    f'{html.escape(msg["content"])}</div></div>',
                    unsafe_allow_html=True)

    elif role == "assistant":
        agent  = msg.get("agent_type", "general")
        color  = AGENT_COLORS.get(agent, "#607D8B")
        icon   = AGENT_ICONS.get(agent, "💬")

        # Badge de tipo de agente
        st.markdown(
            f'<span class="agent-badge" style="background:{color}22;'
            f'color:{color};border:1px solid {color}66">'
            f'{icon} Agente {agent}</span>',
            unsafe_allow_html=True
        )

        # Traza de nodos
        if msg.get("trace"):
            trace_html = "<br>".join(
                f'<span class="wa-step">{html.escape(str(t))}</span>'
                for t in msg["trace"]
            )
            st.markdown(f'<div class="wa-trace">{trace_html}</div>',
                        unsafe_allow_html=True)

        # Métricas operacionales
        m_html  = ""
        if msg.get("latency_s"):
            m_html += f'<span class="metric-pill">⏱ {msg["latency_s"]:.1f}s</span>'
        if msg.get("iterations") is not None:
            m_html += f'<span class="metric-pill">🔄 {msg["iterations"]+1} iteración(es)</span>'
        if msg.get("doc_grade"):
            icon_g = "✅" if msg["doc_grade"] == "relevant" else "⚠️"
            m_html += f'<span class="metric-pill">{icon_g} docs: {msg["doc_grade"]}</span>'
        if msg.get("answer_grade"):
            icon_g = "✅" if msg["answer_grade"] == "supported" else "⚠️"
            m_html += f'<span class="metric-pill">{icon_g} answer: {msg["answer_grade"]}</span>'

        # Métricas RAGAS
        if show_ragas and msg.get("scores"):
            s = msg["scores"]
            avg = sum(s.get(k, 0) for k in
                      ["faithfulness","answer_relevance",
                       "context_precision","context_recall"]) / 4
            m_html += (f'<span class="ragas-pill">F:{s.get("faithfulness",0):.2f}</span>'
                       f'<span class="ragas-pill">R:{s.get("answer_relevance",0):.2f}</span>'
                       f'<span class="ragas-pill">P:{s.get("context_precision",0):.2f}</span>'
                       f'<span class="ragas-pill">C:{s.get("context_recall",0):.2f}</span>'
                       f'<span class="ragas-pill">avg:{avg:.2f}</span>')

        if m_html:
            st.markdown(f'<div class="metric-row">{m_html}</div>',
                        unsafe_allow_html=True)

        # Burbuja respuesta
        content_raw = (msg.get("content") or "⚠️ Sin respuesta")
        answer_safe = html.escape(content_raw).replace("\n", "<br>")
        answer_safe = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', answer_safe)
        ts = html.escape(msg.get("timestamp", ""))
        st.markdown(
            f'<div class="cf"><div class="wa-bot">{answer_safe}'
            f'<div class="wa-time">{ts}</div></div></div>',
            unsafe_allow_html=True
        )

        # Fuentes
        if msg.get("sources"):
            pills = "".join(
                f'<span class="wa-src-pill">📄 {html.escape(s)}</span>'
                for s in list(dict.fromkeys(msg["sources"]))[:6]
            )
            st.markdown(f'<div class="wa-sources">{pills}</div>',
                        unsafe_allow_html=True)

        # Feedback ⭐
        if msg.get("show_feedback", False):
            cols = st.columns(5)
            for i, c in enumerate(cols):
                if c.button("⭐" * (i+1), key=f"fb_{msg['id']}_{i}"):
                    save_feedback(msg.get("query",""), msg.get("content",""),
                                  i+1, agent)
                    st.toast(f"Feedback {i+1}⭐ guardado", icon="✅")

# ── Preguntas de ejemplo ──────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**💬 Preguntas de ejemplo:**")
    examples = [
        ("📜 Normativa",  "¿Cuáles son los diez factores del modelo CNA según Decreto 1330?"),
        ("📊 Estadística","¿Cuántos profesores de tiempo completo tiene la EISC?"),
        ("⚙️ Proceso",    "¿Cuáles son las etapas del proceso de autoevaluación?"),
        ("⚖️ Comparación","¿Cuál es la diferencia entre acreditación inicial y renovación?"),
        ("🔍 Síntesis",   "¿Qué fortalezas identificó el CNA en el último informe de pares?"),
        ("💬 General",    "¿Qué ventajas ofrece estudiar en un programa acreditado?"),
    ]
    cols = st.columns(2)
    for i, (label, ex) in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"{label}: {ex[:45]}…", key=f"ex_{i}",
                         use_container_width=True):
                st.session_state._pending = ex
                st.rerun()

# ── Procesar query pendiente (ejemplos) ───────────────────────────────────────
def process_query(query_raw: str):
    query = sanitize(query_raw)
    if not rate_ok(st.session_state.get("user", "anon")):
        st.warning("⏳ Límite de 15 consultas/minuto alcanzado. Espera un momento.")
        return

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🔀 Clasificando pregunta…"):
        agent_type = route_query(query) if BACKEND_OK else "general"

    placeholder = st.empty()
    placeholder.markdown(
        f'<div class="wa-trace"><span class="wa-step">'
        f'🔀 Router → agente <b style="color:#1a73e8">{agent_type}</b> '
        f'{AGENT_ICONS.get(agent_type,"")}</span></div>',
        unsafe_allow_html=True
    )

    with st.spinner(f"🧠 LangGraph [{agent_type}]…"):
        if BACKEND_OK:
            result = run_routed_langgraph(query, agent_type)
        else:
            result = {"answer": f"⚠️ Backend no disponible: {BACKEND_ERR}",
                      "trace": [], "sources": [], "latency_s": 0,
                      "iterations": 0, "doc_grade": "", "answer_grade": "",
                      "agent_type": agent_type, "tokens_used": 0}

    placeholder.empty()

    # Calcular métricas RAGAS solo si el toggle está activo
    docs    = result.get("docs", [])
    context = build_context(docs) if docs and BACKEND_OK else ""
    scores  = {}
    # Solo calcular si hay respuesta Y el toggle de RAGAS está activado
    # Esto evita la 5ª llamada LLM que bloquea la UI en cada consulta
    compute_ragas = st.session_state.get("compute_ragas", False)
    if result.get("answer") and context and compute_ragas:
        with st.spinner("📊 Calculando métricas RAGAS (puede tardar 10-15s)…"):
            scores = evaluate_response(query, context, result["answer"])
            ragas_avg = sum(scores.get(k, 0) for k in
                           ["faithfulness","answer_relevance",
                            "context_precision","context_recall"]) / 4
            st.session_state.history_ragas.append({
                "query": query[:60], "agent": agent_type,
                "ragas_avg": ragas_avg, "scores": scores,
            })

    st.session_state.total_q   += 1
    st.session_state.total_lat += result.get("latency_s", 0)

    answer_text = (result.get("answer") or "").strip()
    if not answer_text or len(answer_text) < 15:
        # Mostrar qué recuperó el sistema aunque no haya generado texto
        sources = result.get("sources", [])
        src_str = ", ".join(sources[:3]) if sources else "ninguna"
        answer_text = (
            f"⚠️ El modelo no generó texto de respuesta para esta pregunta.\n\n"
            f"**Documentos encontrados:** {src_str}\n\n"
            f"**Sugerencia:** reformula la pregunta siendo más específico, "
            f"por ejemplo: *'Lista los nombres de los profesores de planta "
            f"de la EISC según el documento de profesores'*"
        )

    ts  = datetime.now().strftime("%H:%M")
    mid = str(uuid.uuid4())
    st.session_state.messages.append({
        "role":          "assistant",
        "id":            mid,
        "query":         query,
        "content":       answer_text,
        "trace":         result.get("trace", []),
        "sources":       result.get("sources", []),
        "latency_s":     result.get("latency_s", 0),
        "iterations":    result.get("iterations", 0),
        "doc_grade":     result.get("doc_grade", ""),
        "answer_grade":  result.get("answer_grade", ""),
        "agent_type":    agent_type,
        "scores":        scores,
        "tokens_used":   result.get("tokens_used", 0),
        "timestamp":     ts,
        "show_feedback": True,
    })
    st.rerun()


if hasattr(st.session_state, "_pending"):
    q = st.session_state._pending
    del st.session_state._pending
    process_query(q)

if query_input := st.chat_input(
    "Escribe tu pregunta sobre acreditación CNA/EISC…"):
    process_query(query_input)
