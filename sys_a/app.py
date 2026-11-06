# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RAG Acreditacion — App Streamlit                                           ║
# ║  BGE-M3 + Qdrant + BGE-Reranker-v2-m3 + Gemma 4-26B (OpenRouter)          ║
# ║  Compara RAG (top-5 chunks) vs Long-Context (corpus completo)               ║
# ║  EISC — Universidad del Valle, Cali, Colombia                               ║
# ║  Uso: streamlit run app_rag_acreditacion.py                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import time
import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Acreditacion · EISC Univalle",
    page_icon="🎓",
    layout="wide",
)

QDRANT_URL      = os.getenv("QDRANT_URL", "os.getenv("QDRANT_URL", "")")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "os.getenv("QDRANT_API_KEY", "")")
OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY", "os.getenv("OPENROUTER_API_KEY", "")")

EMBED_MODEL    = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
#GEN_MODEL      = "google/gemma-4-31b-it"
GEN_MODEL      = "deepseek/deepseek-r1-0528"
#GEN_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
#GEN_MODEL ="meta-llama/llama-3.3-70b-instruct:free"
COLLECTION     = "acreditacion"
PDF_DIR        = "pdfs"

TOP_K_CANDIDATES = 20
TOP_K_FINAL      = 8
LC_MAX_CHARS     = 80_000

OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://eisc.univalle.edu.co",
    "X-Title":      "RAG Acreditacion EISC Univalle",
}

RESULTS_CACHE  = "results_acredita.pkl"
LC_CORPUS_CACHE = "corpus_lc.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTAS DE EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────
QUESTIONS = [
    {
        "question": "Cual es el proceso para renovar la acreditacion de alta calidad de un programa ante el CNA?",
        "reference_answer": "El programa realiza autoevaluacion, elabora un informe, lo presenta al CNA, "
            "recibe visita de pares academicos y obtiene el concepto del Consejo Nacional de Acreditacion.",
    },
    {
        "question": "Cuales son los grupos de investigacion de la EISC y quienes los dirigen?",
        "reference_answer": "GUIA dirigido por Oscar Hernan Florez, y otros grupos de la EISC en sistemas y computacion.",
    },
    {
        "question": "Que mecanismos usa el programa para el seguimiento de estudiantes en riesgo de desercion?",
        "reference_answer": "El Semaforo de Alertas y el sistema ASES permiten identificar y hacer seguimiento "
            "a estudiantes en riesgo de desercion.",
    },
    {
        "question": "Que porcentaje de egresados del programa se vincula al sector de informacion y comunicaciones?",
        "reference_answer": "Segun el Observatorio Laboral, un porcentaje significativo de egresados se vincula "
            "al sector de tecnologias de la informacion y comunicaciones.",
    },
    {
        "question": "Cuantos creditos academicos tiene el plan de estudios del programa de Ingenieria de Sistemas?",
        "reference_answer": "El plan de estudios de Ingenieria de Sistemas de la Universidad del Valle "
            "tiene un total de creditos academicos definidos en el pensum vigente.",
    },
    {
        "question": "Cual es la mision de la EISC Universidad del Valle?",
        "reference_answer": "La EISC tiene como mision formar ingenieros de sistemas con alta calidad academica, "
            "capaces de contribuir al desarrollo tecnologico y cientifico del pais.",
    },
    {
        "question": "Cuales son los factores de acreditacion del CNA para programas de pregrado?",
        "reference_answer": "Los factores del CNA incluyen: Mision, Estudiantes, Profesores, Procesos Academicos, "
            "Visibilidad, Investigacion, Pertinencia, Autoevaluacion, Bienestar, Organizacion y Recursos.",
    },
    {
        "question": "Que convenios internacionales tiene la EISC para movilidad estudiantil?",
        "reference_answer": "La EISC cuenta con convenios de cooperacion internacional que permiten "
            "intercambios estudiantiles con universidades extranjeras.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE RECURSOS (cacheados por Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📥 Cargando BGE-M3...")
def load_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device), device


@st.cache_resource(show_spinner="📥 Cargando BGE-Reranker-v2-m3...")
def load_reranker():
    try:
        from FlagEmbedding import FlagReranker
        devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
        return FlagReranker(RERANKER_MODEL, use_fp16=True, devices=devices)
    except Exception as e:
        st.sidebar.warning(f"Reranker no disponible: {e}")
        return None


@st.cache_resource(show_spinner="🔌 Conectando Qdrant...")
def load_qdrant():
    if not QDRANT_URL:
        st.error("❌ Configura QDRANT_URL en .env")
        st.stop()
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    info = client.get_collection(COLLECTION)
    return client, info.points_count


@st.cache_data(show_spinner="📚 Cargando corpus Long-Context...")
def load_lc_corpus():
    """Carga el corpus de PDFs para el modo Long-Context."""
    import fitz
    import pymupdf4llm

    if os.path.exists(LC_CORPUS_CACHE):
        with open(LC_CORPUS_CACHE, "rb") as f:
            corpus = pickle.load(f)
        return corpus

    pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))
    if not pdf_files:
        return []

    corpus = []
    for p in pdf_files:
        doc = fitz.open(p)
        text = ""
        for page_num in range(len(doc)):
            try:
                page_md = pymupdf4llm.to_markdown(
                    doc, pages=[page_num], show_progress=False, page_chunks=False
                )
            except Exception:
                page_md = doc[page_num].get_text()
            text += f"\n--- Pag {page_num+1} ---\n{page_md}"
        doc.close()
        if text.strip():
            corpus.append({"title": p.name, "text": text})

    with open(LC_CORPUS_CACHE, "wb") as f:
        pickle.dump(corpus, f)
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES CORE
# ─────────────────────────────────────────────────────────────────────────────
def to_ascii(text: str) -> str:
    return text.encode("ascii", errors="ignore").decode("ascii")


def retrieve(query: str, embedder, reranker, qdrant_client) -> list[dict]:
    """BGE-M3 → Qdrant top-20 → BGE-Reranker → top-5."""
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()

    hits = qdrant_client.query_points(
        collection_name=COLLECTION,
        query=q_emb,
        limit=TOP_K_CANDIDATES,
        with_payload=True,
    ).points

    candidates = [
        {
            "text":   (h.payload or {}).get("text", ""),
            "source": (h.payload or {}).get("source", ""),
            "score":  h.score,
        }
        for h in hits
    ]

    if reranker and candidates:
        pairs  = [[query, c["text"][:1024]] for c in candidates]
        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    else:
        for c in candidates:
            c["rerank_score"] = c["score"]

    return candidates[:TOP_K_FINAL]


SYSTEM_ESPANOL = (
    "Eres un asistente experto en acreditacion universitaria colombiana (CNA/EISC). "
    "REGLA ABSOLUTA: Responde SIEMPRE en espanol. "
    "Nunca respondas en ingles ni en otro idioma, sin importar el idioma del contexto. "
    "Si el contexto esta en otro idioma, traduce la respuesta al espanol."
)


def call_llm(prompt: str, max_tokens: int = 600) -> tuple[str, float]:
    """
    Llama al modelo via OpenRouter.
    Usa messages=[system, user] para forzar respuesta en español.
    El system message tiene prioridad sobre el idioma del contexto.
    """
    if not OPENROUTER_KEY:
        return "Error: configura OPENROUTER_API_KEY en .env", 0.0
    client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_ESPANOL},
                {"role": "user",   "content": to_ascii(prompt)},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            timeout=60,
            extra_headers=OPENROUTER_HEADERS,
            extra_body={
                # Fallback automático si GEN_MODEL falla o devuelve vacío
                "models": [
                    "google/gemma-3-27b-it:free",          # 2do: estable, sin thinking
                    "meta-llama/llama-3.3-70b-instruct:free", # 3ro: potente y gratis
                    "deepseek/deepseek-r1-0528:free",      # 4to: con thinking
                ]
            },
        )
        import re as _re
        raw = resp.choices[0].message.content or ""
        # Si content viene vacío, intentar reasoning_content (DeepSeek R1/V4, Qwen3)
        if not raw.strip():
            raw = getattr(resp.choices[0].message, "reasoning_content", "") or ""
        # Eliminar bloque <think>...</think> — queda solo la respuesta real
        clean = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        if not clean:
            clean = _re.sub(r"</?think>", "", raw).strip()
        if not clean:
            # Debug: mostrar qué devolvió realmente el modelo
            finish = resp.choices[0].finish_reason or "?"
            model_used = getattr(resp, "model", "?")
            clean = (
                f"⚠️ El modelo ({model_used}) respondió vacío "
                f"[finish_reason={finish}]. "
                f"Intenta cambiar el modelo en el sidebar o reformula la pregunta."
            )
        return clean, time.time() - t0
    except Exception as e:
        return f"ERROR: {str(e)[:150]}", time.time() - t0


def answer_with_rag(question: str, embedder, reranker, qdrant_client) -> dict:
    docs = retrieve(question, embedder, reranker, qdrant_client)
    context = "\n\n---\n\n".join(
        f"[Fuente: {d['source']}]\n{d['text']}" for d in docs
    )
    prompt = (
        "Responde la siguiente pregunta usando UNICAMENTE el contexto proporcionado.\n"
        "Si el contexto no contiene la respuesta, di que no encontraste la informacion.\n"
        "Cita la fuente al final: [Fuente: nombre_documento.pdf]\n\n"
        f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}\nRESPUESTA EN ESPANOL:"
    )
    answer, latency = call_llm(prompt)
    return {"answer": answer, "latency_s": latency, "docs": docs}


def answer_with_long_context(question: str, full_context_trimmed: str) -> dict:
    prompt = (
        "Responde la siguiente pregunta usando UNICAMENTE el contexto proporcionado.\n"
        "Si el contexto no contiene la respuesta, di que no encontraste la informacion.\n\n"
        f"CONTEXTO:\n{full_context_trimmed}\n\nPREGUNTA: {question}\nRESPUESTA EN ESPANOL:"
    )
    answer, latency = call_llm(prompt)
    return {"answer": answer, "latency_s": latency}


def judge_correct(question: str, reference: str, model_answer: str) -> bool:
    prompt = (
        f"Pregunta: {question}\n"
        f"Respuesta de referencia: {reference}\n"
        f"Respuesta del modelo: {model_answer}\n\n"
        "La respuesta del modelo transmite correctamente los hechos clave? "
        "Diferencias menores de redaccion estan bien. "
        "Responde con exactamente una palabra: SI o NO."
    )
    answer, _ = call_llm(prompt, max_tokens=5)
    return answer.strip().upper().startswith("S")


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🎓 RAG Acreditacion CNA — EISC Universidad del Valle")
    st.caption(
        f"BGE-M3 + Qdrant `{COLLECTION}` + BGE-Reranker-v2-m3 · "
        f"Generacion: `{GEN_MODEL}` via OpenRouter"
    )

    # ── Cargar recursos ──────────────────────────────────────────────────────
    embedder, device = load_embedder()
    reranker         = load_reranker()
    qdrant_client, n_chunks = load_qdrant()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Configuracion")
    st.sidebar.success(f"✅ BGE-M3 ({device})")
    st.sidebar.success(f"✅ Reranker: {'OK' if reranker else 'no disponible'}")
    st.sidebar.success(f"✅ Qdrant: {n_chunks:,} chunks")

    top_k = st.sidebar.slider("Top-K chunks al prompt", 1, 10, TOP_K_FINAL)
    top_k_cand = st.sidebar.slider("Candidatos antes del reranker", 5, 40, TOP_K_CANDIDATES)

    st.sidebar.markdown("---")
    modo = st.sidebar.radio(
        "Modo",
        ["💬 Chat libre", "📊 Evaluacion batch RAG vs LC"],
        index=0,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # MODO 1: CHAT LIBRE — Estilo WhatsApp
    # ══════════════════════════════════════════════════════════════════════════
    if modo == "💬 Chat libre":

        # ── CSS WhatsApp ─────────────────────────────────────────────────────
        st.markdown("""
        <style>
        /* Contenedor del chat */
        .wa-chat-wrapper {
            background: #0b141a;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            max-height: 560px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        /* Burbuja usuario (derecha, verde WhatsApp) */
        .wa-msg-user {
            align-self: flex-end;
            background: #005c4b;
            color: #e9edef;
            border-radius: 12px 2px 12px 12px;
            padding: 8px 12px;
            max-width: 75%;
            font-size: 0.92rem;
            line-height: 1.45;
            word-break: break-word;
        }
        /* Burbuja asistente (izquierda, gris oscuro) */
        .wa-msg-bot {
            align-self: flex-start;
            background: #202c33;
            color: #e9edef;
            border-radius: 2px 12px 12px 12px;
            padding: 8px 12px;
            max-width: 75%;
            font-size: 0.92rem;
            line-height: 1.45;
            word-break: break-word;
        }
        /* Nombre del bot en burbuja */
        .wa-sender {
            font-size: 0.74rem;
            color: #53bdeb;
            font-weight: 600;
            margin-bottom: 3px;
        }
        /* Timestamp */
        .wa-time {
            font-size: 0.68rem;
            color: #8696a0;
            text-align: right;
            margin-top: 4px;
        }
        /* Fuentes debajo de burbuja bot */
        .wa-sources {
            font-size: 0.72rem;
            color: #8696a0;
            margin-top: 4px;
            padding-left: 4px;
            border-left: 2px solid #53bdeb;
        }
        /* Latencia */
        .wa-latency {
            font-size: 0.68rem;
            color: #8696a0;
        }
        </style>
        """, unsafe_allow_html=True)

        # ── Estado del historial ─────────────────────────────────────────────
        if "wa_history" not in st.session_state:
            st.session_state.wa_history = []   # lista de dicts {role, text, sources, latency_s, ts}

        # ── Selector de modo RAG / LC (sidebar ya activo) ────────────────────
        modo_respuesta = st.sidebar.radio(
            "Modo de recuperacion",
            ["RAG (BGE-M3 + Qdrant)", "Long-Context (corpus completo)"],
            key="chat_modo_resp",
        )

        # ── Render del historial ─────────────────────────────────────────────
        import datetime

        def _ts():
            return datetime.datetime.now().strftime("%H:%M")

        chat_html_parts = []
        for msg in st.session_state.wa_history:
            ts = msg.get("ts", "")
            if msg["role"] == "user":
                chat_html_parts.append(
                    f'<div class="wa-msg-user">{msg["text"]}'
                    f'<div class="wa-time">{ts}</div></div>'
                )
            else:
                sources_html = ""
                if msg.get("sources"):
                    src_list = " · ".join(
                        f"📄 {os.path.basename(str(s))}" for s in msg["sources"]
                    )
                    sources_html = f'<div class="wa-sources">{src_list}</div>'
                lat_html = ""
                if msg.get("latency_s"):
                    lat_html = f'<div class="wa-latency">⏱ {msg["latency_s"]:.1f}s</div>'
                chat_html_parts.append(
                    f'<div class="wa-msg-bot">'
                    f'<div class="wa-sender">🎓 AcreditaBot</div>'
                    f'{msg["text"]}'
                    f'{sources_html}{lat_html}'
                    f'<div class="wa-time">{ts}</div>'
                    f'</div>'
                )

        if chat_html_parts:
            st.markdown(
                f'<div class="wa-chat-wrapper">{"".join(chat_html_parts)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="wa-chat-wrapper">'
                '<div class="wa-msg-bot">'
                '<div class="wa-sender">🎓 AcreditaBot</div>'
                'Hola 👋 Soy tu asistente de acreditación CNA/EISC. '
                '¿En qué puedo ayudarte?'
                '<div class="wa-time">ahora</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

        # ── Input + botón al estilo barra WhatsApp ───────────────────────────
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            query = st.text_input(
                "Mensaje",
                placeholder="Escribe tu pregunta sobre acreditación…",
                label_visibility="collapsed",
                key="wa_input",
            )
        with col_btn:
            send = st.button("➤", type="primary", use_container_width=True)

        col_clear, _ = st.columns([1, 5])
        with col_clear:
            if st.button("🗑 Limpiar chat"):
                st.session_state.wa_history = []
                st.rerun()

        # ── Procesamiento al enviar ──────────────────────────────────────────
        if send and query.strip():
            # Añadir mensaje usuario
            st.session_state.wa_history.append(
                {"role": "user", "text": query, "ts": _ts()}
            )

            if "RAG" in modo_respuesta:
                with st.spinner("🔍 Buscando en Qdrant + reranking…"):
                    result = answer_with_rag(query, embedder, reranker, qdrant_client)
                sources = [d["source"] for d in result["docs"]]

                # Expander de chunks debajo del chat (fuera de la burbuja)
                with st.expander(f"📚 Chunks recuperados (top-{top_k})", expanded=False):
                    for i, d in enumerate(result["docs"], 1):
                        st.markdown(
                            f"**[{i}]** `score={d['rerank_score']:.3f}` · "
                            f"📄 `{os.path.basename(str(d['source']))}`"
                        )
                        st.caption(d["text"][:300] + "…")
                        st.divider()

                st.session_state.wa_history.append({
                    "role": "bot",
                    "text": result["answer"],
                    "sources": sources,
                    "latency_s": result["latency_s"],
                    "ts": _ts(),
                })

            else:  # Long-Context
                with st.spinner("📚 Cargando corpus y generando respuesta…"):
                    lc_corpus = load_lc_corpus()
                    if not lc_corpus:
                        st.warning(f"No se encontraron PDFs en la carpeta `{PDF_DIR}/`")
                    else:
                        full_ctx = "\n\n===\n\n".join(
                            f"[Fuente: {d['title']}]\n{d['text']}" for d in lc_corpus
                        )[:LC_MAX_CHARS]
                        result = answer_with_long_context(query, full_ctx)
                        st.session_state.wa_history.append({
                            "role": "bot",
                            "text": result["answer"],
                            "sources": [],
                            "latency_s": result["latency_s"],
                            "ts": _ts(),
                        })

            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # MODO 2: EVALUACIÓN BATCH
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.subheader("Evaluacion comparativa: RAG vs Long-Context")
        st.info(
            f"Se evaluan **{len(QUESTIONS)} preguntas** predefinidas sobre acreditacion CNA/EISC. "
            "El juez (Gemma 4-26B) evalua si la respuesta del modelo es correcta. "
            "Los resultados se cachean en disco."
        )

        col_run, col_clear = st.columns([1, 4])
        run_eval   = col_run.button("▶️ Ejecutar evaluacion", type="primary")
        clear_cache = col_clear.button("🗑 Borrar cache y re-evaluar")

        if clear_cache and os.path.exists(RESULTS_CACHE):
            os.remove(RESULTS_CACHE)
            st.success("Cache borrado. Ejecuta la evaluacion de nuevo.")

        # Cargar resultados existentes
        results = []
        if os.path.exists(RESULTS_CACHE):
            with open(RESULTS_CACHE, "rb") as f:
                results = pickle.load(f)

        # Ejecutar evaluacion
        if run_eval:
            lc_corpus = load_lc_corpus()
            full_ctx  = ""
            if lc_corpus:
                full_ctx = "\n\n===\n\n".join(
                    f"[Fuente: {d['title']}]\n{d['text']}" for d in lc_corpus
                )[:LC_MAX_CHARS]
            else:
                st.warning(f"No se encontraron PDFs en `{PDF_DIR}/` — Long-Context no disponible")

            results = []
            progress = st.progress(0, text="Iniciando evaluacion...")
            status   = st.empty()

            for qi, q in enumerate(QUESTIONS):
                progress.progress((qi) / len(QUESTIONS),
                                  text=f"Pregunta {qi+1}/{len(QUESTIONS)}")
                status.markdown(f"**Evaluando:** {q['question'][:80]}...")

                # RAG
                rag = answer_with_rag(q["question"], embedder, reranker, qdrant_client)
                rag_correct = judge_correct(q["question"], q["reference_answer"], rag["answer"])

                # Long-Context
                if full_ctx:
                    lc = answer_with_long_context(q["question"], full_ctx)
                    lc_correct = judge_correct(q["question"], q["reference_answer"], lc["answer"])
                else:
                    lc = {"answer": "N/A — sin PDFs", "latency_s": 0.0}
                    lc_correct = False

                results.append({
                    "question":         q["question"],
                    "reference_answer": q["reference_answer"],
                    "rag_answer":       rag["answer"],
                    "rag_correct":      rag_correct,
                    "rag_latency_s":    rag["latency_s"],
                    "rag_sources":      [d["source"] for d in rag["docs"]],
                    "lc_answer":        lc["answer"],
                    "lc_correct":       lc_correct,
                    "lc_latency_s":     lc["latency_s"],
                })
                time.sleep(2)  # rate limit OpenRouter free tier

            with open(RESULTS_CACHE, "wb") as f:
                pickle.dump(results, f)

            progress.progress(1.0, text="✅ Evaluacion completada")
            status.empty()

        # Mostrar resultados
        if results:
            df = pd.DataFrame(results)

            # Métricas resumen
            st.markdown("### 📊 Resultados")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RAG — Exactitud",
                        f"{df['rag_correct'].mean()*100:.0f}%",
                        f"{df['rag_correct'].sum()}/{len(df)} correctas")
            col2.metric("LC — Exactitud",
                        f"{df['lc_correct'].mean()*100:.0f}%",
                        f"{df['lc_correct'].sum()}/{len(df)} correctas")
            col3.metric("RAG — Latencia media",   f"{df['rag_latency_s'].mean():.1f}s")
            col4.metric("LC — Latencia media",    f"{df['lc_latency_s'].mean():.1f}s")

            # Gráficas
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            labels = ["RAG\n(BGE-M3+Qdrant)", "Long-Context"]
            colors = ["#2563eb", "#e74c3c"]

            axes[0].bar(labels, [df["rag_correct"].mean(), df["lc_correct"].mean()], color=colors)
            axes[0].set_title(f"Exactitud (n={len(df)})")
            axes[0].set_ylim(0, 1)
            axes[0].set_ylabel("Fraccion correcta")

            axes[1].bar(labels, [df["rag_latency_s"].mean(), df["lc_latency_s"].mean()], color=colors)
            axes[1].set_title("Latencia promedio (s)")
            axes[1].set_ylabel("segundos")

            plt.suptitle(f"RAG vs Long-Context — {GEN_MODEL}", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Tabla detallada
            st.markdown("### 📋 Resultados por pregunta")
            for i, row in df.iterrows():
                rag_ok = "✅" if row["rag_correct"] else "❌"
                lc_ok  = "✅" if row["lc_correct"]  else "❌"
                with st.expander(f"{rag_ok} RAG | {lc_ok} LC — {row['question'][:80]}"):
                    col_rag, col_lc = st.columns(2)
                    with col_rag:
                        st.markdown(f"**RAG** ({row['rag_latency_s']:.1f}s) {rag_ok}")
                        st.markdown(row["rag_answer"])
                        st.caption("Fuentes: " + " · ".join(
                            os.path.basename(str(s)) for s in (row["rag_sources"] or [])
                        ))
                    with col_lc:
                        st.markdown(f"**Long-Context** ({row['lc_latency_s']:.1f}s) {lc_ok}")
                        st.markdown(row["lc_answer"])
                    st.divider()
                    st.caption(f"Referencia: {row['reference_answer']}")

            # Exportar
            csv = df.drop(columns=["rag_sources"]).to_csv(index=False)
            st.download_button(
                "⬇️ Descargar resultados CSV",
                data=csv,
                file_name="resultados_rag_acredita.csv",
                mime="text/csv",
            )
        else:
            st.info("Presiona **Ejecutar evaluacion** para comenzar.")


if __name__ == "__main__":
    main()
