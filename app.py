# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RAG Acreditacion — App Streamlit  v2                                       ║
# ║  BGE-M3 + Qdrant + BGE-Reranker-v2-m3                                      ║
# ║  Generadores: Qwen2.5-7B-Acredita | DeepSeek-14B-Acredita | ChatAcredita PRO║
# ║  Backend: HuggingFace local (GPU) | OpenRouter (nube) | Ollama (local GGUF) ║
# ║  EISC — Universidad del Valle, Cali, Colombia                               ║
# ║  Uso: streamlit run rag_acreditacion_v2.py                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Modelos fine-tuneados disponibles:
#   raulgdp/qwen2.5-7b-acredita-cna-col   → 7B  Qwen2.5  (SFT+LoRA sobre CNA)
#   raulgdp/deepseek14b-acredita           → 14B DeepSeek-R1-Distill-Qwen (FT sobre CNA)
#
# Backend por defecto: HuggingFace local (NF4, descarga automática desde HF Hub)
#   - Requiere VRAM ≥ 8 GB para el modelo 7B, ≥ 10 GB para el 14B
# Fallback opcional: OpenRouter (requiere OPENROUTER_API_KEY en .env)
#
# Instalación mínima:
#   pip install streamlit qdrant-client FlagEmbedding sentence-transformers
#   pip install openai python-dotenv
#   pip install transformers accelerate bitsandbytes  # backend HF
#   pip install pymupdf pymupdf4llm                   # modo Long-Context

import os
import re
import sys
import io
import time
import pickle
import builtins
import datetime
from pathlib import Path

# ── Forzar UTF-8 globalmente en Windows ──────────────────────────────────────
# Windows usa charmap/cp1252 por defecto; los tokenizers de HF tienen UTF-8.
os.environ["PYTHONIOENCODING"]  = "utf-8"
os.environ["PYTHONUTF8"]        = "1"
os.environ.setdefault("HF_HUB_VERBOSITY", "warning")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Parchear builtins.open para que siempre use UTF-8 al leer texto
_original_open = builtins.open
def _utf8_open(file, mode="r", buffering=-1, encoding=None, errors=None, **kw):
    if "b" not in mode and encoding is None:
        encoding = "utf-8"
        errors   = errors or "replace"
    return _original_open(file, mode, buffering, encoding=encoding, errors=errors, **kw)
builtins.open = _utf8_open
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import torch
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Acreditacion · EISC Univalle",
    page_icon="🎓",
    layout="wide",
)

QDRANT_URL     = os.getenv("QDRANT_URL",        "https://ac76af0c-59ba-4a65-87e4-ff360d8317ef.us-east-1-1.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY",    "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY","")

EMBED_MODEL    = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
COLLECTION     = "acreditacion"
PDF_DIR        = "pdfs"

TOP_K_CANDIDATES = 20
TOP_K_FINAL      = 8
LC_MAX_CHARS     = 80_000
RESULTS_CACHE    = "results_acredita_v2.pkl"
LC_CORPUS_CACHE  = "corpus_lc.pkl"

OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://eisc.univalle.edu.co",
    "X-Title":      "RAG Acreditacion EISC Univalle",
}

# Modelos fine-tuneados (tus dos modelos)
ACREDITA_MODELS = {
    "🧠 Qwen2.5-7B-Acredita (CNA fine-tuned)": {
        "hf_id":      "raulgdp/qwen2.5-7b-acredita-cna-col",
        "openrouter": "qwen/qwen2.5-7b-instruct",   # equivalente cloud
        "params":     "7B",
        "base":       "Qwen2.5-7B-Instruct",
        "desc":       "SFT + LoRA · 93 829 Q&A CNA",
    },
    "🔬 DeepSeek-14B-Acredita (CNA fine-tuned)": {
        "hf_id":      "raulgdp/deepseek14b-acredita",
        "openrouter": "deepseek/deepseek-r1-distill-qwen-14b",   # equivalente cloud
        "params":     "14B",
        "base":       "DeepSeek-R1-Distill-Qwen-14B",
        "desc":       "Fine-tuned sobre corpus CNA/EISC",
    },
    "🏆 ChatAcredita PRO (Qwen3-14B · HuggingFace)": {
        "hf_id":       "Qwen/Qwen3-14B",
        "ollama_model": "chatacredita-pro:latest",   # fallback Ollama (no se usa por defecto)
        "openrouter":  "qwen/qwen3-14b",
        "params":      "14B",
        "base":        "Qwen3-14B-Instruct",
        "desc":        "QLoRA SFT · RAG-VSD · GGUF Q4_K_M local",
    },
}

SYSTEM_ESPANOL = (
    "Eres un asistente experto en acreditacion universitaria colombiana (CNA/EISC). "
    "REGLA ABSOLUTA: Responde SIEMPRE en espanol. "
    "Nunca respondas en ingles ni en otro idioma, sin importar el idioma del contexto. "
    "Si el contexto esta en otro idioma, traduce la respuesta al espanol. "
    "REGLA CRITICA SOBRE EL CONTEXTO: Cuando se te proporciona un bloque <context>...</context>, "
    "DEBES basar tu respuesta EXCLUSIVAMENTE en esa informacion. "
    "PROHIBIDO responder desde tu conocimiento previo si hay contexto disponible. "
    "Si el contexto contiene informacion relevante, usala TODA — no omitas ni resumas. "
    "Organiza la respuesta con estructura clara (listas, secciones) y desarrolla cada punto con detalle. "
    "Una respuesta de un solo parrafo corto es INSUFICIENTE cuando el contexto tiene mas detalles."
)

# System prompt especial para Qwen3 vía Ollama — refuerza el uso del contexto RAG
SYSTEM_QWEN3_OLLAMA = (
    "Eres AcreditaBot, asistente experto en acreditacion universitaria colombiana (CNA/EISC). "
    "INSTRUCCIONES OBLIGATORIAS:\n"
    "1. Responde SIEMPRE en espanol, sin excepcion.\n"
    "2. Cuando el usuario incluya un bloque <context>...</context>, "
    "tu respuesta debe basarse UNICAMENTE en ese contexto. "
    "No uses conocimiento externo si el contexto tiene la informacion.\n"
    "3. Extrae y usa TODA la informacion relevante del contexto. "
    "No hagas resumenes: desarrolla cada punto con los detalles disponibles.\n"
    "4. Si el contexto no contiene la informacion solicitada, indicalo explicitamente.\n"
    "5. Cita las fuentes al final con el formato: [Fuente: nombre_archivo.pdf]"
)

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
# SIDEBAR — CONFIGURACIÓN Y SELECCIÓN DE MODELO
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Escudo_Univalle.svg/200px-Escudo_Univalle.svg.png",
        width=70,
    )
    st.markdown("## 🎓 RAG Acreditación CNA")
    st.caption("EISC · Universidad del Valle")
    st.divider()

    # ── Selección de modelo ──────────────────────────────────────────────────
    st.markdown("### 🤖 Modelo generador")
    selected_model_name = st.radio(
        "Elige el modelo fine-tuneado:",
        list(ACREDITA_MODELS.keys()),
        index=0,
    )
    selected_model = ACREDITA_MODELS[selected_model_name]

    # Tarjeta informativa del modelo seleccionado
    with st.container(border=True):
        st.markdown(f"**{selected_model_name}**")
        st.caption(f"📦 Parámetros: `{selected_model['params']}`")
        st.caption(f"🔗 Base: `{selected_model['base']}`")
        st.caption(f"🎯 {selected_model['desc']}")
        if selected_model.get("hf_id"):
            st.markdown(
                f"[Ver en HuggingFace ↗](https://huggingface.co/{selected_model['hf_id']})",
                unsafe_allow_html=False,
            )

    st.divider()

    # ── Backend de inferencia ────────────────────────────────────────────────
    st.markdown("### ⚙️ Backend de inferencia")

    _backend_options = ["🤗 HuggingFace (local GPU)", "☁️ OpenRouter (nube)", "🦙 Ollama (local GGUF)"]

    # Al cambiar de modelo, proponer el backend apropiado SOLO si el modelo cambió
    _prev_model_key = "prev_selected_model"
    _backend_key    = "backend_index"
    if st.session_state.get(_prev_model_key) != selected_model_name:
        # Modelo recién cambiado → sugerir backend por defecto para este modelo
        # Qwen3 → HuggingFace (índice 0); resto con ollama_model → Ollama (índice 2); otros → HF (0)
        _is_qwen3 = "qwen3" in selected_model.get("base", "").lower()
        st.session_state[_backend_key] = (
            0 if _is_qwen3 or not selected_model.get("ollama_model") else 2
        )
        st.session_state[_prev_model_key] = selected_model_name

    backend = st.radio(
        "Backend de inferencia",
        _backend_options,
        index=st.session_state.get(_backend_key, 0),
        key="backend_radio",
        label_visibility="collapsed",
        help=(
            "**HuggingFace**: descarga automática desde HF Hub, corre con NF4 en tu GPU local.\n"
            "**OpenRouter**: usa el modelo cloud equivalente (requiere API key).\n"
            "**Ollama**: llama al servidor local Ollama en `localhost:11434`."
        ),
    )
    # Persistir selección manual del usuario
    st.session_state[_backend_key] = _backend_options.index(backend)

    if "HuggingFace" in backend:
        hf_note = ""
        if selected_model.get("ollama_model"):
            hf_note = f" También disponible como GGUF en Ollama (`{selected_model['ollama_model']}`)."
        st.info(f"Los pesos se descargan automáticamente desde HuggingFace Hub la primera vez.{hf_note}", icon="🤗")
    elif "OpenRouter" in backend:
        st.info("Usa el modelo cloud equivalente vía OpenRouter. Requiere `OPENROUTER_API_KEY` en `.env`.", icon="☁️")
    else:
        ollama_name = selected_model.get("ollama_model", "chatacredita-pro:latest")
        st.info(f"Llama al modelo `{ollama_name}` en Ollama local (`localhost:11434`).", icon="🦙")

    st.divider()

    # ── Parámetros de generación ─────────────────────────────────────────────
    st.markdown("### ✍️ Generación")
    _default_max = 2048 if bool(selected_model.get("ollama_model")) else 600
    max_tokens_ui = st.slider(
        "Máx. tokens de respuesta",
        min_value=256, max_value=4096, value=_default_max, step=256,
        help="Ollama/Qwen3 admite valores altos sin problema. HF/OpenRouter pueden tener límites propios.",
    )

    # ── Parámetros de recuperación ───────────────────────────────────────────
    st.markdown("### 🔍 Recuperación")
    top_k_cand = st.slider("Candidatos (pre-rerank)", 5, 40, TOP_K_CANDIDATES)
    top_k      = st.slider("Contextos al prompt (post-rerank)", 1, 15, TOP_K_FINAL)

    st.divider()

    # ── Modo de la app ───────────────────────────────────────────────────────
    st.markdown("### 🗂 Modo")
    modo = st.radio(
        "Modo de la app",
        ["💬 Chat libre", "📊 Evaluación batch RAG vs LC"],
        label_visibility="collapsed",
    )

    # Estado del sistema (se rellena tras cargar recursos)
    st.divider()
    st.markdown("### 📡 Estado del sistema")
    status_embed   = st.empty()
    status_rerank  = st.empty()
    status_qdrant  = st.empty()
    status_model   = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE RECURSOS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📥 Cargando BGE-M3 embeddings…")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device), device

@st.cache_resource(show_spinner="📥 Cargando BGE-Reranker-v2-m3…")
def load_reranker():
    try:
        from FlagEmbedding import FlagReranker
        devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
        return FlagReranker(RERANKER_MODEL, use_fp16=True, devices=devices)
    except Exception as e:
        st.sidebar.warning(f"Reranker no disponible: {e}")
        return None

@st.cache_resource(show_spinner="🔌 Conectando a Qdrant…")
def load_qdrant():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    info   = client.get_collection(COLLECTION)
    return client, info.points_count

@st.cache_resource(show_spinner="🤗 Cargando modelo HuggingFace…")
def load_hf_model(hf_id: str):
    """Carga el modelo local con bitsandbytes NF4 en GPU.
    Con 24 GB VRAM: NF4 ocupa ~10 GB para 14B, quedan ~14 GB para contexto RAG.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline as hf_pipe
    is_qwen3 = "qwen3" in hf_id.lower()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    # Reservar 3 GB para embedder + reranker que ya están en GPU
    max_mem = {0: f"{max(1, int(vram_gb - 3))}GiB", "cpu": "48GiB"}
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id, quantization_config=bnb,
        device_map="auto", max_memory=max_mem, trust_remote_code=True,
    )
    # Qwen3: do_sample=False + temperature irrelevante; otros: leve sampling
    pipe = hf_pipe(
        "text-generation", model=mdl, tokenizer=tok,
        max_new_tokens=512,           # sobreescrito en call_model() con slider
        do_sample=False,              # greedy → más fiel al contexto RAG
        repetition_penalty=1.05,      # suave para evitar loops sin degradar calidad
    )
    return pipe, tok

@st.cache_data(show_spinner="📚 Cargando corpus Long-Context…")
def load_lc_corpus():
    import fitz
    try:
        import pymupdf4llm
        use_pymupdf4llm = True
    except ImportError:
        use_pymupdf4llm = False

    if os.path.exists(LC_CORPUS_CACHE):
        with open(LC_CORPUS_CACHE, "rb") as f:
            return pickle.load(f)

    pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))
    if not pdf_files:
        return []

    corpus = []
    for p in pdf_files:
        doc  = fitz.open(p)
        text = ""
        for pn in range(len(doc)):
            try:
                if use_pymupdf4llm:
                    import pymupdf4llm
                    page_md = pymupdf4llm.to_markdown(doc, pages=[pn], show_progress=False, page_chunks=False)
                else:
                    page_md = doc[pn].get_text()
            except Exception:
                page_md = doc[pn].get_text()
            text += f"\n--- Pag {pn+1} ---\n{page_md}"
        doc.close()
        if text.strip():
            corpus.append({"title": p.name, "text": text})

    with open(LC_CORPUS_CACHE, "wb") as f:
        pickle.dump(corpus, f)
    return corpus

# Cargar recursos base
embedder, device = load_embedder()
reranker         = load_reranker()
qdrant_client, n_chunks = load_qdrant()

# Actualizar indicadores sidebar
status_embed.success(f"✅ BGE-M3 ({device})")
status_rerank.success(f"✅ Reranker: {'OK' if reranker else '⚠️ no disponible'}")
status_qdrant.success(f"✅ Qdrant: {n_chunks:,} chunks")
_backend_short = "Ollama" if "Ollama" in backend else ("HF" if "HuggingFace" in backend else "OpenRouter")
status_model.info(f"🤖 {selected_model['params']} · {_backend_short}")

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES CORE
# ─────────────────────────────────────────────────────────────────────────────
def to_ascii(text: str) -> str:
    return text.encode("ascii", errors="ignore").decode("ascii")


def retrieve(query: str) -> list[dict]:
    """BGE-M3 → Qdrant top-K → BGE-Reranker → top_k final."""
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()

    hits = qdrant_client.query_points(
        collection_name=COLLECTION,
        query=q_emb,
        limit=top_k_cand,
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

    return candidates[:top_k]


def call_model(prompt: str, max_tokens: int = 600) -> tuple[str, float]:
    """
    Despacha la llamada al backend seleccionado en sidebar.
    Retorna (respuesta_limpia, latencia_s).
    """
    # Usar el límite configurado en sidebar (max_tokens_ui) salvo que el llamador
    # pida explícitamente uno menor (ej: judge_correct usa max_tokens=5)
    if max_tokens >= 600:
        max_tokens = max_tokens_ui
    t0 = time.time()

    # ── A) HuggingFace local ──────────────────────────────────────────────
    if "HuggingFace" in backend:
        try:
            pipe, tok = load_hf_model(selected_model["hf_id"])
            # Qwen3 vía HF: usa system prompt reforzado para grounding en contexto RAG
            is_qwen3  = "qwen3" in selected_model.get("hf_id", "").lower()
            sys_msg   = SYSTEM_QWEN3_OLLAMA if is_qwen3 else SYSTEM_ESPANOL
            # to_ascii evita errores charmap en Windows con tildes/ñ en el contexto
            messages  = [
                {"role": "system", "content": to_ascii(sys_msg)},
                {"role": "user",   "content": to_ascii(prompt)},
            ]
            # Qwen3 soporta enable_thinking; desactivarlo para respuestas directas
            try:
                text_in = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **({"enable_thinking": False} if is_qwen3 else {}),
                )
            except TypeError:
                # Tokenizer antiguo sin soporte enable_thinking
                text_in = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Asegurar que text_in sea str UTF-8 válido antes de pasarlo al pipe
            if isinstance(text_in, bytes):
                text_in = text_in.decode("utf-8", errors="replace")

            # Respetar el slider max_tokens del sidebar
            out     = pipe(text_in, max_new_tokens=max_tokens)[0]["generated_text"]
            answer  = out[len(text_in):].strip()
            # Quitar bloque <think> si es DeepSeek-R1 o Qwen3 con thinking activo
            answer  = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            # Decodificar correctamente si el modelo devuelve bytes o latin-1
            if isinstance(answer, bytes):
                answer = answer.decode("utf-8", errors="replace")
            return answer, time.time() - t0
        except Exception as e:
            return f"❌ Error HuggingFace: {e}", time.time() - t0

    # ── B) Ollama local ───────────────────────────────────────────────────
    elif "Ollama" in backend:
        try:
            ollama_model = selected_model.get("ollama_model", "chatacredita-pro:latest")
            client = OpenAI(
                api_key="ollama",                              # Ollama no requiere key real
                base_url="http://localhost:11434/v1",
            )
            # Qwen3 vía Ollama usa system prompt reforzado para grounding en contexto RAG
            is_qwen3_ollama = "qwen3" in selected_model.get("base", "").lower()
            sys_content = SYSTEM_QWEN3_OLLAMA if is_qwen3_ollama else SYSTEM_ESPANOL
            resp = client.chat.completions.create(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,   # más bajo para mayor fidelidad al contexto
                timeout=180,
                extra_body={"options": {"num_ctx": 16384}},  # ampliar ventana para RAG context
            )
            raw   = resp.choices[0].message.content or ""
            # Quitar bloque <think> si el modelo usa razonamiento extendido
            clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return clean or raw.strip(), time.time() - t0
        except Exception as e:
            return f"❌ Error Ollama: {e}\n(¿Está corriendo `ollama serve`?)", time.time() - t0

    # ── C) OpenRouter (fallback nube) ─────────────────────────────────────
    else:
        try:
            client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
            resp = client.chat.completions.create(
                model=selected_model["openrouter"],
                messages=[
                    {"role": "system", "content": SYSTEM_ESPANOL},
                    {"role": "user",   "content": to_ascii(prompt)},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                timeout=60,
                extra_headers=OPENROUTER_HEADERS,
            )
            raw   = resp.choices[0].message.content or ""
            clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return clean or raw.strip(), time.time() - t0
        except Exception as e:
            return f"❌ Error OpenRouter: {e}", time.time() - t0


def answer_with_rag(question: str) -> dict:
    docs    = retrieve(question)
    context = "\n\n---\n\n".join(
        f"[Fuente: {d['source']}]\n{d['text']}" for d in docs
    )
    # Formato <context> que Qwen3 reconoce mejor para grounding
    prompt = (
        "/no_think\n"  # desactiva chain-of-thought interno en Qwen3 GGUF vía Ollama
        "<context>\n"
        f"{context}\n"
        "</context>\n\n"
        f"Usando EXCLUSIVAMENTE la informacion del contexto anterior, "
        f"responde de forma COMPLETA y DETALLADA la siguiente pregunta.\n"
        f"Incluye TODOS los datos relevantes del contexto. "
        f"Organiza con listas o secciones si hay multiples aspectos. "
        f"Cita las fuentes al final: [Fuente: nombre_archivo.pdf]\n\n"
        f"PREGUNTA: {question}\n\n"
        f"RESPUESTA DETALLADA EN ESPANOL:"
    )
    answer, latency = call_model(prompt)
    return {"answer": answer, "latency_s": latency, "docs": docs}


def answer_with_long_context(question: str, full_context: str) -> dict:
    prompt = (
        "/no_think\n"
        "<context>\n"
        f"{full_context}\n"
        "</context>\n\n"
        f"Usando EXCLUSIVAMENTE la informacion del contexto anterior, "
        f"responde de forma COMPLETA y DETALLADA la siguiente pregunta.\n"
        f"Incluye TODOS los datos relevantes. "
        f"Organiza con listas o secciones si hay multiples aspectos.\n\n"
        f"PREGUNTA: {question}\n\n"
        "RESPUESTA DETALLADA EN ESPANOL:"
    )
    answer, latency = call_model(prompt)
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
    answer, _ = call_model(prompt, max_tokens=5)
    return answer.strip().upper().startswith("S")


# ─────────────────────────────────────────────────────────────────────────────
# ENCABEZADO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎓 RAG Acreditación CNA — EISC Universidad del Valle")
backend_label = backend.split("(")[0].strip()
st.caption(
    f"**Modelo activo:** {selected_model_name}  ·  "
    f"**Backend:** {backend_label}  ·  "
    f"**Retrieval:** BGE-M3 → Qdrant `{COLLECTION}` → BGE-Reranker"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# MODO 1: CHAT LIBRE — Estilo WhatsApp
# ─────────────────────────────────────────────────────────────────────────────
if modo == "💬 Chat libre":

    st.markdown("""
    <style>
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
    .wa-sender {
        font-size: 0.74rem;
        color: #53bdeb;
        font-weight: 600;
        margin-bottom: 3px;
    }
    .wa-time { font-size: 0.68rem; color: #8696a0; text-align: right; margin-top: 4px; }
    .wa-sources {
        font-size: 0.72rem; color: #8696a0;
        margin-top: 4px; padding-left: 4px;
        border-left: 2px solid #53bdeb;
    }
    .wa-latency { font-size: 0.68rem; color: #8696a0; }
    .wa-model-tag {
        font-size: 0.68rem; color: #25d366;
        margin-bottom: 2px; font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    if "wa_history" not in st.session_state:
        st.session_state.wa_history = []

    modo_resp = st.sidebar.radio(
        "Modo de recuperación",
        ["RAG (BGE-M3 + Qdrant)", "Long-Context (corpus completo)"],
        key="chat_modo_resp",
    )

    def _ts():
        return datetime.datetime.now().strftime("%H:%M")

    # Render historial
    chat_parts = []
    for msg in st.session_state.wa_history:
        ts = msg.get("ts", "")
        if msg["role"] == "user":
            chat_parts.append(
                f'<div class="wa-msg-user">{msg["text"]}'
                f'<div class="wa-time">{ts}</div></div>'
            )
        else:
            srcs = ""
            if msg.get("sources"):
                src_list = " · ".join(f"📄 {os.path.basename(str(s))}" for s in msg["sources"])
                srcs = f'<div class="wa-sources">{src_list}</div>'
            lat = f'<div class="wa-latency">⏱ {msg["latency_s"]:.1f}s</div>' if msg.get("latency_s") else ""
            mdl_tag = f'<div class="wa-model-tag">🤖 {msg.get("model_name", "")}</div>'
            chat_parts.append(
                f'<div class="wa-msg-bot">'
                f'<div class="wa-sender">🎓 AcreditaBot</div>'
                f'{mdl_tag}'
                f'{msg["text"]}'
                f'{srcs}{lat}'
                f'<div class="wa-time">{ts}</div>'
                f'</div>'
            )

    welcome = (
        '<div class="wa-msg-bot">'
        '<div class="wa-sender">🎓 AcreditaBot</div>'
        'Hola 👋 Soy tu asistente de acreditación CNA/EISC. '
        '¿En qué puedo ayudarte?'
        '<div class="wa-time">ahora</div>'
        '</div>'
    )
    html_content = "".join(chat_parts) if chat_parts else welcome
    st.markdown(f'<div class="wa-chat-wrapper">{html_content}</div>', unsafe_allow_html=True)

    # Input bar
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        query = st.text_input(
            "Mensaje", placeholder="Escribe tu pregunta sobre acreditación…",
            label_visibility="collapsed", key="wa_input",
        )
    with col_btn:
        send = st.button("➤", type="primary", use_container_width=True)

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("🗑 Limpiar chat"):
            st.session_state.wa_history = []
            st.rerun()

    if send and query.strip():
        st.session_state.wa_history.append({"role": "user", "text": query, "ts": _ts()})

        if "RAG" in modo_resp:
            with st.spinner("🔍 Buscando en Qdrant + reranking…"):
                result = answer_with_rag(query)
            sources = [d["source"] for d in result["docs"]]

            with st.expander(f"📚 Chunks recuperados (top-{top_k})", expanded=False):
                for i, d in enumerate(result["docs"], 1):
                    st.markdown(
                        f"**[{i}]** `score={d['rerank_score']:.3f}` · "
                        f"📄 `{os.path.basename(str(d['source']))}`"
                    )
                    st.caption(d["text"][:300] + "…")
                    st.divider()

            st.session_state.wa_history.append({
                "role": "bot", "text": result["answer"],
                "sources": sources, "latency_s": result["latency_s"],
                "model_name": selected_model_name, "ts": _ts(),
            })

        else:  # Long-Context
            with st.spinner("📚 Cargando corpus y generando respuesta…"):
                lc_corpus = load_lc_corpus()
                if not lc_corpus:
                    st.warning(f"No se encontraron PDFs en `{PDF_DIR}/`")
                else:
                    full_ctx = "\n\n===\n\n".join(
                        f"[Fuente: {d['title']}]\n{d['text']}" for d in lc_corpus
                    )[:LC_MAX_CHARS]
                    result = answer_with_long_context(query, full_ctx)
                    st.session_state.wa_history.append({
                        "role": "bot", "text": result["answer"],
                        "sources": [], "latency_s": result["latency_s"],
                        "model_name": selected_model_name, "ts": _ts(),
                    })

        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MODO 2: EVALUACIÓN BATCH
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.subheader("📊 Evaluación comparativa: RAG vs Long-Context")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Preguntas", len(QUESTIONS))
    col_info2.metric("Modelo activo", selected_model["params"])
    col_info3.metric("Backend", backend.split("(")[0].strip())

    st.info(
        f"Se evalúan **{len(QUESTIONS)} preguntas** predefinidas sobre acreditación CNA/EISC. "
        f"El mismo modelo (`{selected_model_name}`) actúa como juez para determinar si la respuesta es correcta. "
        "Los resultados se cachean en disco."
    )

    col_run, col_clear = st.columns([1, 4])
    run_eval    = col_run.button("▶️ Ejecutar evaluación", type="primary")
    clear_cache = col_clear.button("🗑 Borrar caché y re-evaluar")

    if clear_cache and os.path.exists(RESULTS_CACHE):
        os.remove(RESULTS_CACHE)
        st.success("Caché borrado. Ejecuta la evaluación de nuevo.")

    results = []
    if os.path.exists(RESULTS_CACHE):
        with open(RESULTS_CACHE, "rb") as f:
            results = pickle.load(f)

    if run_eval:
        lc_corpus = load_lc_corpus()
        full_ctx  = ""
        if lc_corpus:
            full_ctx = "\n\n===\n\n".join(
                f"[Fuente: {d['title']}]\n{d['text']}" for d in lc_corpus
            )[:LC_MAX_CHARS]
        else:
            st.warning(f"No se encontraron PDFs en `{PDF_DIR}/` — Long-Context no disponible")

        results  = []
        progress = st.progress(0, text="Iniciando evaluación…")
        status   = st.empty()

        for qi, q in enumerate(QUESTIONS):
            progress.progress(qi / len(QUESTIONS), text=f"Pregunta {qi+1}/{len(QUESTIONS)}")
            status.markdown(f"**Evaluando:** {q['question'][:80]}…")

            rag = answer_with_rag(q["question"])
            rag_correct = judge_correct(q["question"], q["reference_answer"], rag["answer"])

            if full_ctx:
                lc = answer_with_long_context(q["question"], full_ctx)
                lc_correct = judge_correct(q["question"], q["reference_answer"], lc["answer"])
            else:
                lc = {"answer": "N/A — sin PDFs", "latency_s": 0.0}
                lc_correct = False

            results.append({
                "question":         q["question"],
                "reference_answer": q["reference_answer"],
                "model":            selected_model_name,
                "backend":          backend,
                "rag_answer":       rag["answer"],
                "rag_correct":      rag_correct,
                "rag_latency_s":    rag["latency_s"],
                "rag_sources":      [d["source"] for d in rag["docs"]],
                "lc_answer":        lc["answer"],
                "lc_correct":       lc_correct,
                "lc_latency_s":     lc["latency_s"],
            })
            time.sleep(1)

        with open(RESULTS_CACHE, "wb") as f:
            pickle.dump(results, f)

        progress.progress(1.0, text="✅ Evaluación completada")
        status.empty()

    if results:
        df = pd.DataFrame(results)

        # ── Métricas resumen ────────────────────────────────────────────────
        st.markdown("### 📈 Resultados")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RAG — Exactitud",
                  f"{df['rag_correct'].mean()*100:.0f}%",
                  f"{int(df['rag_correct'].sum())}/{len(df)} correctas")
        c2.metric("LC — Exactitud",
                  f"{df['lc_correct'].mean()*100:.0f}%",
                  f"{int(df['lc_correct'].sum())}/{len(df)} correctas")
        c3.metric("RAG — Latencia media",  f"{df['rag_latency_s'].mean():.1f}s")
        c4.metric("LC — Latencia media",   f"{df['lc_latency_s'].mean():.1f}s")

        # ── Gráficas ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = ["RAG\n(BGE-M3+Qdrant)", "Long-Context"]
        colors = ["#2563eb", "#e74c3c"]

        axes[0].bar(labels, [df["rag_correct"].mean(), df["lc_correct"].mean()], color=colors)
        axes[0].set_title(f"Exactitud (n={len(df)})")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Fracción correcta")
        for bar, val in zip(axes[0].patches, [df["rag_correct"].mean(), df["lc_correct"].mean()]):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f"{val*100:.0f}%", ha="center", fontsize=11, fontweight="bold")

        axes[1].bar(labels, [df["rag_latency_s"].mean(), df["lc_latency_s"].mean()], color=colors)
        axes[1].set_title("Latencia promedio (s)")
        axes[1].set_ylabel("segundos")

        model_short = selected_model["params"]
        plt.suptitle(f"RAG vs Long-Context · {model_short} · {backend.split('(')[0].strip()}",
                     fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Tabla detallada ──────────────────────────────────────────────────
        st.markdown("### 📋 Resultados por pregunta")
        for _, row in df.iterrows():
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
                st.caption(f"📌 Referencia: {row['reference_answer']}")

        # ── Exportar ─────────────────────────────────────────────────────────
        csv = df.drop(columns=["rag_sources"]).to_csv(index=False)
        st.download_button(
            "⬇️ Descargar resultados CSV",
            data=csv.encode("utf-8"),
            file_name="resultados_rag_acredita.csv",
            mime="text/csv",
        )
    else:
        st.info("Presiona **▶️ Ejecutar evaluación** para comenzar.")
