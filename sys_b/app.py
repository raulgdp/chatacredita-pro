# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RAFT-Chat v2 — Correcciones aplicadas                                      ║
# ║  FIX 1: load_raft_dataset lee messages[user/assistant] correctamente        ║
# ║  FIX 2: limpiar_chunk sin filtro isalpha() que borraba cifras/siglas        ║
# ║  FIX 3: Búsqueda híbrida sobre Qdrant (127k chunks) en vez del dataset RAFT ║
# ║  FIX 4: Prompt RAFT fiel al paper (chain-of-thought + ##begin_quote##)      ║
# ║  FIX 5: Contexto enriquecido con pares Q/A del dataset RAFT                 ║
# ║  EISC — Universidad del Valle, Cali, Colombia                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import re
import json
import time
import unicodedata
import concurrent.futures

import numpy as np
from openai import OpenAI
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAFT-Chat · EISC Univalle",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── OpenRouter / LLM ──────────────────────────────────────────────────────────

OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY", "os.getenv("OPENROUTER_API_KEY", "")")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = "deepseek/deepseek-r1-0528"
# Alias con nombre correcto -- estas variables se llamaban "GROQ_*" porque
# el script uso la API de Groq en una version anterior; el nombre nunca se
# actualizo al migrar despues a OpenRouter y luego a Ollama. No cambia nada
# funcionalmente (mismo valor), solo aclara que hoy corre sobre Ollama local.
LLM_BASE     = OPENROUTER_BASE
LLM_API_KEY  = OPENROUTER_KEY
LLM_MODEL    = OPENROUTER_MODEL

# ── Qdrant (mismo de main.py) ──────────────────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL",     "os.getenv("QDRANT_URL", "")")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "os.getenv("QDRANT_API_KEY", "")")  # SOLO desde .env -- nunca hardcodear la key aqui
COLLECTION_NAME   = "acreditacion"               # colección principal del RAG

# ── Dataset RAFT (solo para enriquecer contextos) ──────────────────────────────
DATASET_PATH = "dataset_raft_v4.json"

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 20   # candidatos al reranker
TOP_K_FINAL     = 8    # resultado final para el prompt RAFT
BM25_TTL        = 3600 # segundos de caché del índice BM25

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .raft-badge {
        display: inline-block;
        background: linear-gradient(90deg,#1a3a5c,#2563eb);
        color: white; border-radius: 6px;
        padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
    }
    .doc-card {
        border-left: 3px solid #2563eb;
        padding: 6px 12px; margin-bottom: 6px;
        background: #f0f4ff; border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
    }
    .score-bar { color: #2563eb; font-weight: 700; }
    .raft-cite {
        background: #fffbeb; border-left: 3px solid #f59e0b;
        padding: 4px 10px; margin: 4px 0;
        border-radius: 0 4px 4px 0; font-style: italic;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: CARGA DEL DATASET RAFT  
# Lee correctamente la estructura messages[user/assistant]
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📂 Cargando dataset RAFT…")
def load_raft_qa_index(path: str) -> dict[str, list[dict]]:
    """
    Carga el dataset RAFT y construye un índice contexto → [(pregunta, respuesta), ...].
    FIX REAL: dataset_raft_v4.json usa el esquema {pregunta, documentos,
    oracle_incluido, oracle_chunk, respuesta_corta, respuesta_cot, fuente,
    es_tabla} -- NO {contexto, messages} del formato viejo. La version
    anterior nunca encontraba nada porque buscaba campos inexistentes.
    """
    if not os.path.exists(path):
        st.warning(f"⚠️ Dataset RAFT no encontrado: {path}. Continuando sin enriquecimiento.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ctx_qa: dict[str, list[dict]] = {}
    for item in data:
        # El contexto real es el oracle_chunk (el documento que de verdad
        # contiene la respuesta). Si oracle_incluido=False, oracle_chunk es
        # None -- ese registro es de abstencion, no hay contexto que enriquecer.
        ctx = (item.get("oracle_chunk") or "").strip()
        pregunta = (item.get("pregunta") or "").strip()
        respuesta = (item.get("respuesta_corta") or "").strip()

        if ctx and pregunta:
            if ctx not in ctx_qa:
                ctx_qa[ctx] = []
            ctx_qa[ctx].append({"pregunta": pregunta, "respuesta": respuesta})

    st.sidebar.success(f"✅ {len(data):,} ejemplos RAFT cargados ({len(ctx_qa):,} contextos únicos)")
    return ctx_qa


# ─────────────────────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📥 Cargando BGE-M3…")
def load_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("BAAI/bge-m3", device=device)


@st.cache_resource(show_spinner="📥 Cargando BGE-Reranker-v2-m3…")
def load_reranker():
    try:
        from FlagEmbedding import FlagReranker
        device = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
        return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices=device)
    except ImportError:
        st.warning("⚠️ FlagEmbedding no instalado. Sin reranker. `pip install FlagEmbedding`")
        return None


@st.cache_resource(show_spinner="🔌 Conectando con Qdrant…")
def load_qdrant():
    """
    FIX 3: usa el mismo Qdrant de main.py (colección 'acreditacion' con ~127k chunks).
    Esto reemplaza el corpus de búsqueda del dataset RAFT (solo 15k fragmentos de 154 chars).
    """
    if QDRANT_URL:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        # Fallback a memoria si no hay URL configurada
        client = QdrantClient(":memory:")
        st.sidebar.warning("⚠️ Sin Qdrant URL: usando memoria. Configura QDRANT_URL en .env")
    try:
        info = client.get_collection(COLLECTION_NAME)
        st.sidebar.success(f"✅ Qdrant: {info.points_count:,} chunks en '{COLLECTION_NAME}'")
    except Exception as e:
        st.sidebar.error(f"❌ Qdrant: {e}")
    return client


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: BÚSQUEDA HÍBRIDA SOBRE QDRANT + BM25
# Reemplaza la búsqueda sobre el dataset RAFT (15k fragmentos cortos)
# por búsqueda sobre la colección Qdrant completa (~127k chunks de 1800 chars)
# ─────────────────────────────────────────────────────────────────────────────
_bm25_cache: dict = {"ts": 0, "index": None, "texts": [], "ids": [], "sources": []}
BM25_SCROLL_LIMIT = 200  # chunks por lote en el scroll

def get_bm25(qdrant_client: QdrantClient) -> tuple:
    """Construye índice BM25 sobre la colección Qdrant (con caché TTL)."""
    now = time.time()
    if now - _bm25_cache["ts"] < BM25_TTL and _bm25_cache["index"] is not None:
        return _bm25_cache["index"], _bm25_cache["texts"], _bm25_cache["ids"], _bm25_cache["sources"]

    texts, ids, sources = [], [], []
    offset = None
    while True:
        res = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=BM25_SCROLL_LIMIT,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in res[0]:
            txt = (pt.payload or {}).get("text", "").strip()
            if txt:
                texts.append(txt)
                ids.append(str(pt.id))
                # BUG CORREGIDO: antes se descartaba la fuente del payload
                # aqui mismo, forzando "source": "" para TODO resultado que
                # viniera de BM25 -- por eso aparecian documentos con fuente
                # en blanco en la interfaz.
                sources.append((pt.payload or {}).get("source", ""))
        offset = res[1]
        if offset is None:
            break

    if not texts:
        return None, [], [], []

    def _norm(t: str) -> str:
        t = unicodedata.normalize("NFD", t.lower())
        return "".join(c for c in t if unicodedata.category(c) != "Mn")

    tokenized = [_norm(t).split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    _bm25_cache.update({"ts": now, "index": bm25, "texts": texts, "ids": ids, "sources": sources})
    return bm25, texts, ids, sources


def _dense_search_qdrant(query: str, embedder, qdrant_client: QdrantClient,
                          top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Búsqueda densa en Qdrant sobre la colección completa."""
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    hits = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        {
            "id":    str(h.id),
            "text":  (h.payload or {}).get("text", ""),
            "score": h.score,
            "source": (h.payload or {}).get("source", ""),
        }
        for h in hits
    ]


def hybrid_rrf_search_qdrant(
    query: str,
    embedder,
    qdrant_client: QdrantClient,
    top_k: int = TOP_K_RETRIEVAL,
    k_rrf: int = 60,
) -> list[dict]:
    """
    FIX 3: Búsqueda híbrida BM25 + Dense sobre Qdrant en paralelo → RRF.
    Mismo pipeline que main.py pero desde Streamlit.
    """
    def _norm(t: str) -> str:
        t = unicodedata.normalize("NFD", t.lower())
        return "".join(c for c in t if unicodedata.category(c) != "Mn")

    bm25, texts, ids, sources = get_bm25(qdrant_client)

    def do_bm25():
        if bm25 is None:
            return []
        scores = bm25.get_scores(_norm(query).split())
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            {"id": ids[i], "text": texts[i], "score": float(scores[i]), "source": sources[i]}
            for i in ranked if scores[i] > 0
        ]

    def do_dense():
        return _dense_search_qdrant(query, embedder, qdrant_client, top_k)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_bm25  = ex.submit(do_bm25)
        f_dense = ex.submit(do_dense)
        bm25_res  = f_bm25.result()
        dense_res = f_dense.result()

    # RRF fusion
    rrf: dict[str, float] = {}
    id_to_payload: dict[str, dict] = {}

    # BUG CORREGIDO: se procesa PRIMERO dense (que siempre trae "source"
    # correctamente) y BM25 despues -- como el diccionario solo guarda el
    # payload la PRIMERA vez que ve un id, antes bastaba con que BM25
    # encontrara un chunk primero para que su "source" vacio quedara fijo
    # aunque dense tambien lo hubiera encontrado con la fuente correcta.
    for rank, r in enumerate(dense_res):
        rrf[r["id"]] = rrf.get(r["id"], 0.0) + 1.0 / (k_rrf + rank + 1)
        if r["id"] not in id_to_payload:
            id_to_payload[r["id"]] = r

    for rank, r in enumerate(bm25_res):
        rrf[r["id"]] = rrf.get(r["id"], 0.0) + 1.0 / (k_rrf + rank + 1)
        if r["id"] not in id_to_payload:
            id_to_payload[r["id"]] = r

    sorted_ids = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {**id_to_payload[pid], "rrf_score": round(score, 5)}
        for pid, score in sorted_ids
        if pid in id_to_payload
    ]


def _son_casi_iguales(texto_a: str, texto_b: str, umbral: float = 0.85) -> bool:
    """Similitud simple por overlap de palabras -- suficiente para detectar
    el mismo parrafo institucional repetido en varias fuentes distintas."""
    palabras_a = set(texto_a.lower().split())
    palabras_b = set(texto_b.lower().split())
    if not palabras_a or not palabras_b:
        return False
    interseccion = len(palabras_a & palabras_b)
    menor = min(len(palabras_a), len(palabras_b))
    return (interseccion / menor) > umbral


def diversificar_por_contenido(candidatos_ordenados: list[dict], max_similares: int = 2) -> list[dict]:
    """
    BUG CORREGIDO: antes, si el mismo parrafo institucional aparecia
    repetido (casi textual) en varias fuentes distintas (comun en
    documentos de acreditacion que se citan entre si), todos esos
    duplicados podian llenar el top-k completo, dejando cero espacio
    para informacion complementaria -- especialmente grave en preguntas
    comparativas ("diferencias entre X y Y"), donde uno de los dos
    terminos de la comparacion terminaba sin ningun documento.
    Aqui permitimos como mucho 'max_similares' copias casi identicas de
    un mismo parrafo, dejando que el resto de los cupos se llenen con
    contenido genuinamente distinto.
    """
    resultado = []
    for cand in candidatos_ordenados:
        similares_ya_incluidos = sum(
            1 for r in resultado if _son_casi_iguales(cand["text"], r["text"])
        )
        if similares_ya_incluidos < max_similares:
            resultado.append(cand)
    return resultado


def rerank(query: str, candidates: list[dict], reranker,
           top_k: int = TOP_K_FINAL) -> list[dict]:
    """BGE-Reranker-v2-m3: score real de relevancia. Fallback: RRF."""
    if not candidates:
        return []
    if reranker is not None:
        try:
            pairs  = [[query, c["text"][:1024]] for c in candidates]
            scores = reranker.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
                c["final_score"]  = 0.7 * c["rerank_score"] + 0.3 * c["rrf_score"]
        except Exception as e:
            st.warning(f"Reranker error: {e}")
            for c in candidates:
                c["final_score"] = c["rrf_score"]
    else:
        for c in candidates:
            c["final_score"] = c["rrf_score"]

    ordenados = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    diversificados = diversificar_por_contenido(ordenados, max_similares=2)
    return diversificados[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: limpiar_chunk — sin filtro isalpha() que eliminaba cifras y siglas
# ─────────────────────────────────────────────────────────────────────────────
def limpiar_chunk(texto: str) -> str:
    """
    Limpia el chunk de ruido HTML/markdown preservando:
    - Cifras y porcentajes (tasas de graduación, fechas, resoluciones)
    - Siglas (CNA, ICFES, SNIES)
    - Texto con guiones, puntos y comas
    FIX: eliminado el filtro isalpha() que borraba datos críticos de acreditación.
    """
    texto = re.sub(r"\([^\)]+\)\([^\)]+\)", r"", texto)  # links markdown
    texto = re.sub(r"https?://\S+", "", texto)                  # URLs
    texto = re.sub(r"~~[^~]*~~", "", texto)                      # tachado
    texto = re.sub(r"<[^>]+>", " ", texto)                       # HTML
    texto = re.sub(r"\[\s*\]", "", texto)                     # corchetes vacíos

    lineas = texto.split("\n")
    lineas_limpias = []
    for l in lineas:
        # FIX: solo eliminar líneas que sean puramente ruido (menos de 3 tokens)
        # NO filtrar por isalpha() — eso eliminaba números, siglas, fechas
        tokens = l.split()
        if len(tokens) >= 3:
            lineas_limpias.append(l)

    texto = "\n".join(lineas_limpias)
    return re.sub(r"\s+", " ", texto).strip()


# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 + FIX 5: PROMPT RAFT MEJORADO
# Fiel al paper: chain-of-thought + cita ##begin_quote##...##end_quote##
# FIX 5: enriquece el contexto con pares Q/A del dataset RAFT si aplica
# ─────────────────────────────────────────────────────────────────────────────
RAFT_SYSTEM = (
    "Eres un asistente academico experto en acreditacion universitaria de la "
    "Universidad del Valle (EISC) y el sistema CNA de Colombia. "
    "Tu metodo de respuesta es:\n"
    "1. Lee TODOS los documentos de contexto.\n"
    "2. Identifica el fragmento mas relevante.\n"
    "3. Razona brevemente sobre ese fragmento.\n"
    "4. Cita con ##begin_quote## texto exacto ##end_quote##.\n"
    "5. Da la respuesta final basada UNICAMENTE en los documentos.\n"
    "Si no encuentras la informacion, di que no la encontraste."
)


def build_raft_prompt(query: str, docs: list[dict],
                      raft_qa_index: dict[str, list]) -> str:
    """
    FIX 4: Prompt fiel al paper RAFT con citas ##begin_quote##...##end_quote##.
    FIX 5: Enriquece cada documento con pares Q/A del índice RAFT si el contexto
           del chunk coincide con algún ejemplo entrenado.
    """
    ctx_block = ""
    for i, d in enumerate(docs, 1):
        ctx_raw   = d["text"]
        ctx_clean = limpiar_chunk(ctx_raw)
        if len(ctx_clean) < 80:
            ctx_clean = ctx_raw[:1000]

        # FIX 5: buscar pares Q/A RAFT que correspondan a este chunk
        # (coincidencia por subcadena ya que los contextos del dataset son subconjuntos)
        raft_enrichment = ""
        for ctx_raft, pares in raft_qa_index.items():
            if ctx_raft[:80] in ctx_raw[:500] or ctx_raw[:80] in ctx_raft:
                qa_lines = "\n".join(
                    f"  Q: {p['pregunta']} → A: {p['respuesta'][:150]}"
                    for p in pares[:2]
                )
                if qa_lines:
                    raft_enrichment = f"\n[Pares conocidos de este fragmento:]\n{qa_lines}"
                break

        fuente = d.get("source", f"doc_{i}")
        if fuente:
            fuente_tag = f"Fuente: {os.path.basename(str(fuente))}"
        else:
            fuente_tag = f"Fragmento {i}"

        ctx_block += f"[Documento {i} · {fuente_tag}]\n{ctx_clean}{raft_enrichment}\n\n"

    return (
        f"### Documentos de contexto:\n\n{ctx_block}"
        f"### Pregunta:\n{query}\n\n"
        f"### Instrucciones:\n"
        f"Sigue el método RAFT: identifica el fragmento relevante, razona, "
        f"cita con ##begin_quote##...##end_quote## y da la respuesta final.\n"
        f"Si la información está distribuida en varios documentos, combínala.\n"
        f"Solo di \'No encontré información\' si los documentos son completamente irrelevantes.\n\n"
        f"### Respuesta:"
    )


def sanitize_for_api(text: str) -> str:
    """
    Sanitiza texto para enviarlo a OpenRouter sin errores de encoding.
    Maneja caracteres Latin-1 que vienen de PDFs indexados con ese encoding.
    """
    if not text:
        return ""
    # 1. Intentar decodificar como Latin-1 si hay bytes sueltos
    try:
        text = text.encode("latin-1").decode("utf-8", errors="replace")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    # 2. Normalizar unicode compuesto
    text = unicodedata.normalize("NFKD", text)
    # 3. Reemplazos explícitos de los caracteres más problemáticos
    replacements = {
        "\xb7": ".",   # middle dot Latin-1 → punto
        "\u00b7": ".", # middle dot unicode → punto
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u00e1": "a", "\u00e9": "e", "\u00ed": "i",
        "\u00f3": "o", "\u00fa": "u", "\u00f1": "n",
        "\u00c1": "A", "\u00c9": "E", "\u00cd": "I",
        "\u00d3": "O", "\u00da": "U", "\u00d1": "N",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # 4. Forzar ASCII puro — eliminar cualquier carácter fuera del rango
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return text


def generate_raft_response(
    query: str,
    docs: list[dict],
    raft_qa_index: dict[str, list],
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> str:
    def to_ascii(s: str) -> str:
        """Convierte cualquier string a ASCII puro sin excepciones."""
        return s.encode("ascii", errors="ignore").decode("ascii")

    user_content  = build_raft_prompt(query, docs, raft_qa_index)
    user_content  = to_ascii(user_content)
    system_prompt = to_ascii(RAFT_SYSTEM)
    model_name    = to_ascii(LLM_MODEL)
    
    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE)
        r = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60,
            extra_headers={
                "HTTP-Referer": "https://eisc.univalle.edu.co",
                "X-Title": "ChatAcredita RAFT EISC Univalle",
            },
        )
        resp = r.choices[0].message.content.strip()

        # BUG CORREGIDO: si el modelo mete TODA la respuesta dentro de
        # <think>...</think> (comun en modelos con razonamiento, como
        # gemma4), quitar ese bloque puede dejar el texto completamente
        # vacio. Antes se sobreescribia resp sin chequear esto -- ahora,
        # si limpiar <think> deja vacio, nos quedamos con el original en
        # vez de mostrar nada.
        sin_think = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
        resp = sin_think if sin_think else resp

        if not resp:
            return ("⚠️ El modelo no generó ningún texto de respuesta (posible corte por "
                    "límite de tokens o salida vacía). Intenta subir 'Max tokens' en la "
                    "barra lateral, o reintenta la pregunta.")

        # Avisar si la respuesta se corto por limite de tokens, en vez de
        # dejar al usuario adivinando por que la respuesta quedo incompleta.
        finish_reason = getattr(r.choices[0], "finish_reason", None)
        if finish_reason == "length":
            resp += ("\n\n⚠️ **Respuesta truncada por límite de tokens.** Sube 'Max tokens' "
                     "en la barra lateral y vuelve a preguntar para la respuesta completa.")

        return resp
    except Exception as e:
        return f"Error OpenRouter: {str(e)[:200]}"# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.sidebar.markdown(
        '<span class="raft-badge">RAFT-Chat v2 · EISC Univalle</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Parámetros")
    temperature     = st.sidebar.slider("🌡️ Temperatura",     0.05, 1.0, 0.3,  0.05)
    max_tokens      = st.sidebar.slider("📝 Max tokens",       256, 2500, 1500,  128)
    top_k_retrieval = st.sidebar.slider("🔍 Candidatos RRF",    5,   40,  TOP_K_RETRIEVAL, 5)
    top_k_final     = st.sidebar.slider("🏆 Docs al prompt",    1,   10,  TOP_K_FINAL,     1)
    show_pipeline   = st.sidebar.checkbox("🔬 Mostrar pipeline", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Estado")

    # ── Carga de recursos ────────────────────────────────────────────────────
    raft_qa_index    = load_raft_qa_index(DATASET_PATH)
    embedder         = load_embedder()
    reranker         = load_reranker()
    qdrant_client    = load_qdrant()

    # ── Cabecera ──────────────────────────────────────────────────────────────
    st.title("🎓 RAFT-Chat v2 · EISC Universidad del Valle")
    st.caption(
        "Retrieval-Augmented Fine-Tuning · "
        "Qdrant (~127k chunks) + BM25 + BGE-M3 + BGE-Reranker-v2-m3 · "
        f"Llama 3.3-70B via OpenRouter"
    )

    # ── Historial ──────────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_pipeline and "pipeline_info" in msg:
                info = msg["pipeline_info"]
                with st.expander("🔬 Pipeline de recuperación", expanded=False):
                    st.markdown(f"**Tiempo total:** {info['elapsed']:.2f}s")
                    cols = st.columns([1, 1, 2])
                    cols[0].metric("Candidatos RRF", info["n_rrf"])
                    cols[1].metric("Docs al prompt", info["n_final"])
                    cols[2].metric("Reranker", "BGE-v2-m3" if info["reranked"] else "RRF solo")
                    st.markdown("**Documentos usados:**")
                    for d in info["docs"]:
                        preview = d["text"][:200].replace("\n", " ")
                        fuente  = os.path.basename(str(d.get("source", "")))
                        st.markdown(
                            f'<div class="doc-card">' +
                            f'<span class="score-bar">⭐ {d.get("final_score", d.get("rrf_score", 0)):.3f}</span> ' +
                            f'· <b>{fuente}</b>: {preview}…</div>',
                            unsafe_allow_html=True,
                        )

    # ── Input ─────────────────────────────────────────────────────────────────
    query = st.chat_input("Escribe tu pregunta sobre acreditación EISC…")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    t0 = time.time()

    # ── Paso 1: Búsqueda híbrida sobre Qdrant (FIX 3) ─────────────────────────
    with st.spinner("🔍 Búsqueda híbrida BM25 + BGE-M3 sobre Qdrant…"):
        candidates = hybrid_rrf_search_qdrant(
            query, embedder, qdrant_client, top_k=top_k_retrieval,
        )

    # ── Paso 2: Reranking ──────────────────────────────────────────────────────
    with st.spinner("🏆 Reranking con BGE-Reranker-v2-m3…"):
        final_docs   = rerank(query, candidates, reranker, top_k=top_k_final)
        reranked_flag = reranker is not None

    # ── Paso 3: Generación RAFT (FIX 4 + FIX 5) ───────────────────────────────
    with st.spinner("🤔 Generando respuesta RAFT con Llama 3.3-70B (OpenRouter)…"):
        if final_docs:
            response = generate_raft_response(
                query, final_docs, raft_qa_index, temperature, max_tokens
            )
        else:
            response = "No encontré documentos relevantes en la colección de acreditación."

    elapsed = time.time() - t0

    # ── Mostrar respuesta ──────────────────────────────────────────────────────
    pipeline_info = {
        "elapsed":  elapsed,
        "n_rrf":    len(candidates),
        "n_final":  len(final_docs),
        "reranked": reranked_flag,
        "docs":     final_docs,
    }

    with st.chat_message("assistant"):
        st.markdown(response)
        if show_pipeline:
            with st.expander("🔬 Pipeline de recuperación", expanded=True):
                cols = st.columns([1, 1, 2])
                cols[0].metric("Candidatos RRF", len(candidates))
                cols[1].metric("Docs al prompt", len(final_docs))
                cols[2].metric("Tiempo", f"{elapsed:.2f}s")
                for d in final_docs:
                    preview = d["text"][:200].replace("\n", " ")
                    fuente  = os.path.basename(str(d.get("source", "")))
                    st.markdown(
                        f'<div class="doc-card">' +
                        f'<span class="score-bar">⭐ {d.get("final_score", d.get("rrf_score", 0)):.3f}</span> ' +
                        f'· <b>{fuente}</b>: {preview}…</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role":          "assistant",
        "content":       response,
        "pipeline_info": pipeline_info if show_pipeline else {},
    })
    st.rerun()


if __name__ == "__main__":
    main()
