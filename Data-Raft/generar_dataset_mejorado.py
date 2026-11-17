#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generar_dataset_mejorado.py
═══════════════════════════════════════════════════════════════════════════════
Genera un dataset Q/A de alta calidad para acreditación CNA (EISC, Univalle)
a partir de documentos PDF, con tres mejoras sobre el dataset original:

  1. LIMPIEZA  — elimina duplicados exactos y respuestas triviales del
                 dataset original para reutilizar lo que ya es bueno.
  2. DIVERSIDAD — 15 plantillas de pregunta cubren hechos, procesos,
                  causas, comparaciones y preguntas multi-factor.
  3. RAFT       — cada pregunta se genera dando al LLM el chunk correcto
                 + 2 distractores, forzando respuestas específicas que
                 no pueden responderse con conocimiento general.

Flujo:
  PDFs ──► chunks ──► RAFT prompts ──► Groq LLM ──► pares Q/A
                                                         │
  dataset_original.json ──► limpieza ──────────────────►│
                                                         ▼
                                              dataset_mejorado.json

Uso:
  pip install groq pymupdf tiktoken tqdm
  python generar_dataset_mejorado.py --pdfs ./pdfs --original dataset_acreditacion.json

Argumentos:
  --pdfs        Carpeta con los PDFs (default: ./pdfs)
  --original    Dataset original a limpiar y fusionar (opcional)
  --output      Nombre del archivo de salida (default: dataset_mejorado.json)
  --model       Modelo de Groq (default: llama-3.3-70b-versatile)
  --max-chunks  Máx. chunks a procesar por PDF (default: 50, -1 = todos)
  --workers     Threads paralelos (default: 3)
  --api-key     API Key de Groq (o variable GROQ_API_KEY)
"""

import os
import re
import json
import time
import random
import hashlib
import argparse
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "TU_GROQ_API_KEY_AQUI")
MODEL        = "llama-3.3-70b-versatile"
CHUNK_SIZE   = 600      # caracteres por chunk
CHUNK_OVERLAP= 100
MIN_CHUNK    = 120      # descartar chunks muy cortos (encabezados, números)
MAX_RETRIES  = 3
RETRY_DELAY  = 2.0
MIN_RESP_LEN = 40       # respuesta mínima aceptable
DISTRACTORES = 2        # chunks distractores por pregunta RAFT

# ── Plantillas de pregunta diversas ──────────────────────────────────────────
# Se elige una aleatoriamente por chunk. Agrupadas por tipo para asegurar
# cobertura balanceada: ~25% factual, ~20% procedimental, ~20% causal,
# ~20% comparativa, ~15% multi-factor.
PLANTILLAS = {
    "factual": [
        "¿Qué establece el siguiente texto en relación con {tema}?",
        "¿Cuáles son los aspectos principales que describe el documento sobre {tema}?",
        "¿Qué información específica se menciona sobre {tema}?",
        "¿Cómo define o caracteriza el documento a {tema}?",
        "¿Cuáles son los elementos clave mencionados sobre {tema}?",
    ],
    "procedimental": [
        "¿Cuál es el proceso o procedimiento descrito para {tema}?",
        "¿Qué pasos o etapas se mencionan en relación con {tema}?",
        "¿Cómo se lleva a cabo o se implementa {tema} según el documento?",
        "¿Qué acciones o actividades se requieren para {tema}?",
    ],
    "causal": [
        "¿Por qué es importante {tema} en el contexto de la acreditación?",
        "¿Qué impacto o consecuencias tiene {tema} según el documento?",
        "¿Cuál es la justificación o razón de ser de {tema}?",
        "¿Qué beneficios o resultados se esperan de {tema}?",
    ],
    "comparativa": [
        "¿Qué diferencias o similitudes se mencionan en el texto respecto a los distintos aspectos de {tema}?",
        "¿Cómo se relaciona {tema} con otros elementos o factores del documento?",
        "¿En qué se distingue o complementa {tema} con los demás aspectos descritos?",
    ],
    "multifactor": [
        "¿Cómo contribuye {tema} al mejoramiento de la calidad y la autoevaluación del programa?",
        "¿De qué manera {tema} se articula con los factores de acreditación del CNA?",
        "¿Qué relación tiene {tema} con la formación integral y el perfil del egresado?",
    ],
}

TIPOS = list(PLANTILLAS.keys())
PESOS = [0.25, 0.20, 0.20, 0.20, 0.15]   # distribución deseada

STOPWORDS = {
    "de","la","el","en","y","a","los","del","se","las","por","un","con",
    "una","su","al","lo","como","más","pero","sus","le","ya","o","este",
    "sí","porque","esta","entre","cuando","muy","sin","sobre","también",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generar")
lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — EXTRACCIÓN Y CHUNKING DE PDFS
# ─────────────────────────────────────────────────────────────────────────────
def extraer_pdf(pdf_path: str) -> str:
    """Extrae texto de un PDF con pymupdf4llm (tablas incluidas)."""
    try:
        import fitz
        import pymupdf4llm
        doc  = fitz.open(pdf_path)
        text = pymupdf4llm.to_markdown(doc, show_progress=False)
        doc.close()
        return text or ""
    except Exception:
        try:
            import fitz
            doc  = fitz.open(pdf_path)
            text = "\n".join(p.get_text() for p in doc)
            doc.close()
            return text
        except Exception as e:
            log.warning(f"No se pudo leer {pdf_path}: {e}")
            return ""


def chunking(text: str) -> list[str]:
    """Chunking recursivo semántico respetando saltos de párrafo."""
    separadores = ["\n\n##", "\n\n###", "\n\n", "\n", " ", ""]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            for sep in separadores:
                pos = text.rfind(sep, start, end)
                if pos > start:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        # Filtros de calidad
        palabras = [w for w in chunk.split() if w.lower() not in STOPWORDS]
        if (len(chunk) >= MIN_CHUNK
                and len(palabras) >= 15
                and not chunk.startswith("|")   # tablas puras
                and sum(c.isalpha() for c in chunk) / max(len(chunk), 1) > 0.5):
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP if end - CHUNK_OVERLAP > start else end
    return chunks


def extraer_tema(chunk: str) -> str:
    """Extrae el tema dominante del chunk para rellenar la plantilla."""
    # Primera oración significativa
    primera = re.split(r'[.!?]', chunk.replace('\n', ' '))[0].strip()
    if len(primera) > 80:
        primera = primera[:80] + "..."
    return primera.lower() if primera else "este aspecto del documento"


def procesar_pdfs(pdf_dir: str, max_chunks: int = -1
                  ) -> dict[str, list[str]]:
    """Devuelve {nombre_pdf: [chunk, ...]}"""
    pdfs = sorted(Path(pdf_dir).rglob("*.pdf"))
    if not pdfs:
        log.error(f"No hay PDFs en '{pdf_dir}'")
        return {}

    log.info(f"📄 {len(pdfs)} PDFs en '{pdf_dir}'")
    resultado = {}
    for pdf in pdfs:
        texto  = extraer_pdf(str(pdf))
        chunks = chunking(texto)
        if max_chunks > 0:
            chunks = chunks[:max_chunks]
        resultado[pdf.name] = chunks
        log.info(f"   {pdf.name}: {len(chunks)} chunks")

    total = sum(len(v) for v in resultado.values())
    log.info(f"   Total: {total:,} chunks útiles")
    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — LIMPIEZA DEL DATASET ORIGINAL
# ─────────────────────────────────────────────────────────────────────────────
def limpiar_dataset(ruta: str) -> list[dict]:
    """Elimina duplicados exactos y respuestas triviales del original."""
    log.info(f"🧹 Limpiando dataset original: {ruta}")
    with open(ruta, encoding="utf-8") as f:
        data = json.load(f)

    visto_p, visto_r, limpio = set(), set(), []
    stats = defaultdict(int)

    for item in data:
        msgs = item.get("messages", [])
        p = next((m["content"] for m in msgs if m["role"] == "user"),      "").strip()
        r = next((m["content"] for m in msgs if m["role"] == "assistant"), "").strip()

        # Filtros
        if len(r) < MIN_RESP_LEN:
            stats["corta"] += 1
            continue
        ph = hashlib.md5(p.lower().encode()).hexdigest()
        rh = hashlib.md5(r.lower().encode()).hexdigest()
        if ph in visto_p:
            stats["dup_p"] += 1
            continue
        if rh in visto_r:
            stats["dup_r"] += 1
            continue
        # Filtrar preguntas con plantillas sobre-representadas
        prefijo = " ".join(p.lower().split()[:6])
        if re.match(r"¿cuál es el (propósito|objetivo|nombre|título) (principal|del|de la)", prefijo):
            # Mantener solo 1 de cada 5 para balancear
            if random.random() > 0.20:
                stats["sobre_repr"] += 1
                continue

        visto_p.add(ph)
        visto_r.add(rh)
        limpio.append(item)

    log.info(f"   Original: {len(data):,}  →  Limpio: {len(limpio):,}")
    log.info(f"   Eliminados: cortas={stats['corta']:,}, "
             f"dup_p={stats['dup_p']:,}, "
             f"dup_r={stats['dup_r']:,}, "
             f"sobre_repr={stats['sobre_repr']:,}")
    return limpio


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — GENERACIÓN RAFT CON GROQ
# ─────────────────────────────────────────────────────────────────────────────
def elegir_plantilla() -> str:
    """Elige un tipo de plantilla con los pesos definidos y devuelve una aleatoria."""
    tipo = random.choices(TIPOS, weights=PESOS, k=1)[0]
    return random.choice(PLANTILLAS[tipo])


def construir_prompt_raft(chunk: str, distractores: list[str],
                          fuente: str) -> str:
    """
    Prompt RAFT: el modelo recibe el chunk correcto + distractores,
    debe generar UNA pregunta que solo el chunk correcto responde,
    y la respuesta completa basada exclusivamente en él.
    """
    tema   = extraer_tema(chunk)
    plantilla = elegir_plantilla()
    pregunta_sugerida = plantilla.format(tema=tema)

    dist_text = ""
    for i, d in enumerate(distractores, 1):
        dist_text += f"\n[FRAGMENTO DISTRACTOR {i}]\n{d}\n"

    return f"""Eres un experto en acreditación universitaria colombiana (CNA, Decreto 1330/2019).

Tu tarea es generar UN par pregunta-respuesta de alta calidad a partir del FRAGMENTO PRINCIPAL.
La pregunta debe ser respondible ÚNICAMENTE con el FRAGMENTO PRINCIPAL, no con los distractores.

[FRAGMENTO PRINCIPAL - FUENTE: {fuente}]
{chunk}
{dist_text}

INSTRUCCIONES ESTRICTAS:
1. Genera UNA sola pregunta específica y concreta. Puedes inspirarte en: "{pregunta_sugerida}"
   Pero adapta la pregunta para que sea precisa sobre el contenido real del fragmento.
2. La pregunta NO puede responderse con los fragmentos distractores.
3. La respuesta debe tener mínimo 2 oraciones completas y máximo 5.
4. La respuesta debe basarse EXCLUSIVAMENTE en el fragmento principal.
5. Usa terminología técnica del CNA cuando sea relevante.
6. NO inventes información que no esté en el fragmento.

FORMATO DE SALIDA (exactamente así, sin nada más):
PREGUNTA: [tu pregunta aquí]
RESPUESTA: [tu respuesta aquí]"""


def llamar_groq(prompt: str, api_key: str, model: str) -> str:
    """Llamada a la API de Groq con reintentos."""
    from groq import Groq
    client = Groq(api_key=api_key)

    for intento in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7,
                top_p=0.95,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if intento < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** intento) + random.uniform(0, 1)
                log.warning(f"   Error Groq (intento {intento+1}): {e}. Esperando {wait:.1f}s")
                time.sleep(wait)
            else:
                log.error(f"   Error definitivo Groq: {e}")
                return ""
    return ""


def parsear_respuesta(texto: str) -> tuple[str, str]:
    """Extrae PREGUNTA y RESPUESTA del texto generado por el modelo."""
    p_match = re.search(r"PREGUNTA:\s*(.+?)(?=RESPUESTA:|$)", texto, re.DOTALL | re.IGNORECASE)
    r_match = re.search(r"RESPUESTA:\s*(.+?)$", texto, re.DOTALL | re.IGNORECASE)

    pregunta  = p_match.group(1).strip() if p_match else ""
    respuesta = r_match.group(1).strip() if r_match else ""

    # Limpiar
    pregunta  = re.sub(r'\s+', ' ', pregunta).strip()
    respuesta = re.sub(r'\s+', ' ', respuesta).strip()

    return pregunta, respuesta


def es_par_valido(pregunta: str, respuesta: str) -> bool:
    """Filtros de calidad mínima para el par generado."""
    if len(pregunta)  < 20:  return False
    if len(respuesta) < MIN_RESP_LEN: return False
    if not pregunta.endswith("?"):    return False
    # Detectar alucinaciones genéricas
    frases_malas = [
        "como inteligencia artificial", "como modelo de lenguaje",
        "no tengo acceso", "no puedo determinar",
        "el fragmento no menciona", "no se proporciona",
        "según el contexto", "basándome en el texto dado",
    ]
    resp_lower = respuesta.lower()
    if any(f in resp_lower for f in frases_malas): return False
    return True


def generar_par(args_tuple):
    """Genera un par Q/A para un chunk. Ejecutado en threads."""
    chunk, todos_chunks, fuente, api_key, model = args_tuple

    # Elegir distractores: chunks de OTRAS fuentes si es posible
    otros = [c for c, f in todos_chunks if f != fuente and c != chunk]
    if len(otros) < DISTRACTORES:
        otros = [c for c, f in todos_chunks if c != chunk]
    distractores = random.sample(otros, min(DISTRACTORES, len(otros)))

    prompt = construir_prompt_raft(chunk, distractores, fuente)
    texto  = llamar_groq(prompt, api_key, model)

    if not texto:
        return None

    pregunta, respuesta = parsear_respuesta(texto)
    if not es_par_valido(pregunta, respuesta):
        return None

    par_id = hashlib.md5(f"{fuente}{chunk[:50]}".encode()).hexdigest()[:16]
    return {
        "id":      f"{Path(fuente).stem}_{par_id}",
        "source":  fuente,
        "messages": [
            {"role": "user",      "content": pregunta},
            {"role": "assistant", "content": respuesta},
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — DEDUPLICACIÓN SEMÁNTICA (fuzzy)
# ─────────────────────────────────────────────────────────────────────────────
def dedup_fuzzy(pares: list[dict], umbral: float = 0.85) -> list[dict]:
    """
    Elimina pares cuyas preguntas sean muy similares por similitud de Jaccard
    sobre bigramas. Más rápido que embeddings para datasets de <200k pares.
    """
    def bigramas(texto: str) -> set[str]:
        tokens = texto.lower().split()
        return {f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)}

    log.info(f"🔍 Deduplicación fuzzy sobre {len(pares):,} pares...")
    vistos: list[set] = []
    limpio = []

    for par in pares:
        msgs = par.get("messages", [])
        p = next((m["content"] for m in msgs if m["role"] == "user"), "")
        bg = bigramas(p)
        if not bg:
            continue
        # Comparar con los últimos 2000 vistos (ventana deslizante)
        duplicado = False
        for bg_prev in vistos[-2000:]:
            intersec = len(bg & bg_prev)
            union    = len(bg | bg_prev)
            if union > 0 and intersec / union > umbral:
                duplicado = True
                break
        if not duplicado:
            vistos.append(bg)
            limpio.append(par)

    eliminados = len(pares) - len(limpio)
    log.info(f"   Eliminados por fuzzy: {eliminados:,} → {len(limpio):,} únicos")
    return limpio


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — ESTADÍSTICAS DEL DATASET GENERADO
# ─────────────────────────────────────────────────────────────────────────────
def estadisticas(pares: list[dict]) -> None:
    import statistics as st
    pregs = []
    resps = []
    for p in pares:
        msgs = p.get("messages", [])
        q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        r = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        pregs.append(len(q))
        resps.append(len(r))

    fuentes = len({p["source"] for p in pares})
    print("\n" + "═" * 55)
    print("  ESTADÍSTICAS DEL DATASET GENERADO")
    print("═" * 55)
    print(f"  Total pares         : {len(pares):,}")
    print(f"  Fuentes únicas      : {fuentes}")
    print(f"  Long. media pregunta: {st.mean(pregs):.0f} chars")
    print(f"  Long. media respuesta: {st.mean(resps):.0f} chars")
    print(f"  Long. media respuesta P95: {sorted(resps)[int(len(resps)*0.95)]:.0f} chars")
    print("═" * 55)

    # Distribución por tipo de pregunta
    tipos_cnt = defaultdict(int)
    for p in pares:
        q = next((m["content"] for m in p.get("messages",[]) if m["role"]=="user"), "").lower()
        if q.startswith("¿cómo"):          tipos_cnt["¿Cómo?"] += 1
        elif q.startswith("¿cuáles"):      tipos_cnt["¿Cuáles?"] += 1
        elif q.startswith("¿qué"):         tipos_cnt["¿Qué?"] += 1
        elif q.startswith("¿por qué"):     tipos_cnt["¿Por qué?"] += 1
        elif q.startswith("¿cuánto"):      tipos_cnt["¿Cuánto?"] += 1
        elif q.startswith("¿de qué"):      tipos_cnt["¿De qué manera?"] += 1
        else:                              tipos_cnt["Otro"] += 1

    print("\n  Distribución por tipo de pregunta:")
    for tipo, cnt in sorted(tipos_cnt.items(), key=lambda x: -x[1]):
        pct = cnt / len(pares) * 100
        bar = "█" * int(pct / 2)
        print(f"    {tipo:20s}: {cnt:6,}  ({pct:5.1f}%)  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdfs",       default="./corpus")
    p.add_argument("--original",   default="")
    p.add_argument("--output",     default="dataset_mejorado.json")
    p.add_argument("--model",      default=MODEL)
    p.add_argument("--max-chunks", type=int, default=50)
    p.add_argument("--workers",    type=int, default=3)
    p.add_argument("--api-key",    default=GROQ_API_KEY)
    p.add_argument("--solo-limpia",action="store_true",
                   help="Solo limpiar el dataset original sin generar nuevos pares")
    return p.parse_args()


def main():
    args = parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  Generador de Dataset Mejorado — ChatAcredita PRO        ║
║  EISC · Universidad del Valle · CNA Colombia             ║
╚══════════════════════════════════════════════════════════╝""")

    pares_base = []

    # ── 1. Limpiar dataset original ──────────────────────────────────────────
    if args.original and Path(args.original).exists():
        pares_base = limpiar_dataset(args.original)
    else:
        log.info("No se especificó dataset original — se generará desde cero")

    if args.solo_limpia:
        salida = Path(args.output)
        with open(salida, "w", encoding="utf-8") as f:
            json.dump(pares_base, f, ensure_ascii=False, indent=2)
        log.info(f"✅ Dataset limpio guardado: {salida} ({len(pares_base):,} pares)")
        estadisticas(pares_base)
        return

    # ── 2. Extraer chunks de PDFs ─────────────────────────────────────────────
    chunks_por_pdf = procesar_pdfs(args.pdfs, args.max_chunks)
    if not chunks_por_pdf:
        log.error("No hay PDFs que procesar. Revisa la carpeta --pdfs")
        return

    # Lista plana (chunk, fuente) para elegir distractores
    todos_chunks = [
        (chunk, fuente)
        for fuente, chunks in chunks_por_pdf.items()
        for chunk in chunks
    ]
    total_chunks = len(todos_chunks)
    log.info(f"\n🚀 Generando pares RAFT para {total_chunks:,} chunks "
             f"({args.workers} workers, modelo {args.model})")

    # ── 3. Generación paralela ────────────────────────────────────────────────
    tareas = [
        (chunk, todos_chunks, fuente, args.api_key, args.model)
        for fuente, chunks in chunks_por_pdf.items()
        for chunk in chunks
    ]
    random.shuffle(tareas)   # mezclar para mejor distribución de distractores

    nuevos_pares = []
    errores      = 0
    try:
        from tqdm import tqdm
        barra = tqdm(total=len(tareas), desc="Generando", unit="par")
    except ImportError:
        barra = None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futuros = {executor.submit(generar_par, t): t for t in tareas}
        for futuro in as_completed(futuros):
            resultado = futuro.result()
            if resultado:
                with lock:
                    nuevos_pares.append(resultado)
            else:
                errores += 1
            if barra:
                barra.update(1)
                barra.set_postfix(ok=len(nuevos_pares), err=errores)

    if barra:
        barra.close()

    log.info(f"\n   Pares generados: {len(nuevos_pares):,}  |  Errores: {errores:,}")

    # ── 4. Fusionar con dataset base y deduplicar ─────────────────────────────
    todos_pares = pares_base + nuevos_pares
    log.info(f"\n🔀 Fusionando: {len(pares_base):,} (base) + "
             f"{len(nuevos_pares):,} (nuevos) = {len(todos_pares):,}")

    # Dedup exacto
    vistos_h = set()
    sin_dup = []
    for par in todos_pares:
        msgs = par.get("messages", [])
        p = next((m["content"] for m in msgs if m["role"]=="user"), "")
        h = hashlib.md5(p.lower().encode()).hexdigest()
        if h not in vistos_h:
            vistos_h.add(h)
            sin_dup.append(par)
    log.info(f"   Tras dedup exacto: {len(sin_dup):,}")

    # Dedup fuzzy
    final = dedup_fuzzy(sin_dup, umbral=0.85)

    # Mezclar aleatoriamente
    random.seed(42)
    random.shuffle(final)

    # ── 5. Guardar ────────────────────────────────────────────────────────────
    salida = Path(args.output)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    log.info(f"\n✅ Dataset mejorado guardado: {salida}")
    estadisticas(final)

    # ── 6. Guardar log de errores si los hubo ─────────────────────────────────
    if errores > 0:
        log_err = salida.with_suffix(".errores.txt")
        with open(log_err, "w") as f:
            f.write(f"Chunks que fallaron: {errores}/{len(tareas)}\n")
        log.info(f"   Log de errores: {log_err}")


if __name__ == "__main__":
    main()
