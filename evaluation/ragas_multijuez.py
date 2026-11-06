"""
Punto5_MultiJuez_RAGAS_v2.py
==============================
Evaluación RAGAS de ChatAcredita PRO directamente desde HuggingFace Inference API,
con 5 jueces independientes de OpenRouter y promedio de ensemble.

Generador  : raulgdp/ChatAcredita-PRO-Qwen3-14B  (HuggingFace Inference API)
Jueces (5) : OpenRouter → qwen3-32b, gpt-4o-mini, gemini-2.0-flash,
                            llama-3.3-70b-instruct, claude-3-5-haiku

Requisitos:
  pip install openai qdrant-client sentence-transformers huggingface_hub --break-system-packages
  export HF_TOKEN="hf_..."
  export OPENROUTER_API_KEY="sk-or-..."

Salidas:
  multijuez_results_v2.csv     — scores por pregunta × juez × métrica
  multijuez_summary_v2.json    — resumen global con ensemble y desglose por juez
"""

import os, json, time, csv, statistics, re
from pathlib import Path
from typing import Optional

# ─── Dependencias ──────────────────────────────────────────────────────────
try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from openai import OpenAI
except ImportError as e:
    print(f"[ERROR] Falta dependencia: {e}")
    print("Instala con: pip install openai qdrant-client sentence-transformers huggingface_hub --break-system-packages")
    raise

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

HF_TOKEN          = os.environ.get("HF_TOKEN", "")
OPENROUTER_KEY    = os.environ.get("OPENROUTER_API_KEY", "")

# Qdrant Cloud (us-east-1)
QDRANT_URL        = os.environ.get("QDRANT_URL", "https://ac76af0c-59ba-4a65-87e4-ff360d8317ef.us-east-1-1.aws.cloud.qdrant.io")   # Ej: https://xxxx.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
COLLECTION        = "acreditacion"
TOP_K_RETRIEVE    = 10
TOP_K_RERANK      = 8

# ─── ChatAcredita PRO en HuggingFace ─────────────────────────────────────
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "raulgdp/qwen3-14b-acredita")

# Modos de generador (en orden de preferencia):
#   "hf"          → HuggingFace Inference API serverless (requiere HF_TOKEN)
#   "hf_dedicated"→ HuggingFace Dedicated Endpoint (requiere HF_ENDPOINT_URL)
#   "openrouter"  → OpenRouter con modelo base Qwen3-14B (fallback sin fine-tune)
#   "ollama"      → Ollama local
# "openrouter" usa qwen3-14b via OpenRouter (mismo modelo base que ChatAcredita PRO)
# "hf"         usa HuggingFace Serverless Inference API
GENERATOR_MODE = os.environ.get("GENERATOR_MODE", "openrouter")

HF_API_BASE        = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}/v1"
HF_ENDPOINT_URL    = os.environ.get("HF_ENDPOINT_URL", "")   # URL de dedicated endpoint si tienes uno
# Modelo OpenRouter a usar como fallback (base sin fine-tune)
# Modelo generador en OpenRouter — mismo base que ChatAcredita PRO (Qwen3-14B)
OPENROUTER_GEN_MODEL = os.environ.get("OPENROUTER_GEN_MODEL", "mistralai/mistral-small-3.1-24b-instruct")

# ─── 5 Jueces RAGAS via OpenRouter ────────────────────────────────────────
JUDGES = [
    {"id": "gemma3-27b",   "model": "google/gemma-3-27b-it",               "label": "Gemma 3 27B-IT"},
    {"id": "llama-70b",    "model": "meta-llama/llama-3.3-70b-instruct",   "label": "Llama-3.3-70B"},
    {"id": "gpt4o-mini",   "model": "openai/gpt-4o-mini",                  "label": "GPT-4o Mini"},
    {"id": "gpt4o",        "model": "openai/gpt-4o",                       "label": "GPT-4o"},
    {"id": "claude-sonnet","model": "anthropic/claude-3-5-sonnet",         "label": "Claude 3.5 Sonnet"},
]
# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK (mismas 142 queries que Punto4)
# ═══════════════════════════════════════════════════════════════════════════

QUERIES = [
    # ── Con ground_truth — EISC específicas (3 preguntas clave) ────────
    {
        "q": "¿Cuáles son los grupos de investigación de la EISC?",
        "gt": (
            "Los grupos de investigación activos de la Escuela de Ingeniería de Sistemas y Computación (EISC) "
            "de la Universidad del Valle en 2024 son:\n"
            "1. GUIA - Grupo de Univalle en Inteligencia Artificial. Director: Oscar Fernando Bedoya Ruiz.\n"
            "2. GOTIC - Grupo de Investigación en Tecnologías de la Información y Comunicaciones. Director: John Sanabria.\n"
            "3. GENIUS-UV - Grupo de Investigación en Ingeniería con Perspectiva de Género. Directora: Laura Rodríguez.\n"
            "4. AVISPA - Ambientes Visuales de Programación Aplicativa. Director: Jesús Alexander Aranda.\n"
            "5. Bioinformática. Director: Pedro Moreno.\n"
            "6. CAMALEON - Grupo de Investigación en Interacción Humano Computador (HCI). Directora: Paola Rodríguez.\n"
            "7. GEDI - Grupo de Estudios Doctorales en Informática. Director: Mauricio Gaona.\n"
            "La EISC cuenta con 6 grupos activos reconocidos por Minciencias. "
            "Los grupos AVISPA, GEDI, BIOINFORMATICA, GENIUS-UV, CAMALEON y Multimedia son históricos y ya no están activos en 2024."
        ),
    },
    {
        "q": "¿Cuáles son las características del Factor 3 de acreditación de la EISC?",
        "gt": (
            "El Factor 3 (Profesores), confirmado en la Renovación de Acreditación en Alta Calidad de la EISC 2024, "
            "tiene 8 características:\n"
            "- Característica 8. Selección, vinculación y permanencia (15%)\n"
            "- Característica 9. Estatuto Profesoral (12%)\n"
            "- Característica 10. Número, Dedicación, Nivel de Formación y Experiencia (20%)\n"
            "- Característica 11. Desarrollo profesoral (15%)\n"
            "- Característica 12. Estímulos a la trayectoria profesoral (12%)\n"
            "- Característica 13. Producción, Pertinencia, Utilización e Impacto de material Docente (7%)\n"
            "- Característica 14. Remuneración por Méritos (8%)\n"
            "- Característica 15. Evaluación de Profesores (11%)"
        ),
    },
    {
        "q": "¿Cuál es la lista de todos los profesores nombrados de la EISC?",
        "gt": (
            "Los profesores nombrados de la Escuela de Ingeniería de Sistemas y Computación (EISC) de la "
            "Universidad del Valle, según el documento de Renovación de Acreditación en Alta Calidad (noviembre 2024), son:\n"
            "1. Jesús Alexander Aranda Bueno - Ph.D. - Director Grupo TEDESOFT\n"
            "2. John Alexander Sanabria Ordóñez - Ph.D.\n"
            "3. Juan Francisco Díaz Frías - Ph.D. - Director del Programa Académico de Ingeniería de Sistemas\n"
            "4. María Patricia Trujillo Uribe - Ph.D.\n"
            "5. Beatriz Eugenia Florián Gaviria - Ph.D.\n"
            "6. Oscar Fernando Bedoya Ruiz - Ph.D. - Director Grupo GUIA\n"
            "7. Paola Johanna Rodríguez Carrillo\n"
            "8. Pedro Antonio Moreno Tovar - Ph.D.\n"
            "9. Raúl Ernesto Gutiérrez de Piñerez Reyes - Ph.D.\n"
            "10. Víctor Bucheli Guerrero - Ph.D.\n"
            "11. Laura Sofía Rodríguez Pulecio - M.Sc. - Directora Grupo GOTIC\n"
            "12. Robinson Andrey Duque Agudelo - Ph.D.\n"
            "13. Andrés Mauricio Castillo Robles - Ph.D.\n"
            "14. Oswaldo Solarte - Ph.D. - Director de la EISC\n"
            "15. Carlos Mauricio Gaona Cuevas - Ph.D.\n"
            "16. Deisy Chaves Sánchez - Ph.D.\n"
            "17. Liliana Esther Machuca - Ph.D.\n"
            "18. Manuel Alejandro Pastrana Pardo - Ph.D."
        ),
    },
    # ── Con ground_truth — CNA general (10) ─────────────────────────────
    {"q": "¿Cuántos factores contempla el modelo de acreditación institucional del CNA según el Decreto 1330 de 2019?",
     "gt": "El modelo de acreditación institucional del CNA contempla 6 factores según el Decreto 1330 de 2019."},
    {"q": "¿Cuál es el período de vigencia máximo que puede otorgar el CNA para una acreditación de alta calidad?",
     "gt": "El período de vigencia máximo que puede otorgar el CNA para una acreditación de alta calidad es de 10 años."},
    {"q": "¿Qué porcentaje mínimo de profesores con título de maestría o doctorado se requiere para acreditación?",
     "gt": "No existe un porcentaje mínimo único definido; el CNA evalúa la pertinencia y suficiencia del cuerpo profesoral según el nivel del programa."},
    {"q": "¿Cuántas características tiene el Factor 1 (Misión y Proyecto Institucional) en el modelo del CNA?",
     "gt": "El Factor 1 (Misión y Proyecto Institucional) tiene 4 características en el modelo del CNA."},
    {"q": "¿Qué organismo expide el certificado de acreditación de alta calidad en Colombia?",
     "gt": "El Ministerio de Educación Nacional expide el certificado de acreditación de alta calidad en Colombia, previo concepto del CNA."},
    {"q": "¿Cuántos años de funcionamiento debe tener un programa para solicitar acreditación por primera vez?",
     "gt": "Un programa debe tener al menos 2 cohortes de egresados para solicitar acreditación por primera vez."},
    {"q": "¿Qué es el registro calificado y cómo se diferencia de la acreditación de alta calidad?",
     "gt": "El registro calificado es el reconocimiento mínimo para funcionamiento; la acreditación de alta calidad es voluntaria y reconoce excelencia superior."},
    {"q": "¿Cuál es el papel de los pares académicos en el proceso de acreditación del CNA?",
     "gt": "Los pares académicos son expertos designados por el CNA para realizar visitas de verificación y emitir conceptos sobre el cumplimiento de los criterios de calidad."},
    {"q": "¿Qué documento orienta el proceso de autoevaluación institucional según el CNA?",
     "gt": "El CNA publica lineamientos y guías de autoevaluación; el documento principal es los Lineamientos para la Acreditación Institucional."},
    {"q": "¿Cuáles son las condiciones de calidad del Decreto 1330 de 2019 para registro calificado?",
     "gt": "El Decreto 1330 de 2019 establece 15 condiciones de calidad distribuidas en condiciones de programa y de institución para el registro calificado."},
    # ── Sin ground_truth (132) ──────────────────────────────────────────
    {"q": "¿Qué es la autoevaluación en el contexto de la acreditación universitaria colombiana?"},
    {"q": "¿Cómo se estructura el factor de investigación en el modelo de acreditación del CNA?"},
    {"q": "¿Qué indicadores usa el CNA para evaluar la internacionalización de un programa académico?"},
    {"q": "¿Cuáles son los criterios de calidad para el factor de bienestar institucional?"},
    {"q": "¿Qué diferencia hay entre acreditación de programas y acreditación institucional?"},
    {"q": "¿Cómo se califica el impacto de los egresados en el medio en el proceso de acreditación?"},
    {"q": "¿Qué rol juega el Consejo Nacional de Educación Superior (CESU) en la acreditación?"},
    {"q": "¿Cuáles son los requisitos para que una institución solicite acreditación institucional?"},
    {"q": "¿Qué es el plan de mejoramiento y cuándo se presenta al CNA?"},
    {"q": "¿Cómo evalúa el CNA la pertinencia de un programa académico?"},
    {"q": "¿Qué son las condiciones mínimas de calidad del Decreto 1330?"},
    {"q": "¿Cuál es el proceso para renovar una acreditación de alta calidad?"},
    {"q": "¿Qué aspectos evalúa el factor de docentes en la acreditación CNA?"},
    {"q": "¿Cómo se mide la calidad del currículo en el proceso de acreditación?"},
    {"q": "¿Qué es el proyecto educativo institucional (PEI) y su relación con la acreditación?"},
    {"q": "¿Cuáles son los criterios de evaluación para los procesos académicos?"},
    {"q": "¿Qué importancia tiene la infraestructura física en la acreditación universitaria?"},
    {"q": "¿Cómo evalúa el CNA los recursos bibliográficos y bases de datos?"},
    {"q": "¿Qué es la visita de pares y en qué etapa del proceso ocurre?"},
    {"q": "¿Cuáles son las etapas del proceso de acreditación de un programa en Colombia?"},
    {"q": "¿Cómo se documenta la coherencia entre misión institucional y oferta académica?"},
    {"q": "¿Qué indicadores miden la calidad de la formación investigativa en pregrado?"},
    {"q": "¿Cuál es el proceso de selección y formación de pares académicos del CNA?"},
    {"q": "¿Cómo influye la extensión y proyección social en la calificación de acreditación?"},
    {"q": "¿Qué métricas de empleabilidad considera el CNA para evaluar programas?"},
    {"q": "¿Cuál es la diferencia entre condiciones de programa y condiciones institucionales en el Decreto 1330?"},
    {"q": "¿Qué es el Sistema Nacional de Acreditación (SNA) en Colombia?"},
    {"q": "¿Cómo se integran los resultados de aprendizaje en la evaluación curricular del CNA?"},
    {"q": "¿Qué documentos debe preparar una institución antes de la visita de pares?"},
    {"q": "¿Cómo evalúa el CNA el factor de gestión institucional?"},
    {"q": "¿Qué papel tienen los estudiantes en el proceso de autoevaluación?"},
    {"q": "¿Cómo se evalúa la coherencia entre el perfil de egreso y el mercado laboral?"},
    {"q": "¿Qué es el informe de autoevaluación y qué debe contener?"},
    {"q": "¿Cuáles son las implicaciones de perder la acreditación de alta calidad?"},
    {"q": "¿Cómo mide el CNA la producción académica de los docentes?"},
    {"q": "¿Qué criterios se usan para evaluar la eficiencia terminal de un programa?"},
    {"q": "¿Cómo se evalúa el factor de recursos físicos y financieros en la acreditación?"},
    {"q": "¿Qué es el proceso de hetero-evaluación en el contexto del CNA?"},
    {"q": "¿Cuáles son las características del factor de visibilidad nacional e internacional?"},
    {"q": "¿Cómo afecta el número de grupos de investigación reconocidos por Minciencias a la acreditación?"},
    {"q": "¿Qué indicadores de bienestar estudiantil evalúa el CNA?"},
    {"q": "¿Cómo se documentan los convenios de cooperación internacional para la acreditación?"},
    {"q": "¿Qué papel tiene el ICFES en el contexto de la acreditación universitaria?"},
    {"q": "¿Cómo evalúa el CNA la formación integral de los estudiantes?"},
    {"q": "¿Qué es el factor de egresados e impacto en el medio?"},
    {"q": "¿Cuál es la relación entre la acreditación y el Sistema de Aseguramiento de la Calidad (SAC)?"},
    {"q": "¿Cómo se evalúa la innovación pedagógica en los programas acreditados?"},
    {"q": "¿Qué documentos son evidencias clave para el factor de investigación?"},
    {"q": "¿Cómo influye la tasa de deserción estudiantil en la acreditación?"},
    {"q": "¿Qué es el plan de desarrollo institucional y su relación con la acreditación?"},
    {"q": "¿Cuáles son los criterios de evaluación para programas de posgrado?"},
    {"q": "¿Cómo evalúa el CNA los programas de formación docente?"},
    {"q": "¿Qué indicadores de impacto social considera el CNA?"},
    {"q": "¿Cómo se mide la satisfacción de egresados en el proceso de acreditación?"},
    {"q": "¿Qué es la evaluación externa y cómo complementa la autoevaluación?"},
    {"q": "¿Cuáles son los factores críticos para lograr la acreditación institucional?"},
    {"q": "¿Cómo se evalúa la calidad de los programas virtuales o a distancia?"},
    {"q": "¿Qué rol tiene la Sala de Consejo del CNA en el proceso de acreditación?"},
    {"q": "¿Cómo afecta el acuerdo 02 de 2020 del CESU al proceso de acreditación?"},
    {"q": "¿Qué es el Sistema Integrado de Información del CNA (SACES)?"},
    {"q": "¿Cómo se evalúan los programas académicos en modalidad dual o alternancia?"},
    {"q": "¿Cuáles son los lineamientos del CNA para la acreditación de especializaciones?"},
    {"q": "¿Qué importancia tiene la movilidad estudiantil para la acreditación internacional?"},
    {"q": "¿Cómo documenta una institución sus políticas de investigación para la acreditación?"},
    {"q": "¿Qué es la acreditación internacional y cómo se articula con el CNA?"},
    {"q": "¿Cuáles son las responsabilidades del rector en el proceso de acreditación?"},
    {"q": "¿Cómo evalúa el CNA el impacto de los programas en el desarrollo regional?"},
    {"q": "¿Qué indica la Resolución 21795 de 2020 sobre acreditación?"},
    {"q": "¿Cómo se evalúa la experiencia y cualificación del cuerpo docente?"},
    {"q": "¿Qué métricas de citación o indexación de publicaciones considera el CNA?"},
    {"q": "¿Cuáles son los criterios para evaluar los laboratorios y equipos especializados?"},
    {"q": "¿Cómo se acredita la suficiencia de recursos tecnológicos para la enseñanza?"},
    {"q": "¿Qué papel juega el reglamento estudiantil en la acreditación CNA?"},
    {"q": "¿Cómo se evalúa la coherencia del currículo con los estándares internacionales?"},
    {"q": "¿Qué es el índice de calidad global del CNA y cómo se calcula?"},
    {"q": "¿Cómo documentar los logros de los egresados para el proceso de acreditación?"},
    {"q": "¿Qué criterios de ética e integridad académica evalúa el CNA?"},
    {"q": "¿Cómo se evalúan los programas de extensión y consultoría en la acreditación?"},
    {"q": "¿Qué es la condicionalidad en la acreditación y cuándo aplica?"},
    {"q": "¿Cómo influyen los resultados Saber Pro en la evaluación de un programa?"},
    {"q": "¿Qué son los lineamientos complementarios del CNA y cuándo se aplican?"},
    {"q": "¿Cómo se evalúa la sostenibilidad financiera de una institución en acreditación?"},
    {"q": "¿Qué indicadores de inclusión y diversidad considera el CNA?"},
    {"q": "¿Cómo se documenta el impacto de la investigación en la comunidad?"},
    {"q": "¿Qué rol tienen los egresados como informantes en la visita de pares?"},
    {"q": "¿Cómo evalúa el CNA la pertinencia de los programas en zonas de difícil acceso?"},
    {"q": "¿Qué diferencia existe entre el modelo de acreditación de 2006 y el del Decreto 1330?"},
    {"q": "¿Cómo se acredita la existencia de políticas de formación docente continua?"},
    {"q": "¿Qué documentos respaldan el cumplimiento del factor de bienestar en la acreditación?"},
    {"q": "¿Cómo evalúa el CNA los programas con doble titulación?"},
    {"q": "¿Qué es el concepto de acreditación del CNA y qué información contiene?"},
    {"q": "¿Cuáles son las causales para la no renovación de una acreditación?"},
    {"q": "¿Cómo se evalúa el impacto de los semilleros de investigación en la acreditación?"},
    {"q": "¿Qué criterios aplica el CNA para programas de formación para el trabajo?"},
    {"q": "¿Cómo se mide el impacto de la internacionalización en la calidad académica?"},
    {"q": "¿Cuál es el rol de la vicerrectoría académica en el proceso de acreditación?"},
    {"q": "¿Qué son las visitas de verificación de condiciones de calidad del MEN?"},
    {"q": "¿Cómo documenta un programa su proceso de revisión y actualización curricular?"},
    {"q": "¿Qué papel tienen las asociaciones profesionales en la acreditación de programas?"},
    {"q": "¿Cómo se evalúa el acceso y uso de plataformas digitales en la formación?"},
    {"q": "¿Cuáles son los indicadores de gestión del conocimiento para la acreditación?"},
    {"q": "¿Qué es el modelo de evaluación de factores, características y aspectos del CNA?"},
    {"q": "¿Cómo se evidencia la participación docente en redes académicas internacionales?"},
    {"q": "¿Qué información debe incluir el informe final de autoevaluación al CNA?"},
    {"q": "¿Cómo se diferencia la acreditación de programas presenciales y virtuales?"},
    {"q": "¿Qué criterios aplica el CNA para la acreditación de programas nocturnos?"},
    {"q": "¿Cómo evalúa el CNA el factor de comunicación e información institucional?"},
    {"q": "¿Cuál es la importancia del seguimiento a graduados para la acreditación?"},
    {"q": "¿Qué significa calificación de plenamente y en alto grado en la escala del CNA?"},
    {"q": "¿Cómo se evalúa la articulación entre pregrado y posgrado en la acreditación?"},
    {"q": "¿Qué son los acuerdos de la Sala del CNA y qué vinculación tienen con la acreditación?"},
    {"q": "¿Cómo se acredita que una institución tiene políticas claras de propiedad intelectual?"},
    {"q": "¿Qué papel juega la acreditación en el acceso a convocatorias de Minciencias?"},
    {"q": "¿Cómo evalúa el CNA los resultados de las pruebas Saber TyT para tecnologías?"},
    {"q": "¿Cuáles son los criterios del CNA para evaluar la gestión curricular?"},
    {"q": "¿Cómo se documenta el impacto social y productivo de la extensión universitaria?"},
    {"q": "¿Qué evidencias requiere el CNA sobre los mecanismos de selección y admisión?"},
    {"q": "¿Cómo se evalúa la disponibilidad y actualización del material de apoyo docente?"},
    {"q": "¿Qué importancia tiene la certificación ISO o similar en la acreditación universitaria?"},
    {"q": "¿Cómo influye la vinculación con el sector productivo en la calificación de acreditación?"},
    {"q": "¿Cuál es el procedimiento para presentar recursos ante el CNA por una decisión de no acreditación?"},
    {"q": "¿Qué métricas de carga laboral docente considera el CNA en su evaluación?"},
    {"q": "¿Cómo evalúa el CNA la existencia y funcionamiento de comités curriculares?"},
    {"q": "¿Qué diferencia hay entre acreditación condicional y acreditación plena?"},
    {"q": "¿Cuáles son las tendencias internacionales en acreditación universitaria que reconoce el CNA?"},
    {"q": "¿Cómo se evalúa el programa de tutorías y acompañamiento estudiantil?"},
    {"q": "¿Qué es la red de aseguramiento de la calidad RIACES y su relación con el CNA?"},
    {"q": "¿Cómo afecta la acreditación de alta calidad al puntaje de bienestar institucional?"},
    {"q": "¿Cuáles son los factores que más peso tienen en la decisión de acreditación institucional?"},
]

# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVER (BGE-M3 + Reranker + Qdrant Cloud)
# ═══════════════════════════════════════════════════════════════════════════

print("[INIT] Cargando modelos de recuperación...")
embedder = SentenceTransformer("BAAI/bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

if not QDRANT_URL:
    raise ValueError(
        "Falta QDRANT_URL. Copia la URL exacta del dashboard de Qdrant Cloud y ponla así:\n"
        "  set QDRANT_URL=https://xxxx-xxxx.us-east4-0.gcp.cloud.qdrant.io:6333\n"
        "  set QDRANT_API_KEY=tu_api_key"
    )
qdrant_kwargs = {"url": QDRANT_URL}
if QDRANT_API_KEY:
    qdrant_kwargs["api_key"] = QDRANT_API_KEY
qdrant = QdrantClient(**qdrant_kwargs)
# Verificar conexión antes de empezar
try:
    cols = [c.name for c in qdrant.get_collections().collections]
    print(f"[OK] Qdrant Cloud conectado. Colecciones: {cols}")
    if COLLECTION not in cols:
        raise ValueError(f"Colección '{COLLECTION}' no encontrada. Disponibles: {cols}")
except Exception as e:
    raise ConnectionError(f"No se pudo conectar a Qdrant Cloud ({QDRANT_URL}): {e}")
print("[OK] Retriever listo.")

def retrieve(question: str) -> list[str]:
    vec = embedder.encode(question, normalize_embeddings=True).tolist()
    # Compatible con qdrant-client >= 1.7 (query_points) y < 1.7 (search)
    result = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=TOP_K_RETRIEVE,
        with_payload=True,
    )
    hits = result.points
    if not hits:
        return []
    pairs  = [(question, h.payload.get("text", "")) for h in hits]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    return [h.payload.get("text", "") for _, h in ranked[:TOP_K_RERANK]]

# ═══════════════════════════════════════════════════════════════════════════
# GENERADOR: ChatAcredita PRO (HuggingFace Inference API)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "Eres ChatAcredita PRO, asistente experto en acreditación universitaria colombiana "
    "(CNA, Decreto 1330 de 2019, CESU, Acuerdo 02 de 2020). "
    "Responde ÚNICAMENTE con información del contexto provisto. "
    "Si el contexto no contiene información suficiente, indícalo explícitamente. "
    "Responde en español de forma clara y estructurada."
)

def _extract_content(resp) -> str:
    """Extrae texto de la respuesta — maneja thinking tokens (Qwen3, Claude) donde content puede ser None."""
    choice = resp.choices[0]
    msg = choice.message
    # Caso normal
    if msg.content:
        return msg.content.strip()
    # Qwen3 thinking: el texto real está en reasoning_content o en el último bloque
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return msg.reasoning_content.strip()
    # OpenRouter a veces devuelve content_parts / tool_calls vacíos — intentar raw
    raw = getattr(choice, "text", None)
    if raw:
        return raw.strip()
    return ""


def _call_openai_compat(client, model_id: str, messages: list, label: str,
                        max_tokens: int = 600) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
            )
            text = _extract_content(resp)
            if text:
                return text
            # Si aún vacío, loguear y reintentar
            print(f"  [{label}] Respuesta vacía (attempt {attempt+1}), reintentando...")
            time.sleep(3)
        except Exception as e:
            err = str(e)
            if "loading" in err.lower() or "503" in err:
                print(f"  [{label}] Modelo cargando, espera 20s...")
                time.sleep(20)
            else:
                print(f"  [{label}] Error attempt {attempt+1}: {err[:120]}")
                time.sleep(5)
    return "[ERROR: generación fallida]"


def build_generator():
    """
    Construye el generador según GENERATOR_MODE:
      hf           → HuggingFace Serverless Inference API
      hf_dedicated → HuggingFace Dedicated Endpoint
      openrouter   → OpenRouter (modelo base Qwen3-14B, sin fine-tune)
      ollama       → Ollama local
    """
    mode = GENERATOR_MODE.lower()

    # ── HuggingFace Serverless ────────────────────────────────────────────
    if mode == "hf":
        if not HF_TOKEN:
            raise ValueError("Falta HF_TOKEN para modo 'hf'.")
        # Detectar URL correcta: algunos modelos usan /v1, otros no
        bases_to_try = [
            f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}/v1",
            f"https://api-inference.huggingface.co/v1",
        ]
        client = None
        for base in bases_to_try:
            try:
                test_client = OpenAI(base_url=base, api_key=HF_TOKEN)
                # Ping rápido
                test_client.chat.completions.create(
                    model="tgi",
                    messages=[{"role": "user", "content": "hola"}],
                    max_tokens=5,
                )
                client = test_client
                print(f"[OK] Generador: HuggingFace Serverless ({base})")
                break
            except Exception as e:
                print(f"  [HF probe] {base} → {str(e)[:80]}")
        if client is None:
            print("[WARN] HuggingFace Serverless no responde. Cambiando a OpenRouter automáticamente.")
            mode = "openrouter"

    # ── HuggingFace Dedicated Endpoint ────────────────────────────────────
    if mode == "hf_dedicated":
        if not HF_ENDPOINT_URL:
            raise ValueError("Falta HF_ENDPOINT_URL para modo 'hf_dedicated'.")
        client = OpenAI(base_url=HF_ENDPOINT_URL.rstrip("/") + "/v1", api_key=HF_TOKEN or "hf")
        print(f"[OK] Generador: HuggingFace Dedicated Endpoint ({HF_ENDPOINT_URL})")
        def generate(question: str, chunks: list[str]) -> str:
            ctx = "\n\n---\n\n".join(chunks)
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Contexto:\n{ctx}\n\nPregunta: {question}"}]
            return _call_openai_compat(client, "tgi", msgs, "HF-Dedicated")
        return generate

    # ── OpenRouter (fallback o modo explícito) ────────────────────────────
    if mode in ("openrouter", "hf"):  # "hf" llega aquí si el probe falló
        if not OPENROUTER_KEY:
            raise ValueError("Falta OPENROUTER_API_KEY para modo 'openrouter'.")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_KEY,
            default_headers={
                "HTTP-Referer": "https://eisc.univalle.edu.co",
                "X-Title": "ChatAcredita-Generador",
            },
        )
        print(f"[OK] Generador: OpenRouter ({OPENROUTER_GEN_MODEL})")
        print(f"     NOTA: usando modelo BASE sin fine-tune de acreditación.")
        def generate(question: str, chunks: list[str]) -> str:
            ctx = "\n\n---\n\n".join(chunks)
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Contexto:\n{ctx}\n\nPregunta: {question}"}]
            return _call_openai_compat(client, OPENROUTER_GEN_MODEL, msgs, "OpenRouter-Gen")
        return generate

    # ── Ollama local ──────────────────────────────────────────────────────
    if mode == "ollama":
        ollama_model = os.environ.get("OLLAMA_MODEL", "chatacredita-pro:latest")
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        print(f"[OK] Generador: Ollama local ({ollama_model})")
        def generate(question: str, chunks: list[str]) -> str:
            ctx = "\n\n---\n\n".join(chunks)
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Contexto:\n{ctx}\n\nPregunta: {question}"}]
            return _call_openai_compat(client, ollama_model, msgs, "Ollama")
        return generate

    raise ValueError(f"GENERATOR_MODE desconocido: {GENERATOR_MODE}. Usa: hf | hf_dedicated | openrouter | ollama")

# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS DE EVALUACIÓN RAGAS
# ═══════════════════════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """You are an expert evaluator. Given a question, retrieved context passages, and an answer, score the FAITHFULNESS of the answer (0.0–1.0).
Faithfulness = proportion of claims in the answer that are directly supported by the context.
1.0 = every factual claim is supported · 0.0 = no claims are supported

Context:
{context}

Question: {question}
Answer: {answer}

Respond with ONLY a decimal number between 0.0 and 1.0."""

RELEVANCY_PROMPT = """You are an expert evaluator. Given a question and an answer, score the ANSWER RELEVANCY (0.0–1.0).
Relevancy = how directly and completely the answer addresses the question.
1.0 = directly and completely addresses · 0.0 = completely irrelevant

Question: {question}
Answer: {answer}

Respond with ONLY a decimal number between 0.0 and 1.0."""

CTX_PRECISION_PROMPT = """You are an expert evaluator. Given a question and retrieved context chunks, score CONTEXT PRECISION (0.0–1.0).
Precision = proportion of retrieved chunks that are actually relevant to answering the question.
1.0 = all chunks relevant · 0.0 = no chunks relevant

Question: {question}
Retrieved context:
{context}

Respond with ONLY a decimal number between 0.0 and 1.0."""

CTX_RECALL_PROMPT = """You are an expert evaluator. Given a ground truth answer and retrieved context, score CONTEXT RECALL (0.0–1.0).
Recall = proportion of ground truth information that is present in the retrieved context.
1.0 = all ground truth info is in the context · 0.0 = none of it

Ground truth: {ground_truth}
Retrieved context:
{context}

Respond with ONLY a decimal number between 0.0 and 1.0."""

CORRECTNESS_PROMPT = """You are an expert evaluator. Given a question, a ground truth, and a generated answer, score ANSWER CORRECTNESS (0.0–1.0).
Correctness = factual and semantic overlap between generated answer and ground truth.
1.0 = completely correct and complete · 0.0 = completely wrong

Question: {question}
Ground truth: {ground_truth}
Generated answer: {answer}

Respond with ONLY a decimal number between 0.0 and 1.0."""


def safe_score(text: str) -> float:
    text = text.strip()
    m = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", text)
    return float(m.group()) if m else 0.5


def judge_metrics(judge: dict, question: str, answer: str,
                  chunks: list[str], ground_truth: Optional[str] = None) -> dict:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
        default_headers={
            "HTTP-Referer": "https://eisc.univalle.edu.co",
            "X-Title": "ChatAcredita-MultiJuez",
        },
    )
    ctx_str = "\n\n---\n\n".join(chunks)

    def ask(prompt: str) -> float:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=judge["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=20,
                )
                text = _extract_content(resp)
                if text:
                    return safe_score(text)
                print(f"    [RETRY {attempt+1}] {judge['id']}: respuesta vacía")
                time.sleep(3)
            except Exception as e:
                print(f"    [RETRY {attempt+1}] {judge['id']}: {str(e)[:100]}")
                time.sleep(3)
        return float("nan")

    results = {
        "faithfulness":      ask(FAITHFULNESS_PROMPT.format(context=ctx_str, question=question, answer=answer)),
        "answer_relevancy":  ask(RELEVANCY_PROMPT.format(question=question, answer=answer)),
        "context_precision": ask(CTX_PRECISION_PROMPT.format(question=question, context=ctx_str)),
        "context_recall":    float("nan"),
        "answer_correctness": float("nan"),
    }
    if ground_truth:
        results["context_recall"]     = ask(CTX_RECALL_PROMPT.format(ground_truth=ground_truth, context=ctx_str))
        results["answer_correctness"] = ask(CORRECTNESS_PROMPT.format(question=question, ground_truth=ground_truth, answer=answer))

    return results

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

METRICS = ["faithfulness", "answer_relevancy", "context_precision",
           "context_recall", "answer_correctness"]


def nanmean(vals: list) -> float:
    valid = [v for v in vals if v == v]
    return statistics.mean(valid) if valid else float("nan")


def fmt(v: float) -> str:
    return f"{v:.4f}" if v == v else "nan"


def main():
    if not HF_TOKEN:
        print("[ERROR] Falta HF_TOKEN. Exporta la variable antes de ejecutar.")
        return
    if not OPENROUTER_KEY:
        print("[ERROR] Falta OPENROUTER_API_KEY.")
        return

    print(f"\n{'='*65}")
    print("  ChatAcredita PRO (HuggingFace) — Evaluación Multi-Juez v2")
    print(f"{'='*65}")
    print(f"  Modelo generador : {HF_MODEL_ID}")
    print(f"  Jueces           : {len(JUDGES)}")
    print(f"  Preguntas        : {len(QUERIES)}")
    print(f"{'='*65}\n")

    generate = build_generator()

    csv_path = "multijuez_results_v2.csv"
    csv_cols = (
        ["idx", "question", "has_gt", "answer"]
        + [f"{m}__{j['id']}" for m in METRICS for j in JUDGES]
        + [f"{m}__ensemble" for m in METRICS]
    )
    ensemble_accum = {m: [] for m in METRICS}
    all_rows = []

    with open(csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=csv_cols)
        writer.writeheader()

        for idx, item in enumerate(QUERIES):
            q  = item["q"]
            gt = item.get("gt")
            print(f"\n[{idx+1:03d}/{len(QUERIES)}] {q[:75]}...")

            # 1) Recuperar
            chunks = retrieve(q)
            print(f"  → {len(chunks)} chunks recuperados")

            # 2) Generar con ChatAcredita PRO
            answer = generate(q, chunks)
            print(f"  → Respuesta: {answer[:80]}{'…' if len(answer)>80 else ''}")

            # 3) Evaluar con 5 jueces
            row = {"idx": idx+1, "question": q, "has_gt": bool(gt), "answer": answer}
            judge_results = {}
            for judge in JUDGES:
                print(f"  → Juez {judge['label']}...", end=" ", flush=True)
                m = judge_metrics(judge, q, answer, chunks, gt)
                judge_results[judge["id"]] = m
                for k in METRICS:
                    row[f"{k}__{judge['id']}"] = fmt(m[k])
                print(f"faith={m['faithfulness']:.3f} rel={m['answer_relevancy']:.3f} prec={m['context_precision']:.3f}")
                time.sleep(0.5)

            # 4) Ensemble
            for k in METRICS:
                vals = [judge_results[j["id"]][k] for j in JUDGES]
                ens  = nanmean(vals)
                row[f"{k}__ensemble"] = fmt(ens)
                if ens == ens:
                    ensemble_accum[k].append(ens)

            writer.writerow(row)
            fout.flush()
            all_rows.append(row)

    # ─── Resumen ──────────────────────────────────────────────────────────
    summary = {
        "model_generator": HF_MODEL_ID,
        "n_queries":  len(QUERIES),
        "n_with_gt":  sum(1 for q in QUERIES if q.get("gt")),
        "judges":     [j["label"] for j in JUDGES],
        "ensemble_global": {},
        "per_judge_global": {},
    }

    print(f"\n{'='*65}")
    print("  ENSEMBLE (promedio 5 jueces) — RESULTADO FINAL")
    print(f"{'='*65}")
    for k in METRICS:
        val = nanmean(ensemble_accum[k])
        summary["ensemble_global"][k] = round(val, 4) if val == val else None
        label = f"{k:25s}"
        print(f"  {label}: {val:.4f}" if val == val else f"  {label}: N/A")

    print(f"\n{'='*65}")
    print("  DESGLOSE POR JUEZ")
    print(f"{'='*65}")
    for judge in JUDGES:
        per = {}
        for k in METRICS:
            col = f"{k}__{judge['id']}"
            vals = []
            for r in all_rows:
                try:
                    v = float(r.get(col, "nan"))
                    if v == v:
                        vals.append(v)
                except Exception:
                    pass
            per[k] = round(nanmean(vals), 4) if vals else None
        summary["per_judge_global"][judge["id"]] = per
        print(f"\n  ▸ {judge['label']}")
        for k, v in per.items():
            print(f"    {k:25s}: {v}" if v is not None else f"    {k:25s}: N/A")

    json_path = "multijuez_summary_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✓ CSV  → {csv_path}")
    print(f"✓ JSON → {json_path}")
    print("\nRecuerda actualizar el paper con los resultados del ensemble.")


if __name__ == "__main__":
    main()
