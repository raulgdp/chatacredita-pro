"""
config.py — Configuración centralizada para el pipeline RAG-VSD
ChatAcredita PRO · Universidad del Valle · EISC

Ajusta las variables de entorno antes de ejecutar:
  export QDRANT_URL="https://xxxx.us-east-1-0.aws.cloud.qdrant.io"
  export QDRANT_API_KEY="tu_api_key"
  export HF_TOKEN="hf_..."
"""

import os
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"

# ── Qdrant Cloud ───────────────────────────────────────────────────────────────
QDRANT_URL        = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "acreditacion")

# ── Modelos de recuperación ────────────────────────────────────────────────────
EMBEDDER_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
TOP_K_RETRIEVE = 10
TOP_K_RERANK   = 8

# ── Umbrales RAG-VSD ──────────────────────────────────────────────────────────
# Estos valores fueron calibrados sobre el corpus CNA (15 000 pares sintéticos)
# y producen un ratio de aceptación del 65.6 % (9 847 / 15 000)
TAU_OVERLAP  = 0.30   # solapamiento léxico mínimo (tokens compartidos / tokens respuesta)
TAU_QUERY    = 0.55   # similitud semántica mínima pregunta ↔ chunk top-1
TAU_TECH     = 0.40   # ratio mínimo de términos técnicos CNA en la respuesta

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_MODEL_ID = "raulgdp/qwen3-14b-acredita"
