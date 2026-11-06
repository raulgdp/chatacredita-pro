"""
utils/qdrant_client.py — Verificador RAG-VSD
============================================
Implementa los tres umbrales del filtro RAG-VSD descrito en el paper:

  τ_overlap  (TAU_OVERLAP = 0.30) : solapamiento léxico respuesta ↔ chunks
  τ_query    (TAU_QUERY   = 0.55) : similitud semántica pregunta ↔ chunk top-1
  τ_tech     (TAU_TECH    = 0.40) : densidad de términos CNA en la respuesta

Un par sintético (pregunta, respuesta) se acepta sólo si las tres
condiciones se cumplen simultáneamente.

Referencia: Gutiérrez de Piñerez et al. (2026), §3.3 RAG-VSD Data Filtering.
"""

import re
import sys
from pathlib import Path
from typing import Optional

try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError as e:
    print(f"[ERROR] Falta dependencia: {e}")
    print("Instala con: pip install qdrant-client sentence-transformers")
    raise

# Términos técnicos del marco CNA / Decreto 1330 usados para τ_tech
CNA_TERMS = {
    "acreditación", "cna", "decreto", "calidad", "factor", "característica",
    "autoevaluación", "pares", "cesu", "men", "minciencias", "registro",
    "programa", "institución", "currículo", "docente", "investigación",
    "extensión", "bienestar", "egresados", "internacionalización",
    "pertinencia", "coherencia", "formación", "renovación", "lineamientos",
    "condición", "visita", "informe", "mejoramiento", "indicador",
}


def _tokenize(text: str) -> set:
    """Tokens en minúsculas, sin puntuación, longitud >= 3."""
    return {w for w in re.findall(r"\b[a-záéíóúüñ]{3,}\b", text.lower())}


def _overlap_score(response: str, chunks: list[str]) -> float:
    """Fracción de tokens de la respuesta presentes en algún chunk."""
    resp_tokens = _tokenize(response)
    if not resp_tokens:
        return 0.0
    ctx_tokens = set()
    for c in chunks:
        ctx_tokens |= _tokenize(c)
    return len(resp_tokens & ctx_tokens) / len(resp_tokens)


def _tech_ratio(response: str) -> float:
    """Fracción de tokens de la respuesta que son términos CNA."""
    tokens = _tokenize(response)
    if not tokens:
        return 0.0
    return len(tokens & CNA_TERMS) / len(tokens)


class RAGVerifier:
    """
    Verifica pares (pregunta, respuesta) contra el índice Qdrant.

    Parameters
    ----------
    qdrant_url : str
        URL del cluster Qdrant Cloud.
    qdrant_api_key : str
        API key de Qdrant Cloud.
    collection : str
        Nombre de la colección (default: "acreditacion").
    timeout : int
        Timeout en segundos para las peticiones a Qdrant (default: 120).
    tau_overlap : float
        Umbral de solapamiento léxico (default: 0.30).
    tau_query : float
        Umbral de similitud semántica (default: 0.55).
    tau_tech : float
        Umbral de densidad de términos CNA (default: 0.40).
    top_k : int
        Chunks a recuperar por consulta (default: 10).
    """

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection: str = "acreditacion",
        timeout: int = 120,
        tau_overlap: float = 0.30,
        tau_query: float = 0.55,
        tau_tech: float = 0.40,
        top_k: int = 10,
    ):
        # Importamos aquí para evitar carga innecesaria si sólo se usa config
        sys.path.insert(0, str(Path(__file__).parents[1]))

        self.collection  = collection
        self.tau_overlap = tau_overlap
        self.tau_query   = tau_query
        self.tau_tech    = tau_tech
        self.top_k       = top_k

        print("[INIT] Cargando embedder BAAI/bge-m3 ...")
        self.embedder = SentenceTransformer("BAAI/bge-m3")
        print("[INIT] Cargando reranker BAAI/bge-reranker-v2-m3 ...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

        self.qdrant = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=timeout,
        )
        cols = [c.name for c in self.qdrant.get_collections().collections]
        if collection not in cols:
            raise ValueError(
                f"Colección '{collection}' no encontrada en Qdrant. "
                f"Colecciones disponibles: {cols}"
            )
        print(f"[OK] RAGVerifier listo (colección: {collection})")

    # ------------------------------------------------------------------
    def _retrieve(self, query: str) -> tuple[list[str], float]:
        """
        Recupera chunks relevantes y devuelve (textos, score_top1).
        score_top1 es la similitud coseno del chunk más parecido a la query.
        """
        vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        result = self.qdrant.query_points(
            collection_name=self.collection,
            query=vec,
            limit=self.top_k,
            with_payload=True,
        )
        hits = result.points
        if not hits:
            return [], 0.0

        # Reranking
        pairs  = [(query, h.payload.get("text", "")) for h in hits]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)

        texts     = [h.payload.get("text", "") for _, h in ranked]
        score_top = float(ranked[0][0]) if ranked else 0.0
        # Normalizar score reranker a [0, 1] usando sigmoide
        import math
        score_top = 1 / (1 + math.exp(-score_top))
        return texts, score_top

    # ------------------------------------------------------------------
    def verify(self, question: str, response: str) -> dict:
        """
        Aplica el filtro RAG-VSD y devuelve un dict con los scores y la
        decisión de aceptación.

        Returns
        -------
        dict con claves:
          supported        : bool  — True si el par pasa los tres umbrales
          overlap_score    : float — τ_overlap calculado
          qdrant_avg_score : float — similitud semántica top-1 (τ_query)
          tech_ratio       : float — densidad de términos CNA (τ_tech)
          chunks_found     : int   — número de chunks recuperados
        """
        chunks, query_score = self._retrieve(question)
        overlap = _overlap_score(response, chunks)
        tech    = _tech_ratio(response)

        accepted = (
            overlap    >= self.tau_overlap
            and query_score >= self.tau_query
            and tech       >= self.tau_tech
        )

        return {
            "supported":        accepted,
            "overlap_score":    round(overlap, 4),
            "qdrant_avg_score": round(query_score, 4),
            "tech_ratio":       round(tech, 4),
            "chunks_found":     len(chunks),
        }

    def close(self):
        self.qdrant.close()
