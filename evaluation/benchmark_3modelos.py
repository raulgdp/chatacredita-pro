"""
Benchmark comparativo — 3 modelos ChatAcredita
================================================
Genera respuestas de los 3 modelos fine-tuned sobre las 142 queries
del benchmark CNA y calcula RAGAS multi-juez (ensemble).

Modelos:
  • raulgdp/qwen3-14b-acredita        (ChatAcredita PRO)
  • raulgdp/deepseek14b-acredita      (DeepSeek-14B-Acredita)
  • raulgdp/qwen2.5-7b-acredita-cna-col (Qwen2.5-7B-Acredita)

Requisitos:
  pip install transformers accelerate torch ragas datasets groq pandas tqdm

Uso (RTX 4090 recomendado — carga un modelo a la vez):
  python benchmark_3modelos.py \
      --queries benchmark_queries.txt \
      --rag_url  http://localhost:8000  \
      --output   comparativo_3modelos.csv

  Si NO tienes RAG activo, usa --no_rag para modo standalone
  (el modelo responde solo con su conocimiento parametric).

Formato de benchmark_queries.txt: una pregunta por línea.
También acepta el CSV anterior: --queries multijuez_results_v2.csv
  (extrae la columna 'question' automáticamente).
"""

import os
import gc
import json
import argparse
import time
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Configuración de modelos ───────────────────────────────────────────────────

MODELS = [
    {
        "key":      "ChatAcredita_PRO",
        "hf_id":    "raulgdp/qwen3-14b-acredita",
        "sys_prompt": (
            "Eres ChatAcredita PRO, un asistente experto en acreditación "
            "universitaria bajo el marco del CNA en Colombia."
        ),
        "dtype":    torch.bfloat16,
        "enable_thinking": False,   # Qwen3 soporta thinking mode; lo desactivamos para eval
    },
    {
        "key":      "DeepSeek14B_Acredita",
        "hf_id":    "raulgdp/deepseek14b-acredita",
        "sys_prompt": (
            "Eres un asistente experto en acreditación universitaria colombiana "
            "bajo el marco del CNA (Decreto 1330/2019)."
        ),
        "dtype":    torch.float16,
        "enable_thinking": False,
    },
    {
        "key":      "Qwen25_7B_Acredita",
        "hf_id":    "raulgdp/qwen2.5-7b-acredita-cna-col",
        "sys_prompt": (
            "Eres un asistente experto en acreditación universitaria colombiana "
            "bajo el marco del CNA (Decreto 1330/2019)."
        ),
        "dtype":    torch.bfloat16,
        "enable_thinking": False,
    },
]

# ── Jueces RAGAS (vía Groq — gratuito) ────────────────────────────────────────

JUDGES = {
    "gemma4-31b":  "gemma2-9b-it",          # proxy — Gemma 4 no está en Groq aún; usar Gemma 2 9B
    "llama-70b":   "llama-3.3-70b-versatile",
    "gpt4o-mini":  "llama-3.1-8b-instant",   # proxy económico si no tienes OpenAI key
}
# Si tienes OPENAI_API_KEY, cambia "gpt4o-mini" a gpt-4o-mini real:
# JUDGES["gpt4o-mini"] = "gpt-4o-mini"  # requiere openai, no groq

MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.7


# ── Cargar queries ─────────────────────────────────────────────────────────────

def load_queries(path: str) -> list[str]:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        col = "question" if "question" in df.columns else df.columns[1]
        return df[col].dropna().tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]


# ── Recuperación RAG opcional ──────────────────────────────────────────────────

def retrieve_context(query: str, rag_url: str, top_k: int = 3) -> str:
    """
    Llama al endpoint RAG (Qdrant + bge-m3 + BM25 RRF).
    Asume API REST: POST /retrieve  →  {"query": "...", "top_k": N}
    Devuelve los chunks concatenados.
    Si rag_url es None devuelve string vacío.
    """
    if not rag_url:
        return ""
    try:
        import requests
        r = requests.post(
            f"{rag_url}/retrieve",
            json={"query": query, "top_k": top_k},
            timeout=10,
        )
        r.raise_for_status()
        chunks = r.json().get("chunks", [])
        return "\n\n".join(c.get("text", "") for c in chunks)
    except Exception as e:
        print(f"  ⚠️  RAG error: {e} — usando modo standalone")
        return ""


# ── Generación de respuesta con un modelo ─────────────────────────────────────

def generate_answer(
    model, tokenizer, query: str, context: str, cfg: dict
) -> str:
    user_content = query
    if context:
        user_content = (
            f"Contexto recuperado:\n{context}\n\n"
            f"Pregunta: {query}"
        )

    messages = [
        {"role": "system", "content": cfg["sys_prompt"]},
        {"role": "user",   "content": user_content},
    ]

    # Qwen3 soporta enable_thinking en apply_chat_template
    tpl_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if cfg.get("enable_thinking") is not None:
        try:
            text = tokenizer.apply_chat_template(
                messages, enable_thinking=cfg["enable_thinking"], **tpl_kwargs
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, **tpl_kwargs)
    else:
        text = tokenizer.apply_chat_template(messages, **tpl_kwargs)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    return answer


# ── RAGAS multi-juez simplificado (faithfulness via LLM) ──────────────────────

def ragas_faithfulness_judge(
    query: str, context: str, answer: str, judge_model: str, client
) -> float:
    """
    Fidelidad 0–1: ¿cada afirmación de la respuesta está en el contexto?
    Retorna float o None si falla.
    """
    if not context:
        return None

    prompt = f"""You are a factual consistency evaluator.

Question: {query}
Context: {context[:1500]}
Answer: {answer[:1000]}

Task: What fraction of the factual claims in the Answer are supported by the Context?
Respond with ONLY a JSON: {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""

    try:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        start = raw.find("{"); end = raw.rfind("}") + 1
        return float(json.loads(raw[start:end])["score"])
    except Exception as e:
        print(f"    judge error: {e}")
        return None


# ── Pipeline principal ─────────────────────────────────────────────────────────

def run_benchmark(queries: list[str], rag_url: str, output_path: str):
    records = []

    for model_cfg in MODELS:
        key    = model_cfg["key"]
        hf_id  = model_cfg["hf_id"]

        print(f"\n{'='*60}")
        print(f"  Cargando modelo: {key}  ({hf_id})")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=model_cfg["dtype"],
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print(f"  ✅ Modelo cargado en: {next(model.parameters()).device}")

        for i, query in enumerate(tqdm(queries, desc=f"  {key}")):
            context = retrieve_context(query, rag_url)
            answer  = generate_answer(model, tokenizer, query, context, model_cfg)

            records.append({
                "idx":     i + 1,
                "model":   key,
                "question": query,
                "context":  context[:500],   # truncado para CSV
                "answer":   answer,
            })

        # Liberar VRAM antes de cargar el siguiente modelo
        print(f"\n  🔄 Liberando VRAM...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

    # Guardar respuestas
    df = pd.DataFrame(records)
    raw_path = output_path.replace(".csv", "_respuestas.csv")
    df.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"\n✅ Respuestas guardadas: {raw_path}")

    # ── Tabla pivotada (una fila por query, columnas por modelo) ──────────────
    pivot = df.pivot_table(
        index=["idx", "question"],
        columns="model",
        values="answer",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot_path = output_path.replace(".csv", "_pivot.csv")
    pivot.to_csv(pivot_path, index=False, encoding="utf-8")
    print(f"✅ Tabla pivotada guardada: {pivot_path}")
    print("\n📌 Próximo paso: pasa el CSV pivotado al script RAGAS multi-juez")
    print("   o al script geval_acredita.py para evaluación G-Eval.")

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera respuestas de 3 modelos ChatAcredita sobre benchmark CNA"
    )
    parser.add_argument(
        "--queries", required=True,
        help="Archivo con preguntas: .txt (una por línea) o .csv (columna 'question')"
    )
    parser.add_argument(
        "--rag_url", default=None,
        help="URL del servidor RAG, ej: http://localhost:8000 (omitir = modo standalone)"
    )
    parser.add_argument(
        "--output", default="comparativo_3modelos.csv",
        help="Nombre base del CSV de salida"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["ChatAcredita_PRO", "DeepSeek14B_Acredita", "Qwen25_7B_Acredita"],
        help="Subset de modelos a correr (por key)"
    )
    args = parser.parse_args()

    # Filtrar modelos si se especificó subset
    filtered = [m for m in MODELS if m["key"] in args.models]
    MODELS.clear()
    MODELS.extend(filtered)
    print(f"Modelos seleccionados: {[m['key'] for m in MODELS]}")

    queries = load_queries(args.queries)
    print(f"Queries cargadas: {len(queries)}")

    run_benchmark(queries, args.rag_url, args.output)
