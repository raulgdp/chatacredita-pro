# -*- coding: utf-8 -*-
"""
G-Eval (LLM-as-a-Judge) para ChatAcredita
==========================================
Evalua los 3 modelos fine-tuned usando GPT-4o Mini via OpenRouter como juez.
Complementa RAGAS con metricas independientes y reproducibles.

Dimensiones evaluadas (escala 1-5):
  - Faithfulness  -- Las afirmaciones estan respaldadas por el contexto CNA?
  - Relevance     -- La respuesta responde directamente la pregunta?
  - Completeness  -- Cubre todos los aspectos clave de la pregunta?
  - Coherence     -- Es clara, bien estructurada y sin contradicciones?

Uso:
  pip install openai pandas tqdm bert-score rouge-score
  set OPENROUTER_API_KEY=sk-or-...
  python geval_acredita.py --input benchmark_142.json --output resultados_geval.csv
"""

import os
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# -- Configuracion -----------------------------------------------------------

JUDGE_MODEL  = "openai/gpt-4o-mini"
TEMPERATURE  = 0.0
MAX_RETRIES  = 3
SLEEP_RETRY  = 5


def get_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "Falta OPENROUTER_API_KEY.\n"
            "Consiguelo en https://openrouter.ai/keys\n"
            "Luego: set OPENROUTER_API_KEY=sk-or-..."
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/raulgdp",
            "X-Title":      "ChatAcredita G-Eval",
        },
    )


MODELS = {
    "ChatAcredita_PRO":        "raulgdp/qwen3-14b-acredita",
    "Qwen2.5-7B-Acredita":     "raulgdp/qwen2.5-7b-acredita",
    "DeepSeek-14B-Acredita":   "raulgdp/deepseek-14b-acredita",
}

# -- Prompts G-Eval ----------------------------------------------------------

DIMENSIONS = {
    "faithfulness": {
        "name": "Faithfulness",
        "description": (
            "Does every factual claim in the answer appear in or follow directly "
            "from the provided context? Claims not supported by the context should lower the score."
        ),
        "steps": [
            "Read the context carefully.",
            "Identify each factual claim made in the answer.",
            "Check whether each claim is supported by the context.",
            "Assign a score from 1 (many unsupported claims) to 5 (all claims grounded in the context).",
        ],
    },
    "relevance": {
        "name": "Answer Relevance",
        "description": (
            "Does the answer directly and completely address the question asked? "
            "Off-topic or tangential information should lower the score."
        ),
        "steps": [
            "Read the question carefully.",
            "Determine whether the answer addresses exactly what was asked.",
            "Penalise irrelevant or off-topic content.",
            "Assign a score from 1 (largely irrelevant) to 5 (perfectly on-target).",
        ],
    },
    "completeness": {
        "name": "Completeness",
        "description": (
            "Does the answer cover all key aspects required by the question? "
            "Missing important information should lower the score."
        ),
        "steps": [
            "Identify the key aspects or sub-questions implied by the question.",
            "Check whether the answer addresses each key aspect.",
            "Penalise omissions of important information.",
            "Assign a score from 1 (major omissions) to 5 (fully comprehensive).",
        ],
    },
    "coherence": {
        "name": "Coherence",
        "description": (
            "Is the answer well-structured, logically connected, and easy to follow? "
            "Contradictions, repetitions, or disorganised writing should lower the score."
        ),
        "steps": [
            "Read the answer for logical flow and structure.",
            "Check for contradictions or unsupported jumps.",
            "Assess readability and organisation.",
            "Assign a score from 1 (incoherent) to 5 (clear and well-structured).",
        ],
    },
}


def build_prompt(question, context, answer, dim):
    steps_text = "\n".join("  %d. %s" % (i + 1, s) for i, s in enumerate(dim["steps"]))
    return (
        "You are an expert evaluator for a Colombian university accreditation (CNA) QA system.\n"
        "Your task is to rate the quality of an AI assistant's answer on ONE specific criterion.\n\n"
        "=== CRITERION: %s ===\n%s\n\n"
        "Evaluation steps:\n%s\n\n"
        "=== INPUT ===\n"
        "Question: %s\n\n"
        "Context (retrieved from CNA documents):\n%s\n\n"
        "Answer to evaluate:\n%s\n\n"
        "=== OUTPUT ===\n"
        'Respond with ONLY a JSON object: {"score": <integer 1-5>, "reason": "<one sentence>"}'
    ) % (dim["name"], dim["description"], steps_text, question, context[:2000], answer)


# -- Llamada al juez ---------------------------------------------------------

def judge(client, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            return json.loads(raw[start:end])
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print("  Reintento %d/%d: %s" % (attempt + 1, MAX_RETRIES, e))
                time.sleep(SLEEP_RETRY)
            else:
                return {"score": None, "reason": "ERROR: %s" % e}


# -- Evaluacion G-Eval -------------------------------------------------------

def run_geval(benchmark_path, output_path):
    client = get_client()

    if benchmark_path.endswith(".csv"):
        df_in = pd.read_csv(benchmark_path)
        meta_cols = {"idx", "question", "context", "has_gt"}
        model_cols = [c for c in df_in.columns if c not in meta_cols]
        data = []
        for _, row in df_in.iterrows():
            item = {
                "query":   row.get("question", ""),
                "context": row.get("context", ""),
            }
            for mc in model_cols:
                item["%s_answer" % mc] = row.get(mc, "")
            data.append(item)
        MODELS.clear()
        MODELS.update({mc: mc for mc in model_cols})
    else:
        with open(benchmark_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    print("Queries cargadas: %d. Modelos: %s" % (len(data), list(MODELS.keys())))
    print("Juez: %s via OpenRouter\n" % JUDGE_MODEL)

    records = []

    for i, item in enumerate(tqdm(data, desc="Evaluando queries")):
        query   = item.get("query", item.get("question", ""))
        context = item.get("context", "")

        for model_key in MODELS:
            answer_key = "%s_answer" % model_key
            answer = item.get(answer_key, "")
            if not answer:
                continue

            row = {
                "query_id": i,
                "query":    query,
                "model":    model_key,
                "answer":   answer[:500],
            }

            for dim_key, dim in DIMENSIONS.items():
                prompt = build_prompt(query, context, answer, dim)
                result = judge(client, prompt)
                row["geval_%s" % dim_key]        = result.get("score")
                row["geval_%s_reason" % dim_key] = result.get("reason", "")
                time.sleep(0.3)

            scores = [row["geval_%s" % d] for d in DIMENSIONS if row["geval_%s" % d] is not None]
            row["geval_composite"] = round(sum(scores) / len(scores), 3) if scores else None

            records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print("\nResultados guardados en: %s" % output_path)

    print("\n" + "=" * 60)
    print("RESUMEN G-EVAL -- SCORES PROMEDIO POR MODELO")
    print("=" * 60)

    cols = ["geval_faithfulness", "geval_relevance",
            "geval_completeness", "geval_coherence", "geval_composite"]

    summary = (
        df.groupby("model")[cols]
        .mean()
        .round(3)
        .rename(columns={
            "geval_faithfulness":  "Faithfulness",
            "geval_relevance":     "Relevance",
            "geval_completeness":  "Completeness",
            "geval_coherence":     "Coherence",
            "geval_composite":     "Composite",
        })
    )
    print(summary.to_string())

    summary_path = output_path.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, encoding="utf-8")
    print("\nResumen guardado en: %s" % summary_path)
    return df, summary


# -- BERTScore ---------------------------------------------------------------

def run_bertscore(benchmark_path, output_path):
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("bert-score no instalado. Ejecuta: pip install bert-score")
        return

    if benchmark_path.endswith(".csv"):
        print("BERTScore requiere gold_answer -- omitido para CSV sin gold answers")
        return

    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\nCalculando BERTScore para %d queries..." % len(data))
    records = []

    for model_key in MODELS:
        refs  = [item.get("gold_answer", "") for item in data]
        hyps  = [item.get("%s_answer" % model_key, "") for item in data]

        valid = [(r, h) for r, h in zip(refs, hyps) if r and h]
        if not valid:
            print("  %s: sin gold answers -- omitiendo" % model_key)
            continue

        refs_v, hyps_v = zip(*valid)
        _, _, F1 = bert_score_fn(
            list(hyps_v), list(refs_v),
            lang="es",
            model_type="dccuchile/bert-base-spanish-wwm-cased",
            verbose=False,
        )
        mean_f1 = F1.mean().item()
        print("  %s: BERTScore F1 = %.4f" % (model_key, mean_f1))
        records.append({"model": model_key, "bertscore_f1": round(mean_f1, 4)})

    bs_df = pd.DataFrame(records)
    bs_path = output_path.replace(".csv", "_bertscore.csv")
    bs_df.to_csv(bs_path, index=False)
    print("BERTScore guardado en: %s" % bs_path)
    return bs_df


# -- ROUGE-L -----------------------------------------------------------------

def run_rouge(benchmark_path, output_path):
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("rouge-score no instalado. Ejecuta: pip install rouge-score")
        return

    if benchmark_path.endswith(".csv"):
        print("ROUGE-L requiere gold_answer -- omitido para CSV sin gold answers")
        return

    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    records = []

    for model_key in MODELS:
        scores = []
        for item in data:
            gold = item.get("gold_answer", "")
            hyp  = item.get("%s_answer" % model_key, "")
            if gold and hyp:
                s = scorer.score(gold, hyp)
                scores.append(s["rougeL"].fmeasure)

        if scores:
            mean_r = sum(scores) / len(scores)
            print("  %s: ROUGE-L = %.4f" % (model_key, mean_r))
            records.append({"model": model_key, "rougeL": round(mean_r, 4)})

    r_df = pd.DataFrame(records)
    r_path = output_path.replace(".csv", "_rouge.csv")
    r_df.to_csv(r_path, index=False)
    print("ROUGE-L guardado en: %s" % r_path)
    return r_df


# -- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-Eval + BERTScore + ROUGE-L para ChatAcredita")
    parser.add_argument("--input",  required=True,
                        help="JSON o CSV pivot con respuestas de los modelos")
    parser.add_argument("--output", required=True,
                        help="Path CSV para resultados G-Eval")
    parser.add_argument("--judge",  default=JUDGE_MODEL,
                        help="Modelo juez OpenRouter (default: %s)" % JUDGE_MODEL)
    parser.add_argument("--skip-geval",     action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--skip-rouge",     action="store_true")
    args = parser.parse_args()

    if args.judge != JUDGE_MODEL:
        JUDGE_MODEL = args.judge
        print("Juez cambiado a: %s" % JUDGE_MODEL)

    if not args.skip_geval:
        run_geval(args.input, args.output)

    if not args.skip_bertscore:
        run_bertscore(args.input, args.output)

    if not args.skip_rouge:
        run_rouge(args.input, args.output)

    print("\nEvaluacion completa.")
