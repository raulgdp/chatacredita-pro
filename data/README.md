# data/

Este directorio contiene el benchmark de evaluación y la estructura esperada para el pipeline de filtrado RAG-VSD.

## benchmark_142q.jsonl

**142 preguntas** de evaluación sobre el marco CNA / Decreto 1330, construidas para medir la calidad de los modelos ChatAcredita. El archivo tiene el siguiente formato:

```json
{"q": "¿Cuáles son los grupos de investigación de la EISC?", "gt": "Los grupos de investigación activos..."}
{"q": "¿Qué es la autoevaluación en el contexto de la acreditación universitaria colombiana?"}
```

Campos:
- `q` : pregunta en español
- `gt` (opcional): ground truth — sólo disponible para 13 de las 142 preguntas (3 EISC + 10 CNA general)

> Las preguntas sin `gt` se evalúan únicamente con las métricas que no requieren referencia (faithfulness, answer_relevancy, context_precision).

## Estructura esperada para RAG-VSD

```
data/
├── synthetic_raw/
│   └── synthetic_15k.jsonl      ← 15 000 pares generados (no incluidos)
└── synthetic_verified/
    ├── verified.jsonl            ← 9 847 pares aceptados por RAG-VSD
    ├── rejected.jsonl            ← 5 153 pares rechazados
    └── checkpoint.json           ← punto de reanudación automática
```

Los pares sintéticos brutos (`synthetic_15k.jsonl`) se generaron con el pipeline RAFT descrito en §3.2 del paper y **no se distribuyen** por razones de tamaño (> 500 MB). El archivo `verified.jsonl` (dataset de entrenamiento final) está disponible en HuggingFace: [`raulgdp/cna-acredita-dataset`](https://huggingface.co/datasets/raulgdp/cna-acredita-dataset).
