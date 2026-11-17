# generar_dataset_mejorado.py

Script para generar el dataset de 15 000 pares Q/A de acreditación universitaria colombiana (CNA) que se usa para entrenar ChatAcredita PRO. Implementa la estrategia **RAFT** (Retrieval-Augmented Fine-Tuning): cada pregunta se genera dándole al LLM el fragmento correcto más 2 distractores de otros PDFs, forzando respuestas específicas que no pueden contestarse con conocimiento general.

---

## Instalación

```bash
pip install groq pymupdf pymupdf4llm tiktoken tqdm
```

Configurar la API Key de Groq:

```bash
export GROQ_API_KEY="tu_api_key"
```

---

## Uso rápido

```bash
# Generar desde cero con los PDFs del corpus CNA
python generar_dataset_mejorado.py --pdfs ./corpus --output dataset_mejorado.json

# Fusionar con dataset existente + generar nuevos pares
python generar_dataset_mejorado.py \
  --pdfs ./corpus \
  --original dataset_acreditacion.json \
  --output dataset_mejorado.json

# Solo limpiar y deduplicar un dataset existente (sin llamadas a la API)
python generar_dataset_mejorado.py \
  --original dataset_acreditacion.json \
  --output dataset_limpio.json \
  --solo-limpia
```

---

## Argumentos

| Argumento | Valor por defecto | Descripción |
|-----------|------------------|-------------|
| `--pdfs` | `./corpus` | Carpeta con los PDFs fuente |
| `--original` | *(vacío)* | Dataset anterior para limpiar y fusionar |
| `--output` | `dataset_mejorado.json` | Archivo de salida |
| `--model` | `llama-3.3-70b-versatile` | Modelo de Groq |
| `--max-chunks` | `50` | Máx. chunks por PDF (`-1` = todos) |
| `--workers` | `3` | Threads paralelos para las llamadas a la API |
| `--api-key` | `$GROQ_API_KEY` | API Key de Groq (alternativa a la variable de entorno) |
| `--solo-limpia` | — | Solo deduplicar el dataset original; no llama a la API |

---

## Pipeline interno

```
PDFs ──► chunks (600 chars, overlap 100) ──► filtro calidad
                                                   │
                                     prompt RAFT: chunk correcto
                                     + 2 distractores de otros PDFs
                                                   │
                                         Groq API (llama-3.3-70b)
                                                   │
                                     validación: len ≥ 20/40 chars,
                                     termina en "?", sin frases de
                                     alucinación
                                                   │
              dataset_original ──► limpieza ───────►│
                                                   ▼
                              dedup exacto (MD5) + fuzzy (Jaccard bigrams, θ=0.85)
                                                   │
                                       dataset_mejorado.json
```

### Plantillas de pregunta (15 en total, 5 categorías)

| Categoría | Peso | Ejemplos |
|-----------|------|---------|
| Factual | 25 % | "¿Qué establece el texto sobre {tema}?" |
| Procedimental | 20 % | "¿Cuál es el proceso descrito para {tema}?" |
| Causal | 20 % | "¿Por qué es importante {tema} en la acreditación?" |
| Comparativa | 20 % | "¿Qué diferencias se mencionan respecto a {tema}?" |
| Multi-factor | 15 % | "¿Cómo contribuye {tema} a la autoevaluación del programa?" |

### Filtros de calidad del par generado (`es_par_valido`)

- Pregunta ≥ 20 caracteres y termina en `?`
- Respuesta ≥ 40 caracteres
- Sin frases de alucinación: `"como inteligencia artificial"`, `"no tengo acceso"`, `"el fragmento no menciona"`, etc.

### Deduplicación

1. **Exacta**: hash MD5 sobre el texto de la pregunta en minúsculas
2. **Fuzzy**: similitud de Jaccard sobre bigramas, umbral 0.85, ventana deslizante de 2 000 pares

---

## Formato de salida

Cada par en `dataset_mejorado.json` sigue el formato ChatML:

```json
{
  "id": "decreto1330_a3f4b2c1d8e9f012",
  "source": "decreto1330.pdf",
  "messages": [
    {"role": "user",      "content": "¿Qué establece el Decreto 1330 respecto a...?"},
    {"role": "assistant", "content": "El Decreto 1330 de 2019 establece que..."}
  ]
}
```

---

## Resultados en ChatAcredita PRO

Con el corpus CNA de EISC/Univalle (10 GB, ~40 PDFs):

| Métrica | Valor |
|---------|-------|
| Pares generados | 15 000 |
| Pares aceptados por RAG-VSD | 9 847 (65.6 %) |
| Coste estimado (Groq free tier) | ~USD 0 — USD 19 |
| Tiempo aproximado (3 workers) | ~5 horas |

> **Nota**: El script utiliza `llama-3.3-70b-versatile` vía Groq. El paper de ChatAcredita PRO documenta GPT-4o Mini como generador original; ambos producen datasets funcionalmente equivalentes para el corpus CNA.

---

## Dependencias completas

```
groq>=0.9.0
pymupdf>=1.24.0
pymupdf4llm>=0.0.17
tiktoken>=0.7.0
tqdm>=4.66.0
```
