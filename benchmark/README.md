# CNA-EISC-320 Benchmark

320-question evaluation benchmark for the Colombian university accreditation domain (CNA framework, EISC — Universidad del Valle).

## Files

| File | Questions | Description |
|------|-----------|-------------|
| `benchmark_342.json` | 320 | Full benchmark (expert + corpus) |
| `benchmark_178_new.json` | 178 | Corpus-derived questions only |

## Composition

- **142** expert-curated questions (manually written by domain expert)
- **178** corpus-derived questions (extracted from accreditation documents using LLM pipeline)

## Question Format

```json
{
  "id": "N01",
  "category": "normative",
  "question": "¿Cuáles son los diez factores del modelo de acreditación CNA?",
  "reference": "Los diez factores son: Misión y Proyecto Institucional, ...",
  "source": "Lineamientos_CNA_2020.pdf",
  "page": 14,
  "tipo": "texto"
}
```

## Category Distribution

| Category | n | Description |
|----------|---|-------------|
| `normative` | 55 | Regulatory questions (CNA framework, decrees, resolutions) |
| `statistical` | 77 | Numerical questions (counts, rates, rankings) |
| `process` | 54 | Procedural questions (steps, workflows) |
| `comparison` | 29 | Comparative questions (between years, programs, factors) |
| `synthesis` | 52 | Summary/overview questions |
| `general` | 53 | Other accreditation questions |

## Usage

```python
import json

with open("benchmark_342.json") as f:
    questions = json.load(f)

print(f"Total: {len(questions)} questions")
print(f"Categories: {set(q['category'] for q in questions)}")
```

## How to evaluate

```bash
# Evaluate Sys-C
python ../evaluation/eval_sysC.py \
  --benchmark benchmark_342.json \
  --output results_sysC.json \
  --delay 3

# Evaluate Sys-B and Sys-D
python ../evaluation/eval_sysB_sysD.py \
  --benchmark benchmark_342.json \
  --systems B D \
  --output results_sysB_sysD.json \
  --delay 3
```
