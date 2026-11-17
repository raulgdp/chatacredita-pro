---
language:
- es
license: mit
tags:
- gpt2
- causal-lm
- accreditation
- colombian-education
- cna
- domain-specific
- pretraining
datasets:
- custom (CNA regulatory corpus, EISC/Univalle, 10 GB)
metrics:
- perplexity
base_model: null
pipeline_tag: text-generation
---

# Gpt-acredita-350M

A GPT-2-style decoder-only language model (350 M parameters) pretrained from scratch on the Colombian university accreditation corpus of **Universidad del Valle's Escuela de Ingeniería de Sistemas y Computación (EISC)**. It serves as the domain-adapted base model and synthetic dataset generator for the ChatAcredita PRO pipeline.

---

## Model Description

| Property | Value |
|----------|-------|
| Architecture | GPT-2 decoder-only transformer |
| Parameters | ~350 M |
| Layers | 24 |
| Attention heads | 16 |
| Embedding dim | 1 024 |
| Context window | 1 024 tokens |
| Vocabulary | 32 000 (BPE, domain-trained) |
| Training hardware | NVIDIA RTX 4090 (24 GB VRAM) |
| Training time | ~38 hours |
| Training tokens | ~200 M |
| Precision | bfloat16 |

The model uses **weight tying** between the token embedding and the language modeling head, and a custom ChatML prompt format with the special tokens `<|system|>`, `<|user|>`, `<|assistant|>`, `<|endofturn|>` and `<|eos|>`.

---

## Training Data

The pretraining corpus consists of 10 GB of CNA (Consejo Nacional de Acreditación) regulatory documents:

- CNA accreditation guidelines (all nine factors)
- Decreto 1330 de 2019 (quality conditions for higher education programs)
- Resolución 21795 de 2020
- EISC self-study annexes and curricular redesign reports
- Universidad del Valle institutional documents

The corpus was converted to ChatML format with 71 864 Q/A pairs from an earlier RAFT-generated dataset:

```
<|system|>
Eres ChatAcredita, un asistente experto en acreditación universitaria colombiana...
<|endofturn|>
<|user|>
{pregunta}
<|endofturn|>
<|assistant|>
{respuesta}
<|eos|>
```

---

## Training Procedure

Training used the script `llms-train-v2.py` with the following configuration:

```
Model size    : 350m
Max iters     : 50 000
Batch size    : 8
Grad accum    : 8  (effective batch = 64 sequences)
Learning rate : 3e-4 (cosine decay with 2 000-step warmup)
Weight decay  : 0.1
Grad clip     : 1.0
Optimizer     : AdamW (β₁=0.9, β₂=0.95, fused=True)
Precision     : bfloat16 AMP
Compile       : torch.compile() + Triton (Linux only)
Checkpoints   : every 5 000 steps; best by validation loss
```

The dataset was split 95 % train / 5 % validation. Tokens were stored as a memory-mapped NumPy array for efficient I/O during training.

---

## Downstream Use: ChatAcredita PRO Pipeline

This model is designed to be used in Stage 2 of the ChatAcredita PRO pipeline — as a draft generator for the synthetic Q/A dataset:

```
Stage 1  Gpt-acredita-350M pretraining (this model)
           └─► used as domain vocabulary + generation seed

Stage 2  generar_dataset_mejorado.py
           └─► Groq/LLaMA-3.3-70B + RAFT → 15 000 Q/A pairs

Stage 3  RAG-VSD filtering (τ_o=0.15, τ_q=0.52, τ_t=0.26)
           └─► 9 847 verified pairs

Stage 4  QLoRA fine-tuning on Qwen3-14B
           └─► ChatAcredita PRO (RAGAS faithfulness 0.799)
```

---

## Evaluation

Evaluated on a held-out 5 % split of the pretraining corpus:

| Metric | Value |
|--------|-------|
| Validation loss (best checkpoint) | ~2.1 |
| Perplexity (approx.) | ~8.2 |

Generation quality was assessed qualitatively on five accreditation questions covering CNA factors 1–9. The model demonstrates adequate domain vocabulary and can generate plausible regulatory language, though it may hallucinate specific figures (percentages, article numbers) not seen in context.

---

## How to Use

The model is saved in PyTorch `.pt` format with a custom architecture (not HuggingFace Transformers compatible out of the box). To load and generate:

```python
import torch
from llms_train_v2 import GPTAcredita  # the training script's model class

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)

config = checkpoint["config"]
model  = GPTAcredita(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Generation with ChatML prompt:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer_acredita/tokenizer.json")

SYSTEM = (
    "Eres ChatAcredita, un asistente experto en acreditación universitaria "
    "colombiana. Conoces el marco CNA, el Decreto 1330 de 2019, la Resolución "
    "21795 de 2020 y los procesos de autoevaluación de la EISC en la "
    "Universidad del Valle. Responde de forma clara, precisa y en español."
)

pregunta = "¿Cuáles son los 10 factores del CNA?"
prompt = (
    f"<|system|>\n{SYSTEM}\n<|endofturn|>\n"
    f"<|user|>\n{pregunta}\n<|endofturn|>\n"
    f"<|assistant|>\n"
)

input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=200,
        temperature=0.7,
        top_k=40,
        repetition_penalty=1.5,
    )

generated = tokenizer.decode(output_ids[0].tolist())
response  = generated.split("<|assistant|>")[-1].split("<|eos|>")[0].strip()
print(response)
```

---

## Limitations

- **Domain-specific**: trained exclusively on CNA/EISC documents; performs poorly outside Colombian accreditation context.
- **Hallucination risk**: may confabulate specific article numbers, dates, or percentages not in the retrieved context. Always use with RAG.
- **Language**: Spanish only; English queries will produce degraded output.
- **Context window**: 1 024 tokens; long regulatory documents must be chunked before use.
- **Not RLHF-tuned**: the base model has no safety alignment. It is intended as a component in the ChatAcredita PRO pipeline, not as a standalone assistant.

---

## Repository

- **Training script**: [`llms-train-v2.py`](https://github.com/raulgdp/chatacredita-pro)
- **Dataset generation**: [`generar_dataset_mejorado.py`](https://github.com/raulgdp/chatacredita-pro)
- **Full pipeline**: [raulgdp/chatacredita-pro](https://github.com/raulgdp/chatacredita-pro)

---

## Citation

If you use this model, please cite the associated paper:

```bibtex
@article{chatacredita2025,
  title   = {RAG-Verified Self-Distillation for Domain-Specific LLMs:
             A Bootstrap Pipeline for Colombian University Accreditation},
  author  = {Gutiérrez, Raúl and {Universidad del Valle, EISC}},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2025}
}
```

---

## License

MIT License — see [LICENSE](https://github.com/raulgdp/chatacredita-pro/blob/main/LICENSE) for details.

---

*Developed at the Escuela de Ingeniería de Sistemas y Computación (EISC), Universidad del Valle, Cali, Colombia.*
