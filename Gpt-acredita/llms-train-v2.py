# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  train_acredita_llm.py — Entrena un LLM desde cero para acreditación   ║
║  Modelo: GPT-2 style (125M/350M/774M parámetros)                       ║
║  GPU: NVIDIA RTX 4090 (24GB VRAM)                                       ║
║  Dominio: Acreditación universitaria colombiana (CNA, EISC, Univalle)   ║
╚══════════════════════════════════════════════════════════════════════════╝
REQUISITOS:
pip install torch transformers datasets tokenizers accelerate wandb tqdm
pip install fitz pymupdf4llm
"""
import os
import sys
import json
import glob
import argparse
import time
import math
import re
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # ✅ Progreso visual

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL MODELO (CLAVES SIN ESPACIOS)
# ══════════════════════════════════════════════════════════════════════════
MODEL_CONFIGS = {
    "125m": {
        "n_layer": 12, "n_head": 12, "n_embd": 768, "vocab_size": 32000, "block_size": 1024,
        "params_approx": "125M", "batch_size": 16, "gradient_accumulation_steps": 4,
        "learning_rate": 6e-4, "vram_estimate_gb": 6,
    },
    "350m": {
        "n_layer": 24, "n_head": 16, "n_embd": 1024, "vocab_size": 32000, "block_size": 1024,
        "params_approx": "350M", "batch_size": 8, "gradient_accumulation_steps": 8,
        "learning_rate": 3e-4, "vram_estimate_gb": 12,
    },
    "774m": {
        "n_layer": 36, "n_head": 20, "n_embd": 1280, "vocab_size": 32000, "block_size": 1024,
        "params_approx": "774M", "batch_size": 4, "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4, "vram_estimate_gb": 20,
    },
}

# Directorios y archivos
CORPUS_TEXT_FILE = "corpus_acreditacion.txt"   # ← archivo Q/A directo, sin carpeta corpus/
TOKENIZER_DIR    = "tokenizer_acredita"
CHECKPOINT_DIR   = "checkpoints"
MODEL_DIR        = "modelo_final"

# Training Hyperparams
MAX_ITERS = 50000
EVAL_INTERVAL = 1000
SAVE_INTERVAL = 5000
WARMUP_ITERS = 2000
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
DTYPE = "bfloat16"

# ══════════════════════════════════════════════════════════════════════════
# PASO 1: PREPARAR CORPUS (con tqdm)
# ══════════════════════════════════════════════════════════════════════════
def prepare_corpus():
    print("\n" + "=" * 70)
    print("📚 PASO 1: Preparando corpus CHATBOT desde dataset_mejorado.json (71,864 pares RAFT)")
    print("=" * 70)

    DATASET_PATH = "dataset_final.json"   # ← dataset mejorado con RAFT + dedup

    if not os.path.exists(DATASET_PATH):
        tqdm.write(f"\n❌ No se encontró '{DATASET_PATH}'")
        tqdm.write("   Pon dataset_acreditacion.json en la misma carpeta y repite.")
        return

    import json, random
    tqdm.write(f"\n📂 Cargando {DATASET_PATH}...")
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    tqdm.write(f"   {len(data):,} pares cargados")

    random.seed(42)
    random.shuffle(data)

    # ── Formato ChatML para chatbot ────────────────────────────────────────
    # El modelo aprende a:
    #   1. Leer el sistema (rol del asistente)
    #   2. Leer la pregunta del usuario
    #   3. Generar la respuesta del asistente hasta <|eos|>
    # Al inferir basta dar <|system|>...<|user|>... y el modelo completa
    # <|assistant|>...<|eos|>
    SYSTEM = (
        "Eres ChatAcredita, un asistente experto en acreditación universitaria "
        "colombiana. Conoces el marco CNA, el Decreto 1330 de 2019, la Resolución "
        "21795 de 2020 y los procesos de autoevaluación de la EISC en la "
        "Universidad del Valle. Responde de forma clara, precisa y en español."
    )

    n_ok = n_skip = total_chars = 0

    with open(CORPUS_TEXT_FILE, "w", encoding="utf-8") as out:
        for item in tqdm(data, desc="📝 Convirtiendo", unit="par"):
            msgs = item.get("messages", [])
            p = next((m["content"] for m in msgs if m["role"] == "user"),      None)
            r = next((m["content"] for m in msgs if m["role"] == "assistant"), None)

            if not p or not r:
                p = item.get("pregunta") or item.get("question") or item.get("input")
                r = item.get("respuesta") or item.get("answer") or item.get("output")

            if not p or not r or len(p.strip()) < 10 or len(r.strip()) < 10:
                n_skip += 1
                continue

            # Formato ChatML — el modelo aprende este patrón completo
            bloque = (
                f"<|system|>\n{SYSTEM}\n<|endofturn|>\n"
                f"<|user|>\n{p.strip()}\n<|endofturn|>\n"
                f"<|assistant|>\n{r.strip()}\n<|eos|>\n\n"
            )
            out.write(bloque)
            total_chars += len(bloque)
            n_ok += 1

    tam_mb = os.path.getsize(CORPUS_TEXT_FILE) / 1024 / 1024
    tqdm.write(f"\n✅ Corpus CHATBOT guardado en: {CORPUS_TEXT_FILE}")
    tqdm.write(f"   📊 {n_ok:,} pares | {n_skip} saltados | {tam_mb:.1f} MB | ~{total_chars//4:,} tokens")
    tqdm.write(f"\n   Formato aprendido:")
    tqdm.write(f"   <|system|> rol del asistente <|endofturn|>")
    tqdm.write(f"   <|user|> pregunta <|endofturn|>")
    tqdm.write(f"   <|assistant|> RESPUESTA <|eos|>")


def clean_text(text: str) -> str:
    """Limpia texto extraído de PDFs"""
    text = re.sub(r"(Diseño|Rediseño) Curricular.*\n", " ", text)
    text = re.sub(r"SNIES.*\n", " ", text)
    text = re.sub(r"^\d+\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════
# PASO 2: ENTRENAR TOKENIZER
# ══════════════════════════════════════════════════════════════════════════
def train_tokenizer():
    print("\n" + "=" * 70)
    print("🔤 PASO 2: Entrenando tokenizer para acreditación")
    print("=" * 70)
    
    if not os.path.exists(CORPUS_TEXT_FILE):
        tqdm.write(f"❌ No existe {CORPUS_TEXT_FILE}. Corre primero --step prepare")
        return

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        min_frequency=2,
        special_tokens=[
            "<|pad|>", "<|eos|>", "<|bos|>", "<|unk|>",
            "<|system|>", "<|user|>", "<|assistant|>", "<|endofturn|>",
        ],
        show_progress=True,
    )

    tqdm.write(f"   Entrenando con {CORPUS_TEXT_FILE}...")
    tokenizer.train([CORPUS_TEXT_FILE], trainer)

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save(os.path.join(TOKENIZER_DIR, "tokenizer.json"))

    # Test
    test_text = "El programa de Tecnología en Desarrollo de Software de la EISC"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    
    tqdm.write(f"✅ Tokenizer guardado. Vocab: {tokenizer.get_vocab_size():,}")
    tqdm.write(f"   Test: '{test_text}' -> '{decoded}'")


# ══════════════════════════════════════════════════════════════════════════
# MODELO GPT (Transformer Decoder-Only)
# ══════════════════════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0
        self.c_attn = nn.Linear(config["n_embd"], 3 * config["n_embd"])
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"])
        self.n_head, self.n_embd = config["n_head"], config["n_embd"]
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["n_embd"], 4 * config["n_embd"])
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config["n_embd"], config["n_embd"])
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["n_embd"])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPTAcredita(nn.Module):
    """GPT-2 style decoder-only transformer para acreditación"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config["vocab_size"], config["n_embd"]),
            "wpe": nn.Embedding(config["block_size"], config["n_embd"]),
            "h": nn.ModuleList([Block(config) for _ in range(config["n_layer"])]),
            "ln_f": nn.LayerNorm(config["n_embd"]),
        })
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        self.transformer["wte"].weight = self.lm_head.weight  # Weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 * (2 * self.config["n_layer"]) ** -0.5 if hasattr(module, "SCALE_INIT") else 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer["wte"](idx) + self.transformer["wpe"](pos)
        for block in self.transformer["h"]:
            x = block(x)
        logits = self.lm_head(self.transformer["ln_f"](x))
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.3, top_k=50,
                 repetition_penalty=2.0):
        """Genera texto token por token con penalizacion de repeticion."""
        eos_id = self.config.get("eos_token_id", None)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_id is not None and idx_next.item() == eos_id:
                break
        return idx


# ══════════════════════════════════════════════════════════════════════════
# DATASET CON TQDM Y MEMMAP
# ══════════════════════════════════════════════════════════════════════════
class AcreditaDataset(Dataset):
    """Dataset que tokeniza por chunks y guarda en disco con memmap"""
    def __init__(self, text_file, tokenizer_path, block_size=1024):
        from tokenizers import Tokenizer
        self.block_size = block_size
        self.tokens_file = text_file + ".tokens.npy"

        if os.path.exists(self.tokens_file):
            tqdm.write(f"   ♻️ Cargando tokens pre-procesados: {self.tokens_file}")
            self.tokens = np.memmap(self.tokens_file, dtype=np.int32, mode="r")
            return

        tokenizer = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))
        file_size = os.path.getsize(text_file)
        chunk_size = 10 * 1024 * 1024  # 10MB por chunk

        # 1️⃣ BARRA: Conteo de tokens
        tqdm.write("   📊 Contando tokens...")
        total_tokens = 0
        with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
            with tqdm(total=file_size, desc="   📊 Lectura/Conteo", unit="B", unit_scale=True) as pbar:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    total_tokens += len(tokenizer.encode(chunk).ids)
                    pbar.update(len(chunk))
        tqdm.write(f"   Total tokens: {total_tokens:,}")

        # 2️⃣ BARRA: Tokenización a disco
        tqdm.write("   💾 Guardando tokens a memmap...")
        tokens_mmap = np.memmap(self.tokens_file, dtype=np.int32, mode="w+", shape=(total_tokens,))
        offset = 0
        with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
            with tqdm(total=file_size, desc="   💾 Tokenizando", unit="B", unit_scale=True) as pbar:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    ids = tokenizer.encode(chunk).ids
                    end = min(offset + len(ids), total_tokens)
                    tokens_mmap[offset:end] = ids[:end - offset]
                    offset = end
                    pbar.update(len(chunk))
        tokens_mmap.flush()
        del tokens_mmap
        self.tokens = np.memmap(self.tokens_file, dtype=np.int32, mode="r")
        tqdm.write(f"   ✅ Tokens listos: {len(self.tokens):,} | {os.path.getsize(self.tokens_file)/1024**3:.2f} GB")

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = torch.from_numpy(self.tokens[start:start + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


# ══════════════════════════════════════════════════════════════════════════
# PASO 3: ENTRENAR CON TQDM + FIX WINDOWS/TRITON
# ══════════════════════════════════════════════════════════════════════════
def train(model_size="350m"):
    print("\n" + "=" * 70)
    print(f"🚀 PASO 3: Entrenando GPT-Acredita ({model_size.upper()})")
    print("=" * 70)
    
    config = MODEL_CONFIGS[model_size]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm.write(f"   🖥️ Device: {device} | 📐 Params: {config['params_approx']}")

    # Dataset
    dataset = AcreditaDataset(CORPUS_TEXT_FILE, TOKENIZER_DIR, config["block_size"])
    train_size = int(0.95 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Modelo
    tqdm.write("\n🧠 Creando modelo...")
    model = GPTAcredita(config).to(device)
    n_params = model.count_parameters()
    tqdm.write(f"   Parámetros: {n_params:,} ({n_params / 1e6:.1f}M)")

    # 🔧 torch.compile() con detección segura de Triton (Windows-compatible)
    USE_COMPILE = (
        sys.platform != "win32" and
        hasattr(torch, "compile") and
        os.environ.get("DISABLE_COMPILE", "0") != "1"
    )
    if USE_COMPILE:
        try:
            import triton  # Verificar que Triton esté disponible
            tqdm.write("   ⚡ Compilando modelo con torch.compile() + Triton...")
            model = torch.compile(model)
        except ImportError:
            tqdm.write("   ⚠️ torch.compile() omitido: Triton no disponible")
    else:
        tqdm.write("   ℹ️ torch.compile() desactivado (Windows o DISABLE_COMPILE=1)")

    # Optimizer (AdamW con weight decay)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0}
    ], lr=config["learning_rate"], betas=(0.9, 0.95), fused=True)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Learning rate scheduler (cosine con warmup)
    def get_lr(it):
        if it < WARMUP_ITERS:
            return config["learning_rate"] * (it + 1) / WARMUP_ITERS
        decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
        return config["learning_rate"] * 0.1 + 0.5 * (1 + math.cos(math.pi * decay_ratio)) * (config["learning_rate"] * 0.9)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    grad_accum = config["gradient_accumulation_steps"]
    best_val_loss = float("inf")
    start_time = time.time()
    train_iter = iter(train_loader)

    # ⬅️ BARRA PRINCIPAL DE ENTRENAMIENTO
    with tqdm(total=MAX_ITERS, desc="🏋️ Entrenando", unit="step", leave=True) as pbar:
        for step in range(MAX_ITERS):
            # Update learning rate
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation
            optimizer.zero_grad()
            total_loss = 0.0

            for _ in range(grad_accum):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                with autocast_ctx:
                    _, loss = model(x, y)
                scaler.scale(loss / grad_accum).backward()
                total_loss += loss.item()

            # Gradient clipping y optimización
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            # Logging cada 100 steps
            if step % 100 == 0:
                elapsed = time.time() - start_time
                tok_sec = (step * config["batch_size"] * grad_accum * config["block_size"]) / max(elapsed, 1)
                eta = (MAX_ITERS - step) / max(step / max(elapsed, 1), 1e-9) / 3600
                mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
                pbar.set_postfix(
                    loss=f"{total_loss:.4f}",
                    lr=f"{lr:.2e}",
                    tok_s=f"{tok_sec:,.0f}",
                    ETA=f"{eta:.1f}h",
                    VRAM=f"{mem:.1f}GB"
                )
                tqdm.write(f"  Step {step:6d}/{MAX_ITERS} | Loss: {total_loss:.4f} | LR: {lr:.2e} | Tok/s: {tok_sec:,.0f} | VRAM: {mem:.1f}GB")

            # Evaluación periódica
            if step > 0 and step % EVAL_INTERVAL == 0:
                model.eval()
                val_loss = val_steps = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        with autocast_ctx:
                            _, vloss = model(vx, vy)
                        val_loss += vloss.item()
                        val_steps += 1
                        if val_steps >= 50:
                            break
                val_loss /= max(val_steps, 1)
                ppl = math.exp(min(val_loss, 20))
                tqdm.write(f"  📊 EVAL {step}: Val Loss={val_loss:.4f} | PPL={ppl:.1f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "step": step,
                        "val_loss": val_loss,
                    }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
                    tqdm.write("  💾 Mejor modelo guardado")
                model.train()

            # Checkpoint periódico
            if step > 0 and step % SAVE_INTERVAL == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "step": step,
                }, os.path.join(CHECKPOINT_DIR, f"step_{step}.pt"))
                tqdm.write(f"  📦 Checkpoint step_{step}.pt")

            pbar.update(1)

    # Guardar modelo final
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_path = os.path.join(MODEL_DIR, "gpt_acredita.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step,
        "val_loss": best_val_loss,
    }, final_path)
    
    total_time = (time.time() - start_time) / 3600
    tqdm.write(f"\n✅ ENTRENAMIENTO FINALIZADO en {MODEL_DIR}/gpt_acredita.pt")
    tqdm.write(f"   Mejor Val Loss: {best_val_loss:.4f} | Tiempo: {total_time:.1f}h")


# ══════════════════════════════════════════════════════════════════════════
# PASO 4: PROBAR EL MODELO
# ══════════════════════════════════════════════════════════════════════════
def test_model():
    print("\n" + "=" * 70)
    print("🧪 PASO 4: Probando GPT-Acredita CHATBOT")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "gpt_acredita.pt")
    if not os.path.exists(model_path):
        tqdm.write("❌ No se encontró modelo entrenado")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config     = checkpoint["config"]
    model      = GPTAcredita(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tqdm.write(f"   Modelo cargado: {config['params_approx']}")
    tqdm.write(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(TOKENIZER_DIR, "tokenizer.json"))

    EOS_ID = tokenizer.token_to_id("<|eos|>")

    SYSTEM = (
        "Eres ChatAcredita, un asistente experto en acreditación universitaria "
        "colombiana. Conoces el marco CNA, el Decreto 1330 de 2019, la Resolución "
        "21795 de 2020 y los procesos de autoevaluación de la EISC."
    )

    def construir_prompt(pregunta, historial=None):
        """Construye el prompt en formato ChatML con historial opcional."""
        prompt = f"<|system|>\n{SYSTEM}\n<|endofturn|>\n"
        if historial:
            for u, a in historial[-3:]:   # máx 3 turnos de contexto
                prompt += f"<|user|>\n{u}\n<|endofturn|>\n"
                prompt += f"<|assistant|>\n{a}\n<|eos|>\n"
        prompt += f"<|user|>\n{pregunta}\n<|endofturn|>\n<|assistant|>\n"
        return prompt

    def responder(pregunta, historial=None, max_tokens=200, temp=0.7):
        prompt    = construir_prompt(pregunta, historial)
        encoded   = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_k=40,
                repetition_penalty=1.5,
            )
        generated = tokenizer.decode(output_ids[0].tolist())
        if "<|assistant|>" in generated:
            resp = generated.split("<|assistant|>")[-1]
            resp = resp.split("<|eos|>")[0].strip()
            return resp
        return generated[len(prompt):].strip()

    # Preguntas de prueba
    preguntas_test = [
        "¿Cuáles son los 10 factores del CNA?",
        "¿Qué establece el Decreto 1330 de 2019?",
        "¿Qué diferencia hay entre registro calificado y acreditación?",
        "¿Qué es el programa TEDESOFT de la EISC?",
        "¿Qué evalúa el Factor 6 de Investigación e Innovación?",
    ]

    tqdm.write("\n" + "=" * 70)
    tqdm.write("🧪 Respuestas de prueba:")
    tqdm.write("=" * 70)
    for p in preguntas_test:
        resp = responder(p)
        tqdm.write(f"\n❓ {p}")
        tqdm.write(f"🎓 {resp[:300]}")
        tqdm.write("─" * 60)

    # Modo interactivo con historial
    tqdm.write("\n" + "=" * 70)
    tqdm.write("💬 Chatbot interactivo — escribe 'salir' para terminar")
    tqdm.write("   El modelo recuerda los últimos 3 turnos de conversación")
    tqdm.write("=" * 70 + "\n")

    historial = []
    while True:
        try:
            pregunta = input("Tú > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        if not pregunta:
            continue
        resp = responder(pregunta, historial)
        tqdm.write(f"🎓 {resp}\n")
        historial.append((pregunta, resp))
        historial = historial[-3:]   # mantener solo últimos 3 turnos


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena GPT-Acredita desde cero")
    parser.add_argument(
        "--step", choices=["prepare", "tokenizer", "train", "test", "all"],
        required=True, help="Qué paso ejecutar"
    )
    parser.add_argument(
        "--size", choices=["125m", "350m", "774m"], default="350m",
        help="Tamaño del modelo (default: 350m)"
    )
    args = parser.parse_args()

    if args.step in ["prepare", "all"]:
        prepare_corpus()
    if args.step in ["tokenizer", "all"]:
        train_tokenizer()
    if args.step in ["train", "all"]:
        train(args.size)
    if args.step in ["test", "all"]:
        test_model()