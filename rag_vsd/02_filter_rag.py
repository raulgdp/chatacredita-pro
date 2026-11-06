#!/usr/bin/env python3
# 02_filter_rag.py — Filtrado automático con RAG (final corregido)
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from utils.qdrant_client import RAGVerifier

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN ANTI-TIMEOUT Y RATE LIMITING
# ══════════════════════════════════════════════════════════════════════════
CHECKPOINT_FILE = DATA_DIR / "synthetic_verified" / "checkpoint.json"
DELAY_BETWEEN_QUERIES = 2.0       # segundos entre queries
MAX_RETRIES = 5
RETRY_DELAYS = [5, 10, 20, 40, 60]
CHECKPOINT_EVERY = 100
QDRANT_TIMEOUT = 120              # 2 minutos de timeout para Qdrant


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"🔄 Reanudando desde item {data['processed']} | {data['timestamp']}")
        return data
    return {"processed": 0, "verified": [], "rejected": []}


def save_checkpoint(processed, verified, rejected):
    CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "processed": processed,
            "verified": verified,
            "rejected": rejected,
            "timestamp": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)


def verify_with_retry(verifier, item, item_index):
    for attempt in range(MAX_RETRIES):
        try:
            result = verifier.verify(item["input"], item["output"])
            return result
        except Exception as e:
            error_msg = str(e)
            if any(keyword in error_msg.lower() for keyword in 
                   ["timeout", "read operation", "connection", "reset", "refused"]):
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS)-1)]
                print(f"\n   ⚠️ Item {item_index} | Intento {attempt+1}/{MAX_RETRIES}")
                print(f"      Error: {error_msg[:100]}")
                print(f"      ⏳ Esperando {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n   ❌ Item {item_index} | Error no recuperable: {error_msg[:100]}")
                break
    
    return {
        "supported": False,
        "overlap_score": 0.0,
        "qdrant_avg_score": 0.0,
        "chunks_found": 0,
        "tech_ratio": 0.0,
        "error": f"Failed after {MAX_RETRIES} retries: {error_msg[:200]}",
    }


def main():
    print("=" * 70)
    print("🔍 02_filter_rag.py — Filtrado automático con RAG")
    print("=" * 70)
    
    # ══════════════════════════════════════════════════════════════════════
    # INICIALIZAR VERIFICADOR CON TIMEOUT
    # ══════════════════════════════════════════════════════════════════════
    verifier = RAGVerifier(
        QDRANT_URL, 
        QDRANT_API_KEY, 
        QDRANT_COLLECTION,
        timeout=QDRANT_TIMEOUT,  # ← AHORA FUNCIONA
    )
    
    input_file = DATA_DIR / "synthetic_raw" / "synthetic_15k.jsonl"
    output_file = DATA_DIR / "synthetic_verified" / "verified.jsonl"
    rejected_file = DATA_DIR / "synthetic_verified" / "rejected.jsonl"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    total_lines = len(lines)
    print(f"📄 Total de items a verificar: {total_lines}")
    
    # Checkpoint
    checkpoint = load_checkpoint()
    start_idx = checkpoint["processed"]
    
    if start_idx > 0:
        verified = checkpoint.get("verified", [])
        rejected = checkpoint.get("rejected", [])
        with open(output_file, "w", encoding="utf-8") as f:
            for v in verified:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        with open(rejected_file, "w", encoding="utf-8") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"   ✅ {len(verified)} verificados y {len(rejected)} rechazados recuperados")
    else:
        verified = []
        rejected = []
        output_file.write_text("") if output_file.exists() else None
        rejected_file.write_text("") if rejected_file.exists() else None
    
    start_time = time.time()
    
    # Loop principal
    for i in tqdm(range(start_idx, total_lines), initial=start_idx, total=total_lines, 
                  desc="Verificando"):
        
        item = json.loads(lines[i])
        result = verify_with_retry(verifier, item, i)
        item["verification"] = result
        
        if result.get("supported", False):
            verified.append(item)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            rejected.append(item)
            with open(rejected_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # Rate limiting
        time.sleep(DELAY_BETWEEN_QUERIES)
        
        # Checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(i + 1, verified, rejected)
            elapsed = (time.time() - start_time) / 60
            rate = (i + 1 - start_idx) / max(elapsed, 0.1)
            remaining = total_lines - (i + 1)
            eta_min = remaining / max(rate, 0.1)
            print(f"\n💾 Checkpoint {i+1}/{total_lines} | "
                  f"Rate: {rate:.1f} items/min | ETA: {eta_min:.0f} min")
    
    save_checkpoint(total_lines, verified, rejected)
    
    # Estadísticas finales
    total = len(verified) + len(rejected)
    ratio = len(verified) / total if total > 0 else 0
    
    print(f"\n{'='*70}")
    print("📊 RESULTADOS DEL FILTRADO")
    print(f"{'='*70}")
    print(f"   Total evaluados: {total}")
    print(f"   ✅ Verificados: {len(verified)} ({ratio:.1%})")
    print(f"   ❌ Rechazados: {len(rejected)} ({1-ratio:.1%})")
    
    if verified:
        avg_overlap = sum(v["verification"].get("overlap_score", 0) for v in verified) / len(verified)
        avg_qdrant = sum(v["verification"].get("qdrant_avg_score", 0) for v in verified) / len(verified)
        print(f"\n   Métricas promedio de verificados:")
        print(f"   - Overlap léxico: {avg_overlap:.3f}")
        print(f"   - Score Qdrant: {avg_qdrant:.3f}")
    
    errors = [r for r in rejected if "error" in r.get("verification", {})]
    if errors:
        print(f"\n   ⚠️ Items con error de conexión: {len(errors)}")
    
    print(f"\n✅ Dataset verificado: {output_file}")
    print(f"\n   Siguiente paso: python 03_prepare_format.py")
    
    verifier.close()


if __name__ == "__main__":
    main()