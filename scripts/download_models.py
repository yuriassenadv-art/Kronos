"""
Download one-shot dos 3 modelos Kronos do Hugging Face Hub para disco local.

Este script roda 1 vez na VPS (após clone do repo) para baixar os pesos dos
modelos fine-tunados pela savycorp e armazená-los em disco local. Após esse
passo, o bot multi-asset (`trading.workflow`) NÃO precisa mais de internet
para o Hugging Face Hub — todas as inferências carregam do disco local.

Layout final em `models/`:
    models/
    ├── BTC/
    │   ├── tokenizer/best_model/{model.safetensors, config.json}
    │   └── basemodel/best_model/{model.safetensors, config.json}
    ├── ETH/
    │   └── (mesmo layout)
    └── SOL/
        └── (mesmo layout)

Uso na VPS:
    HF_TOKEN=hf_xxx python3 scripts/download_models.py

Tamanho total aproximado: ~333 MB (3 × ~111 MB).
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPOS = {
    "BTC": "savycorp/kronos-bn-btc-15m",
    "ETH": "savycorp/kronos-bn-eth-15m",
    "SOL": "savycorp/kronos-bn-sol-15m",
}
DEST_ROOT = Path(__file__).parent.parent / "models"


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERRO: defina HF_TOKEN (modelos privados).", file=sys.stderr)
        sys.exit(1)

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    for coin, repo in REPOS.items():
        dest = DEST_ROOT / coin
        print(f"[{coin}] baixando {repo} -> {dest}")
        snapshot_download(
            repo_id=repo,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=token,
        )
        size_mb = sum(
            f.stat().st_size for f in dest.rglob("*") if f.is_file()
        ) // 1_000_000
        print(f"[{coin}] OK ({size_mb} MB)")

    print("\nDone. Modelos em:", DEST_ROOT)


if __name__ == "__main__":
    main()
