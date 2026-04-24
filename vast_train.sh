#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# vast_train.sh — Kronos fine-tune pipeline para vast.ai (CUDA)
#
# Uso (na instância vast.ai após clonar o repo):
#   bash vast_train.sh
#
# O script:
#   1. Instala dependências Python (PyTorch CUDA)
#   2. Baixa dados Binance 15m para BTC, SOL, ETH, DOGE (730 dias)
#   3. Treina Kronos sequencialmente para cada par
#   4. Salva modelos em finetune_csv/finetuned/
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SYMBOLS=("BTC" "SOL" "ETH" "DOGE")
INTERVAL="15m"
DAYS=730
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

echo "══════════════════════════════════════════════════════════"
echo "  Kronos vast.ai Training — ${SYMBOLS[*]} / $INTERVAL"
echo "══════════════════════════════════════════════════════════"

# ── GPU check ─────────────────────────────────────────────────────────────────
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: CUDA not available, using CPU')
"

# ── 1. Dependencies ───────────────────────────────────────────────────────────
echo ""
echo "▸ Verificando dependências…"

MISSING=""
python3 -c "import torch" 2>/dev/null     || MISSING="$MISSING torch torchvision"
python3 -c "import ccxt" 2>/dev/null      || MISSING="$MISSING ccxt"
python3 -c "import pandas" 2>/dev/null    || MISSING="$MISSING pandas"
python3 -c "import yaml" 2>/dev/null      || MISSING="$MISSING pyyaml"
python3 -c "import huggingface_hub" 2>/dev/null || MISSING="$MISSING huggingface_hub transformers"

if [ -n "$MISSING" ]; then
    echo "  Instalando:$MISSING"
    # Detecta versão CUDA e instala PyTorch compatível
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" | cut -d. -f1,2 || echo "12.1")
    CUDA_TAG=$(echo "$CUDA_VER" | tr -d '.')
    pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/cu${CUDA_TAG}" 2>/dev/null \
        || pip install -q torch torchvision  # fallback sem especificar CUDA
    pip install -q ccxt pandas pyyaml huggingface_hub transformers numpy requests
    echo "  Dependências instaladas."
else
    echo "  Todas as dependências presentes."
fi

# ── 2. Download Binance data ──────────────────────────────────────────────────
echo ""
echo "▸ Baixando dados Binance $INTERVAL ($DAYS dias)…"

for SYMBOL in "${SYMBOLS[@]}"; do
    CSV="$REPO_DIR/finetune_csv/data/BN_${SYMBOL}_${INTERVAL}.csv"
    if [ -f "$CSV" ]; then
        LINES=$(wc -l < "$CSV")
        echo "  $SYMBOL: já existe ($LINES linhas)"
    else
        echo "  $SYMBOL: baixando…"
        PYTHONPATH="$REPO_DIR" python3 -m infra.binance_data_pipeline \
            --symbols "$SYMBOL" --interval "$INTERVAL" --days "$DAYS" \
            2>&1 | tail -5
    fi
done

# ── 3. Train each symbol ──────────────────────────────────────────────────────
echo ""
echo "▸ Iniciando treinamento sequencial…"
echo "  Ordem: ${SYMBOLS[*]}"
echo ""

TOTAL_START=$(date +%s)
FAILED=()

for SYMBOL in "${SYMBOLS[@]}"; do
    SYMBOL_LOWER=$(echo "$SYMBOL" | tr '[:upper:]' '[:lower:]')
    CONFIG="$REPO_DIR/finetune_csv/configs/config_bn_${SYMBOL_LOWER}_${INTERVAL}.yaml"
    LOG="$LOG_DIR/train_${SYMBOL}_${INTERVAL}.log"

    if [ ! -f "$CONFIG" ]; then
        echo "  ✗ $SYMBOL: config não encontrada ($CONFIG) — pulando"
        FAILED+=("$SYMBOL")
        continue
    fi

    echo "══════════════════════════════════════════════════════════"
    echo "  Treinando $SYMBOL/$INTERVAL"
    echo "  Config : $CONFIG"
    echo "  Log    : $LOG"
    echo "══════════════════════════════════════════════════════════"

    SYMBOL_START=$(date +%s)

    if PYTHONPATH="$REPO_DIR" python3 finetune_csv/train_sequential.py \
            --config "$CONFIG" 2>&1 | tee "$LOG"; then
        SYMBOL_END=$(date +%s)
        ELAPSED=$(( (SYMBOL_END - SYMBOL_START) / 60 ))
        echo ""
        echo "  ✓ $SYMBOL concluído em ${ELAPSED} min"
        echo "    Modelo: finetune_csv/finetuned/BN_${SYMBOL}_${INTERVAL}/"
    else
        echo "  ✗ $SYMBOL FALHOU — ver $LOG"
        FAILED+=("$SYMBOL")
    fi
    echo ""
done

# ── 4. Summary ────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_MIN=$(( (TOTAL_END - TOTAL_START) / 60 ))

echo "══════════════════════════════════════════════════════════"
echo "  TREINAMENTO CONCLUÍDO"
echo "══════════════════════════════════════════════════════════"
echo "  Tempo total: ${TOTAL_MIN} min"
echo ""
echo "  Modelos salvos:"
for SYMBOL in "${SYMBOLS[@]}"; do
    SYMBOL_LOWER=$(echo "$SYMBOL" | tr '[:upper:]' '[:lower:]')
    DIR="$REPO_DIR/finetune_csv/finetuned/BN_${SYMBOL}_${INTERVAL}"
    if [ -d "$DIR" ]; then
        echo "  ✓ $SYMBOL → $DIR/"
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  Falhas: ${FAILED[*]}"
    exit 1
fi
echo "══════════════════════════════════════════════════════════"
