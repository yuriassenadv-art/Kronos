#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# vast_train.sh — Kronos fine-tune pipeline para vast.ai (CUDA)
#
# Uso (na instância vast.ai após clonar o repo):
#   bash vast_train.sh
#
# Com múltiplas GPUs (ex: 4x RTX 3090), treina os 4 pares em PARALELO,
# um símbolo por GPU — reduz tempo de ~4h para ~1h.
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
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "  GPUs disponíveis: $GPU_COUNT"
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} — {p.total_memory/1e9:.1f} GB')
" 2>/dev/null || echo "  (CUDA não disponível)"

# ── 1. Dependencies ───────────────────────────────────────────────────────────
echo ""
echo "▸ Verificando dependências…"

MISSING=""
python3 -c "import torch" 2>/dev/null          || MISSING="$MISSING torch torchvision"
python3 -c "import ccxt" 2>/dev/null           || MISSING="$MISSING ccxt"
python3 -c "import pandas" 2>/dev/null         || MISSING="$MISSING pandas"
python3 -c "import yaml" 2>/dev/null           || MISSING="$MISSING pyyaml"
python3 -c "import huggingface_hub" 2>/dev/null || MISSING="$MISSING huggingface_hub transformers"

if [ -n "$MISSING" ]; then
    echo "  Instalando:$MISSING"
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" | cut -d. -f1,2 || echo "12.1")
    CUDA_TAG=$(echo "$CUDA_VER" | tr -d '.')
    pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/cu${CUDA_TAG}" 2>/dev/null \
        || pip install -q torch torchvision
    pip install -q ccxt pandas pyyaml huggingface_hub transformers numpy requests
    echo "  Dependências instaladas."
else
    echo "  Todas as dependências presentes."
fi

# ── 2. Download Binance data (sequencial — compartilha rede) ──────────────────
echo ""
echo "▸ Baixando dados Binance $INTERVAL ($DAYS dias)…"

for SYMBOL in "${SYMBOLS[@]}"; do
    CSV="$REPO_DIR/finetune_csv/data/BN_${SYMBOL}_${INTERVAL}.csv"
    if [ -f "$CSV" ]; then
        echo "  $SYMBOL: já existe ($(wc -l < "$CSV") linhas)"
    else
        echo "  $SYMBOL: baixando…"
        PYTHONPATH="$REPO_DIR" python3 -m infra.binance_data_pipeline \
            --symbols "$SYMBOL" --interval "$INTERVAL" --days "$DAYS" \
            2>&1 | tail -3
    fi
done

# ── 3. Train — paralelo se múltiplas GPUs, sequencial se 1 GPU ───────────────
echo ""
TOTAL_START=$(date +%s)
PIDS=()
FAILED=()

if [ "$GPU_COUNT" -ge "${#SYMBOLS[@]}" ]; then
    # ── Modo paralelo: 1 símbolo por GPU ─────────────────────────────────────
    echo "▸ Modo PARALELO — ${#SYMBOLS[@]} símbolos em ${#SYMBOLS[@]} GPUs simultâneas"
    echo ""

    for i in "${!SYMBOLS[@]}"; do
        SYMBOL="${SYMBOLS[$i]}"
        GPU_ID=$i
        SYMBOL_LOWER=$(echo "$SYMBOL" | tr '[:upper:]' '[:lower:]')
        CONFIG="$REPO_DIR/finetune_csv/configs/config_bn_${SYMBOL_LOWER}_${INTERVAL}.yaml"
        LOG="$LOG_DIR/train_${SYMBOL}_${INTERVAL}.log"

        if [ ! -f "$CONFIG" ]; then
            echo "  ✗ $SYMBOL: config não encontrada — pulando"
            continue
        fi

        echo "  Iniciando $SYMBOL na GPU $GPU_ID → $LOG"
        CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="$REPO_DIR" \
            python3 finetune_csv/train_sequential.py --config "$CONFIG" \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done

    echo ""
    echo "  Aguardando conclusão de ${#PIDS[@]} jobs paralelos…"
    for i in "${!PIDS[@]}"; do
        PID="${PIDS[$i]}"
        SYMBOL="${SYMBOLS[$i]}"
        if wait "$PID"; then
            echo "  ✓ $SYMBOL concluído"
        else
            echo "  ✗ $SYMBOL FALHOU — ver $LOG_DIR/train_${SYMBOL}_${INTERVAL}.log"
            FAILED+=("$SYMBOL")
        fi
    done

else
    # ── Modo sequencial: 1 GPU para todos ────────────────────────────────────
    echo "▸ Modo SEQUENCIAL — ${#SYMBOLS[@]} símbolos na GPU 0"
    echo ""

    for SYMBOL in "${SYMBOLS[@]}"; do
        SYMBOL_LOWER=$(echo "$SYMBOL" | tr '[:upper:]' '[:lower:]')
        CONFIG="$REPO_DIR/finetune_csv/configs/config_bn_${SYMBOL_LOWER}_${INTERVAL}.yaml"
        LOG="$LOG_DIR/train_${SYMBOL}_${INTERVAL}.log"

        if [ ! -f "$CONFIG" ]; then
            echo "  ✗ $SYMBOL: config não encontrada — pulando"
            FAILED+=("$SYMBOL")
            continue
        fi

        echo "══════════════════════════════════════════════════════════"
        echo "  Treinando $SYMBOL/$INTERVAL → $LOG"
        SYMBOL_START=$(date +%s)

        if PYTHONPATH="$REPO_DIR" python3 finetune_csv/train_sequential.py \
                --config "$CONFIG" 2>&1 | tee "$LOG"; then
            ELAPSED=$(( ($(date +%s) - SYMBOL_START) / 60 ))
            echo "  ✓ $SYMBOL concluído em ${ELAPSED} min"
        else
            echo "  ✗ $SYMBOL FALHOU"
            FAILED+=("$SYMBOL")
        fi
    done
fi

# ── 4. Summary ────────────────────────────────────────────────────────────────
TOTAL_MIN=$(( ($(date +%s) - TOTAL_START) / 60 ))

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  TREINAMENTO CONCLUÍDO em ${TOTAL_MIN} min"
echo "══════════════════════════════════════════════════════════"
for SYMBOL in "${SYMBOLS[@]}"; do
    DIR="$REPO_DIR/finetune_csv/finetuned/BN_${SYMBOL}_${INTERVAL}"
    [ -d "$DIR" ] && echo "  ✓ $SYMBOL → $DIR/"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Falhas: ${FAILED[*]}"
    exit 1
fi
echo "══════════════════════════════════════════════════════════"
