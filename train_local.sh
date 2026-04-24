#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_local.sh — Kronos fine-tune pipeline no Mac (MPS / CPU)
#
# Uso:
#   bash train_local.sh          # treina BTC/1h (dados já existem)
#   bash train_local.sh 10m      # treina BTC/10m (baixa dados automaticamente)
#   bash train_local.sh 1h ETH   # treina ETH/1h
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

INTERVAL="${1:-1h}"
SYMBOL="${2:-BTC}"
DAYS=730

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python3"
DATA_DIR="$REPO_DIR/finetune_csv/data"
CFG_DIR="$REPO_DIR/finetune_csv/configs"

echo "══════════════════════════════════════════════════════════"
echo "  Kronos Local Training — $SYMBOL / $INTERVAL"
echo "══════════════════════════════════════════════════════════"

# ── 1. Virtual environment ────────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo "▸ Criando virtualenv Python 3.12…"
    /opt/homebrew/bin/python3.12 -m venv "$VENV_DIR"
    "$PYTHON" -m pip install --upgrade pip -q
fi

# ── 2. Dependencies ───────────────────────────────────────────────────────────
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    echo "▸ Instalando dependências (pytorch, pandas, ccxt, yaml…)"
    "$PYTHON" -m pip install -q \
        torch torchvision \
        pandas requests ccxt pyyaml \
        huggingface_hub transformers \
        numpy
    echo "  Dependências instaladas."
fi

# ── 3. Binance data ───────────────────────────────────────────────────────────
BN_CSV="$DATA_DIR/BN_${SYMBOL}_${INTERVAL}.csv"
if [ ! -f "$BN_CSV" ]; then
    echo "▸ Baixando dados Binance $SYMBOL/$INTERVAL ($DAYS dias)…"
    PYTHONPATH="$REPO_DIR" "$PYTHON" -m infra.binance_data_pipeline \
        --symbols "$SYMBOL" --interval "$INTERVAL" --days "$DAYS"
else
    echo "▸ Binance CSV já existe: $(basename "$BN_CSV")"
fi

# ── 4. Hyperliquid data ───────────────────────────────────────────────────────
# Para 10m buscamos 5m e o build_training_dataset reamostiza automaticamente
if [ "$INTERVAL" = "10m" ]; then
    HL_SRC="5m"
else
    HL_SRC="$INTERVAL"
fi

HL_CSV="$DATA_DIR/HL_${SYMBOL}_${HL_SRC}.csv"
if [ ! -f "$HL_CSV" ]; then
    echo "▸ Baixando dados Hyperliquid $SYMBOL/$HL_SRC (200 dias)…"
    PYTHONPATH="$REPO_DIR" "$PYTHON" -m infra.crypto_data_pipeline \
        --coin "$SYMBOL" --interval "$HL_SRC" --days 200
else
    echo "▸ Hyperliquid CSV já existe: $(basename "$HL_CSV")"
fi

# ── 5. Build combined dataset ─────────────────────────────────────────────────
COMBINED_CSV="$DATA_DIR/COMBINED_${SYMBOL}_${INTERVAL}.csv"
echo "▸ Combinando datasets (BN pre-HL + HL)…"
PYTHONPATH="$REPO_DIR" "$PYTHON" -m infra.build_training_dataset \
    --symbol "$SYMBOL" --interval "$INTERVAL"

# ── 6. Train ──────────────────────────────────────────────────────────────────
SYMBOL_LOWER="$(echo "$SYMBOL" | tr '[:upper:]' '[:lower:]')"
CONFIG="$CFG_DIR/config_combined_${SYMBOL_LOWER}_${INTERVAL}.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "ERRO: config não encontrada em $CONFIG" >&2
    exit 1
fi

echo ""
echo "▸ Iniciando treinamento…"
echo "  Config: $CONFIG"
echo "  Device: MPS (se disponível) → CPU"
echo ""

cd "$REPO_DIR"
PYTHONPATH="$REPO_DIR" "$PYTHON" finetune_csv/train_sequential.py --config "$CONFIG"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Treinamento concluído!"
echo "  Modelo salvo em: finetune_csv/finetuned/COMBINED_${SYMBOL}_${INTERVAL}/"
echo "══════════════════════════════════════════════════════════"
