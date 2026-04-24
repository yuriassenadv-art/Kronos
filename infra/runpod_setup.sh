#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RunPod Setup — Kronos Fine-tune
#
# Como usar:
#   1. Crie um pod em runpod.io:
#      - Template: "RunPod PyTorch 2.1"
#      - GPU: RTX 4090 (24GB) para Kronos-small  |  A100 (40GB) para Kronos-base
#      - Disco: 30GB Network Volume (persistente)
#
#   2. Copie os CSVs para o pod:
#      scp -P <PORT> finetune_csv/data/BN_BTC_1h.csv root@<HOST>:/workspace/data/
#
#   3. No terminal do pod, execute este script:
#      bash runpod_setup.sh BTC 1h
#
# Argumentos:
#   $1 = SYMBOL   (ex: BTC)
#   $2 = INTERVAL (ex: 1h)
# ─────────────────────────────────────────────────────────────────────────────

set -e  # para em qualquer erro

SYMBOL="${1:-BTC}"
INTERVAL="${2:-1h}"
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/Kronos"
DATA_DIR="$WORKSPACE/data"

echo "════════════════════════════════════════════════"
echo "  Kronos Fine-tune — RunPod Setup"
echo "  Símbolo : $SYMBOL | Intervalo: $INTERVAL"
echo "════════════════════════════════════════════════"

# ── 1. GPU info ───────────────────────────────────────────────────────────────
echo ""
echo "▸ GPU disponível:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── 2. Clona repositório ──────────────────────────────────────────────────────
echo ""
echo "▸ Clonando repositório Kronos..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/shiyu-coder/Kronos.git "$REPO_DIR"
else
    echo "  Repo já existe, atualizando..."
    cd "$REPO_DIR" && git pull
fi
cd "$REPO_DIR"

# ── 3. Dependências ───────────────────────────────────────────────────────────
echo ""
echo "▸ Instalando dependências..."
pip install -q -r requirements.txt
pip install -q ccxt

# ── 4. Dados ──────────────────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/finetune_csv/data"
CSV_SRC="$DATA_DIR/BN_${SYMBOL}_${INTERVAL}.csv"
CSV_DST="$REPO_DIR/finetune_csv/data/BN_${SYMBOL}_${INTERVAL}.csv"

if [ -f "$CSV_SRC" ]; then
    echo ""
    echo "▸ Copiando CSV de $CSV_SRC..."
    cp "$CSV_SRC" "$CSV_DST"
else
    echo ""
    echo "▸ CSV não encontrado em $CSV_SRC — baixando da Binance..."
    cd "$REPO_DIR"
    python3 -m infra.binance_data_pipeline \
        --symbols "$SYMBOL" \
        --interval "$INTERVAL" \
        --days 1000 \
        --gen-configs
fi

echo "  Linhas do CSV: $(wc -l < "$CSV_DST")"

# ── 5. Gera config apontando para paths do RunPod ─────────────────────────────
echo ""
echo "▸ Gerando config de fine-tune..."

CONFIG_PATH="$REPO_DIR/finetune_csv/configs/runpod_${SYMBOL,,}_${INTERVAL}.yaml"

cat > "$CONFIG_PATH" << YAML
data:
  data_path: "$CSV_DST"
  lookback_window: 512
  predict_window: 24
  max_context: 512
  clip: 5.0
  train_ratio: 0.85
  val_ratio: 0.10
  test_ratio: 0.05

training:
  tokenizer_epochs: 30
  basemodel_epochs: 20
  batch_size: 64
  log_interval: 50
  num_workers: 4
  seed: 42
  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000001
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1
  accumulation_steps: 1

model_paths:
  pretrained_tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
  pretrained_predictor: "NeoQuasar/Kronos-small"
  exp_name: "BN_${SYMBOL}_${INTERVAL}"
  base_path: "$WORKSPACE/finetuned"
  base_save_path: ""
  finetuned_tokenizer: ""
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "kronos_bn_${SYMBOL,,}_${INTERVAL}"
  description: "Kronos fine-tuned on Binance ${SYMBOL}/USDT perp ${INTERVAL} (RunPod)"
  use_comet: false
  train_tokenizer: true
  train_basemodel: true
  skip_existing: false
  pre_trained_tokenizer: true
  pre_trained_predictor: true

device:
  use_cuda: true
  device_id: 0
YAML

echo "  Config: $CONFIG_PATH"

# ── 6. Treina ─────────────────────────────────────────────────────────────────
echo ""
echo "▸ Iniciando treinamento..."
echo "  Checkpoints serão salvos em: $WORKSPACE/finetuned/BN_${SYMBOL}_${INTERVAL}/"
echo ""

mkdir -p "$WORKSPACE/finetuned"

python3 "$REPO_DIR/finetune_csv/train_sequential.py" \
    --config "$CONFIG_PATH"

# ── 7. Resumo ─────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  ✅ Treinamento concluído!"
echo ""
echo "  Modelos salvos em:"
echo "    Tokenizer : $WORKSPACE/finetuned/BN_${SYMBOL}_${INTERVAL}/tokenizer/best_model"
echo "    Predictor : $WORKSPACE/finetuned/BN_${SYMBOL}_${INTERVAL}/basemodel/best_model"
echo ""
echo "  Para baixar o modelo treinado (rode localmente):"
echo "    scp -rP <PORT> root@<HOST>:$WORKSPACE/finetuned/BN_${SYMBOL}_${INTERVAL} \\"
echo "      ./finetune_csv/finetuned/"
echo "════════════════════════════════════════════════"
