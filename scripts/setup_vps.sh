#!/bin/bash
###############################################################################
# Setup automatizado do Bot Kronos × Hyperliquid em VPS Ubuntu 22.04 LTS
#
# Uso: rode UMA VEZ na VPS recém-criada (Lightsail/EC2):
#
#   curl -sSL https://raw.githubusercontent.com/yuriassenadv-art/Kronos/feature/multi-asset-bot-vps/scripts/setup_vps.sh | bash
#
# O que esse script faz:
#   1. Atualiza Ubuntu e instala dependências (Python 3.11, git, build tools)
#   2. Clona o branch feature/multi-asset-bot-vps do GitHub
#   3. Cria venv Python e instala todas as dependências
#   4. Verifica latência com a Hyperliquid (ping)
#   5. Imprime próximos passos manuais (download de modelos + credenciais)
#
# NÃO baixa modelos (precisa de HF_TOKEN — você vai colar manualmente depois)
# NÃO inicia o bot (precisa de credenciais Hyperliquid — você vai colar depois)
###############################################################################

set -e  # qualquer erro → para o script

REPO_URL="https://github.com/yuriassenadv-art/Kronos.git"
BRANCH="feature/multi-asset-bot-vps"
PROJECT_DIR="$HOME/Kronos"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  KRONOS BOT — SETUP AUTOMÁTICO DA VPS"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# ── PASSO 1: Atualiza Ubuntu ──────────────────────────────────────────────────
echo "[1/5] Atualizando Ubuntu e instalando dependências do sistema…"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    curl \
    > /dev/null
echo "     ✓ Sistema atualizado"
echo ""

# ── PASSO 2: Clone do repositório ────────────────────────────────────────────
echo "[2/5] Baixando código do GitHub (branch $BRANCH)…"
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "     Repositório já existe — fazendo git pull"
    cd "$PROJECT_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone -b "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi
echo "     ✓ Código em $PROJECT_DIR"
echo ""

# ── PASSO 3: Ambiente Python isolado (venv) ──────────────────────────────────
echo "[3/5] Criando ambiente Python isolado e instalando dependências…"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet \
    hyperliquid-python-sdk \
    huggingface_hub \
    eth_account
echo "     ✓ Python e dependências OK"
echo ""

# ── PASSO 4: Latency check com a Hyperliquid ─────────────────────────────────
echo "[4/5] Medindo latência até a Hyperliquid API…"
PING_OUTPUT=$(ping -c 5 -W 2 api.hyperliquid.xyz 2>/dev/null | tail -1 || echo "")
if [ -n "$PING_OUTPUT" ]; then
    echo "     $PING_OUTPUT"
else
    echo "     (ping não respondeu — algumas redes bloqueiam ICMP, OK)"
fi

# Mede latência HTTP real (mais relevante que ping)
echo "     Testando chamada HTTP real…"
HTTP_TIME=$(curl -s -o /dev/null -w "%{time_total}" -X POST \
    https://api.hyperliquid.xyz/info \
    -H "Content-Type: application/json" \
    -d '{"type":"meta"}')
echo "     ✓ HTTP round-trip: ${HTTP_TIME}s (esperado < 0.10s em Singapore/Tokyo)"
echo ""

# ── PASSO 5: Próximos passos ─────────────────────────────────────────────────
echo "[5/5] Setup completo! ✅"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  PRÓXIMOS PASSOS (manuais — precisa das suas credenciais)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "1) BAIXAR OS MODELOS DO HUGGING FACE (~333 MB, 5-10 min)"
echo ""
echo "   Você precisa do HF_TOKEN (já tem — usado no treino do RunPod)."
echo "   Cole o comando abaixo trocando hf_xxxxx pelo seu token real:"
echo ""
echo "   cd ~/Kronos"
echo "   source venv/bin/activate"
echo "   HF_TOKEN=hf_xxxxx python3 scripts/download_models.py"
echo ""
echo "2) CRIAR API WALLET NA HYPERLIQUID"
echo ""
echo "   - Acesse https://app.hyperliquid.xyz com sua wallet (Metamask/Rabby)"
echo "   - Deposite USDC (recomendado começar com \$100-500)"
echo "   - Account → API → Generate New API Wallet"
echo "   - Anote 2 valores:"
echo "       a) API private key (0x… 64 caracteres hex)"
echo "       b) Account address (0x… 40 caracteres hex)"
echo ""
echo "3) RODAR SMOKE TEST (envia ordem real \$12 e fecha em 5s, custo ~\$0.10)"
echo ""
echo "   HYPERLIQUID_PRIVATE_KEY=0x... \\"
echo "   HYPERLIQUID_ACCOUNT_ADDRESS=0x... \\"
echo "   python3 scripts/first_contact_test.py BTC"
echo ""
echo "4) ME CHAME DE VOLTA com o resultado dos passos 1-3 e eu termino."
echo ""
echo "════════════════════════════════════════════════════════════════════════"
