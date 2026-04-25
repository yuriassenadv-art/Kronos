# Trading Multi-Asset Bot — Kronos × Hyperliquid

Bot de trading automatizado que opera 3 ativos (BTC, ETH, SOL) simultâneos na exchange Hyperliquid, usando modelos Kronos fine-tunados em timeframe 15m. O sistema executa previsões em paralelo, aloca capital entre sinais e envia ordens com gerenciamento de risco.

**Modelos utilizados:**
- `savycorp/kronos-bn-btc-15m` (val_loss: 2.7659)
- `savycorp/kronos-bn-eth-15m` (val_loss: 2.7445)
- `savycorp/kronos-bn-sol-15m` (val_loss: 3.0662)

Cada modelo prevê 96 candles de 15 minutos (24 horas) à frente.

## Arquitetura

```
┌─────────────────────────────────────────────────────┐
│  data_fetcher.py — busca candles Hyperliquid API   │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┴────────────┬──────────────────┐
        │                         │                  │
    ┌───▼────┐             ┌─────▼────┐      ┌──────▼──┐
    │  BTC   │             │   ETH    │      │   SOL   │
    │ Ensemble           │ Ensemble │      │Ensemble │
    └───┬────┘             └─────┬────┘      └──────┬──┘
        │                         │                  │
        │ Signal (confidence)     │                  │
        │                         │                  │
        └────────────────┬────────┴──────────────────┘
                         │
        ┌────────────────▼──────────────────┐
        │  portfolio.py — allocate_capital  │
        │  (decisão multi-asset)            │
        └────────────────┬──────────────────┘
                         │
        ┌────────────────▼──────────────────┐
        │  risk_manager.py — calc SL/TP     │
        │  (OrderParams per signal)         │
        └────────────────┬──────────────────┘
                         │
        ┌────────────────▼──────────────────┐
        │  state_manager.py — persistência  │
        │  (JSON recovery)                  │
        └────────────────┬──────────────────┘
                         │
        ┌────────────────▼──────────────────┐
        │  order_executor.py — dry-run ou   │
        │  envio real Hyperliquid           │
        └────────────────┬──────────────────┘
                         │
        ┌────────────────▼──────────────────┐
        │  monitor.py — alertas Telegram    │
        └──────────────────────────────────┘
```

### Camadas do Sistema

| Módulo | Responsabilidade | Entrada | Saída |
|--------|------------------|---------|-------|
| `ensemble_predictor.py` (infra) | Multi-sample prediction + agreement scoring | DataFrame histórico | Signal com confidence |
| `data_fetcher.py` | Busca candles Hyperliquid REST API | coin, interval, lookback | DataFrame OHLCV |
| `signal_generator.py` | Converte forecast → Direction (LONG/SHORT/FLAT) | forecast + histórico | Signal object |
| `risk_manager.py` | Calcula tamanho, SL, TP baseado no sinal | Signal + balance | OrderParams |
| `portfolio.py` | Aloca capital entre múltiplos sinais (TODO: implementar) | dict[coin → Signal] | dict[coin → OrderParams] |
| `order_executor.py` | Envia ordem real ou simula (dry-run) | OrderParams | result (filled/rejected) |
| `state_manager.py` (infra) | Persiste posições abertas em JSON | (read/write local) | crash recovery |
| `monitor.py` (infra) | Notificações Telegram | event (signal/order/error) | mensagem enviada |
| `config.py` | Centraliza todas as configurações | env vars | Config object |
| `workflow.py` | Loop principal — orquestra todas as camadas | Config | ciclos contínuos |

## Setup Inicial

### Pré-requisitos

- **Python 3.10+** instalado
- **pip** ou **uv** para gerenciar dependências
- Acesso à exchange **Hyperliquid** (testnet ou mainnet)
- Token **Hugging Face Hub** (modelos são privados)
- *(Opcional)* Token **Telegram** para alertas

### 1. Instalar Dependências

```bash
cd /Users/savycorp/Kronos
pip install -r requirements.txt
```

### 2. Setup VPS AWS (one-shot) — Download dos Modelos

Os modelos `savycorp/kronos-bn-{btc,eth,sol}-15m` são **privados** no Hugging Face Hub e foram usados apenas como mecanismo de transferência RunPod (treino) → VPS (produção). Após esse passo único, o bot **NÃO precisa mais de internet para o HF** — todos os pesos passam a viver em disco local em `models/{BTC,ETH,SOL}/`.

```bash
HF_TOKEN=hf_xxx python3 scripts/download_models.py
```

- Tamanho total aproximado: **~333 MB** (3 × ~111 MB)
- Token HF: criar em https://huggingface.co/settings/tokens
- Após esse comando, o bot carrega exclusivamente do disco — nenhuma chamada subsequente ao HF

### 3. Configurar Variáveis de Ambiente

| Variável | Obrigatório | Descrição | Exemplo |
|----------|-------------|-----------|---------|
| `HF_TOKEN` | ✅ | Token Hugging Face para modelos privados | `hf_xxxxxxx...` |
| `TELEGRAM_BOT_TOKEN` | ❌ | Token do bot Telegram (criar com @BotFather) | `123456:ABC...` |
| `TELEGRAM_CHAT_ID` | ❌ | ID do chat para receber alertas (obter com /start) | `987654321` |
| `HYPERLIQUID_PRIVATE_KEY` | ❌* | Private key para modo live (deixe vazio em dry-run) | `0x...` |
| `HYPERLIQUID_ACCOUNT_ADDRESS` | ❌* | Endereço da conta (deixe vazio em dry-run) | `0x...` |

*Obrigatório apenas para **live trading**. Em dry-run, pode deixar vazios.

### 4. Setup Telegram (Opcional)

1. Crie um bot em [@BotFather](https://t.me/botfather): `/newbot`
2. Copie o `BOT_TOKEN` fornecido
3. Inicie o bot e converse com ele
4. Obtenha seu `CHAT_ID` acessando: `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates`
5. Configure as variáveis de ambiente

## Como Rodar — Dry-Run (Default, Seguro)

**Dry-run** é o modo padrão: o bot não envia ordens reais, apenas loga e notifica via Telegram.

```bash
cd /Users/savycorp/Kronos
python -m trading.workflow
```

### O Que Esperar

**Logs no console:**
```
2026-04-25 14:30:00 [INFO] Carregando EnsemblePredictor (savycorp/kronos-bn-btc-15m)…
2026-04-25 14:30:15 [INFO] Buscando BTC/15m…
2026-04-25 14:30:16 [INFO] Close atual: 67534.2500
2026-04-25 14:30:17 [INFO] Gerando sinal (sample_count=3)…
2026-04-25 14:30:30 [INFO] Sinal: LONG | conf=0.72 | forecast trending up 24h
2026-04-25 14:30:30 [INFO] Risk manager: size=0.25 BTC @ SL=-1.5% TP=+3.0%
2026-04-25 14:30:30 [INFO] DRY-RUN: seria enviado ORDER LONG BTC …
```

**Telegram (se configurado):**
- `[CYCLE START] BTC/15m — ciclo 1`
- `[SIGNAL] BTC: LONG | conf=0.72 | forecast trending up 24h`
- `[DRY ORDER] LONG 0.25 BTC @ 67534.25 | SL 66497 | TP 69620`

Dry-run continua em loop infinito até `Ctrl+C`. Para interromper: `Ctrl+C`.

## Como Rodar — Modo Live (PRODUÇÃO)

⚠️ **AVISO CRÍTICO**: Modo live envia ordens reais e pode resultar em perdas financeiras.

### Pré-requisitos para Live

1. **Backtest validado**: execute antes de live
   ```bash
   python -m infra.backtester --coin BTC --start 2024-01-01
   ```
   Valide que **Sharpe ratio > 1.0** em todos os 3 ativos antes de ativar live.

2. **Credenciais Hyperliquid**: obtenha em https://app.hyperliquid.xyz
   - Private key (formato `0x...`)
   - Account address (sua carteira)

3. **Configurar variáveis de ambiente**:
   ```bash
   export HYPERLIQUID_PRIVATE_KEY="0x..."
   export HYPERLIQUID_ACCOUNT_ADDRESS="0x..."
   ```

### Ativar Live Trading

Edite `trading/workflow.py` e mude a linha:

```python
# Linha ~150
cfg.trading.dry_run = False  # PERIGO: isto envia ordens reais
```

Depois execute:

```bash
python -m trading.workflow
```

O bot agora envia **ordens reais** na Hyperliquid. Qualquer erro (saldo insuficiente, preço fora de banda, etc) será reportado em Telegram.

**Recomendações operacionais:**
- Comece com saldo pequeno (ex: $100 USDC)
- Monitore Telegram constantemente
- Em caso de anomalia, execute `Ctrl+C` para parar imediatamente
- Revise `trading/.state.json` para estado persistido das posições

## Como Interpretar Logs e Alertas Telegram

### Tipos de Mensagem Telegram

| Tipo | Exemplo | Interpretação |
|------|---------|---------------|
| `[CYCLE START]` | `[CYCLE START] BTC/15m — ciclo 1` | Novo ciclo iniciado (verifica posição aberta, busca dados) |
| `[SIGNAL]` | `[SIGNAL] BTC: LONG \| conf=0.72 \| forecast...` | Sinal gerado; confidence indica qualidade (0.0–1.0) |
| `[DRY ORDER]` / `[ORDER]` | `[DRY ORDER] LONG 0.25 BTC @ 67534.25` | Ordem que seria enviada (dry) ou foi enviada (live) |
| `[ERROR]` | `[ERROR] Failed to fetch BTC/15m: timeout` | Erro na busca de dados ou execução |
| `[POSITION OPEN]` | `[POSITION OPEN] BTC LONG age=0s` | Posição aberta; bot aguarda SL/TP |
| `[POSITION CLOSED]` | `[POSITION CLOSED] BTC: +2.3% (TP hit)` | Posição fechada com P&L |

### Fluxo Típico de um Ciclo

1. **CYCLE START** — ciclo inicia
2. **Verifica posição aberta** — se sim, aguarda SL/TP e pula para próximo ciclo
3. **BUSCA DADOS** — fetch candles BTC/ETH/SOL
4. **SIGNALS** — gera sinal para cada ativo
5. **PORTFOLIO ALLOCATION** — decide qual(is) ativo(s) recebe(m) capital
6. **RISK CALC** — calcula size, SL, TP
7. **ORDER** — envia (ou simula) ordem
8. **TELEGRAM** — notifica resultado
9. **SLEEP** — aguarda `loop_interval_seconds` (padrão 3600s = 1h)

### Mapeamento Log → Estado do Bot

| Log | Significado | Ação do Usuário |
|-----|-------------|-----------------|
| `DRY-RUN: seria enviado...` | Modo dry-run ativo, nenhuma ordem real | Normal em testes |
| `Posição X aberta há 45m — aguardando SL/TP` | Posição aguardando fechamento | Monitore até SL/TP |
| `Sinal: FLAT \| conf=...` | Ensemble não vê oportunidade | OK, aguarde próximo ciclo |
| `Risk manager bloqueou a ordem` | Saldo insuficiente ou tamanho inválido | Aumentar saldo ou diminuir risk_per_trade_pct |
| `Failed to fetch ... 451` | Geo-block Hyperliquid (IP fora de zona permitida) | Usar VPN ou data.binance.vision |
| `ModuleNotFoundError` | Dependência faltando | `pip install -r requirements.txt` |

## Configuração Multi-Asset

### Alterar Coins Operados

Edite `trading/config.py`:

```python
@dataclass
class TradingConfig:
    coins: list[str] = field(
        default_factory=lambda: ["BTC", "ETH", "SOL"]  # Mude aqui
    )
    max_concurrent_positions: int = 3  # Máximo de posições abertas simultaneamente
```

Exemplo: para operar apenas BTC e ETH:

```python
coins: list[str] = field(
    default_factory=lambda: ["BTC", "ETH"]
)
max_concurrent_positions: int = 2
```

### Adicionar Novo Ativo (ex: AVAX)

1. **Fine-tunação**: treine um modelo `savycorp/kronos-bn-avax-15m` usando `scripts/finetune.py`

2. **Adicionar ao mapeamento** (path local):
   ```python
   # trading/config.py
   DEFAULT_MODEL_PATHS: dict[str, str] = {
       "BTC":  str(PROJECT_ROOT / "models" / "BTC"),
       "ETH":  str(PROJECT_ROOT / "models" / "ETH"),
       "SOL":  str(PROJECT_ROOT / "models" / "SOL"),
       "AVAX": str(PROJECT_ROOT / "models" / "AVAX"),  # NOVO
   }
   ```

   E atualize `scripts/download_models.py` para incluir o novo repo HF antes de rodar o download na VPS.

3. **Atualizar TradingConfig**:
   ```python
   coins: list[str] = field(
       default_factory=lambda: ["BTC", "ETH", "SOL", "AVAX"]
   )
   max_concurrent_positions: int = 4
   ```

4. **Backtest e validar Sharpe**:
   ```bash
   python -m infra.backtester --coin AVAX --start 2024-01-01
   ```

## Decisão de Alocação de Capital

A função `trading/portfolio.py:allocate_capital()` decide como dividir o saldo entre múltiplos sinais simultâneos. **Esta é uma decisão de design deixada para você implementar** — veja TODO no arquivo.

### 4 Abordagens Canônicas

1. **Equal-weight** — cada ativo ativo recebe `risk_per_trade_pct / N`
   - Simples, previsível, sem concentração

2. **Confidence-weighted** — aloca proporcional a `signal.confidence`
   - Usa informação do ensemble; pode concentrar se confidence dispara

3. **Sharpe-weighted** — aloca proporcional ao Sharpe histórico do ativo
   - Prioriza onde modelo tem edge; requer backtests recentes

4. **Best-only** — aloca 100% ao sinal com maior confidence
   - Máxima simplicidade; descarta diversificação

**Constraints obrigatórios:**
- Filtrar `Direction.FLAT` antes de alocar
- Respeitar `cfg.max_concurrent_positions` — se > N sinais, pegar top-N
- Nunca alocar mais que `cfg.risk_per_trade_pct` por trade

Exemplo de implementação equal-weight (15 linhas):

```python
def allocate_capital(signals, balance, cfg):
    active = {c: s for c, s in signals.items() if s.direction != Direction.FLAT}
    if not active:
        return {}
    
    n_active = min(len(active), cfg.max_concurrent_positions)
    risk_adjusted = cfg.risk_per_trade_pct / n_active
    
    result = {}
    for coin, signal in list(active.items())[:n_active]:
        cfg_copy = dataclasses.replace(cfg, risk_per_trade_pct=risk_adjusted)
        result[coin] = calculate_order(signal, balance, cfg_copy)
    
    return result
```

Veja `trading/portfolio.py` para documentação completa das 4 abordagens.

## Deploy em VPS AWS

1. Provisione `t3.medium` (2 vCPU / 4GB RAM) — suficiente para inferência CPU dos 3 modelos
2. Clone o repo na VPS
3. Instale deps: `pip install -r requirements.txt` + `hyperliquid-python-sdk`
4. Rode `scripts/download_models.py` uma vez (precisa `HF_TOKEN`) — após isso, sem dependência runtime do HF
5. Configure systemd unit (exemplo abaixo) para rodar `python3 -m trading.workflow` em background
6. Ative `dry_run=True` por pelo menos 1 semana antes de live

### Exemplo de systemd unit

`/etc/systemd/system/kronos-bot.service`:

```ini
[Unit]
Description=Kronos Multi-Asset Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/Kronos
Environment="TELEGRAM_BOT_TOKEN=123456:ABC..."
Environment="TELEGRAM_CHAT_ID=987654321"
Environment="HYPERLIQUID_PRIVATE_KEY=0x..."
Environment="HYPERLIQUID_ACCOUNT_ADDRESS=0x..."
ExecStart=/usr/bin/python3 -m trading.workflow
# Auto-restart: o bot sai com exit 1 quando detecta falha terminal de
# carregamento de modelo ou ≥2 health checks consecutivos. systemd reinicia
# automaticamente após RestartSec.
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/kronos-bot.log
StandardError=append:/var/log/kronos-bot.err

[Install]
WantedBy=multi-user.target
```

Ativação:

```bash
sudo systemctl daemon-reload
sudo systemctl enable kronos-bot
sudo systemctl start kronos-bot
sudo systemctl status kronos-bot          # verificar
sudo journalctl -u kronos-bot -f          # acompanhar logs
```

## Troubleshooting

### Erro: `ModuleNotFoundError: No module named 'transformers'`

**Solução:**
```bash
pip install -r requirements.txt
```

### Erro: `Path do modelo {COIN} não encontrado: ...`

**Causa:** Os pesos ainda não foram baixados para o disco local da VPS.

**Solução:**
```bash
HF_TOKEN=hf_xxx python3 scripts/download_models.py
```

Após esse comando, `models/{BTC,ETH,SOL}/` ficam populados e o bot carrega
exclusivamente do disco — nenhuma chamada subsequente ao Hugging Face Hub.

### Erro: `401 Unauthorized — Hugging Face Hub` (durante download_models.py)

**Causa:** Modelos privados exigem token HF válido.

**Solução:** garanta que `HF_TOKEN` está exportado e tem permissão nos repos
`savycorp/kronos-bn-{btc,eth,sol}-15m` antes de rodar o script de download.

### Erro: `429 Too Many Requests — Hyperliquid API`

**Causa:** Rate limit atingido.

**Solução:** Aumentar `loop_interval_seconds` em `trading/config.py`:

```python
@dataclass
class TradingConfig:
    loop_interval_seconds: int = 7200  # Aumentar de 3600 para 7200
```

### Erro: `HTTP 451 Unavailable For Legal Reasons`

**Causa:** Geo-block Hyperliquid (IP em região não-permitida).

**Solução:** Usar VPN ou alternar para `data.binance.vision` em `data_fetcher.py`.

### Erro: `Telegram não envia mensagens`

**Causa:** Token ou Chat ID inválido/faltando.

**Solução:**
```bash
export TELEGRAM_BOT_TOKEN="sua_bot_token_aqui"
export TELEGRAM_CHAT_ID="seu_chat_id_aqui"
```

Teste manualmente:
```python
from infra import monitor
monitor.alert_cycle_start("BTC", "15m", 1)
```

### Erro: `Posição duplicada / state.json corrompido`

**Causa:** Crash durante execução deixa JSON inconsistente.

**Solução:**
```bash
rm trading/.state.json
# Bot recria o arquivo na próxima execução
python -m trading.workflow
```

## Próximos Passos

1. **Implementar `allocate_capital()`** — escolha uma das 4 abordagens em `trading/portfolio.py`

2. **Rodar backtest** dos 3 modelos:
   ```bash
   for coin in BTC ETH SOL; do
       python -m infra.backtester --coin $coin --start 2024-01-01
   done
   ```

3. **Validar Sharpe ratio > 1.0** em todos os 3 ativos antes de live

4. **Começar com pequena alocação** (~$100 USDC em live)

5. **Monitorar Telegram continuamente** nos primeiros ciclos

6. **Documentar resultados** em `project_kronos_trading.md` no memory

## Referências

- [Kronos — arXiv paper](https://arxiv.org/abs/2508.02739)
- [Hyperliquid API docs](https://hyperliquid-testnet.gitbook.io/api)
- [Hugging Face Hub — Private Models](https://huggingface.co/docs/hub/security-tokens)
- [`infra/ensemble_predictor.py`](../infra/ensemble_predictor.py) — multi-sample prediction
- [`infra/backtester.py`](../infra/backtester.py) — walk-forward backtest
- [`trading/portfolio.py`](portfolio.py) — alocação de capital (TODO)
- [`trading/config.py`](config.py) — configuração centralizada
