"""
Loop principal multi-asset — opera 3 ativos (BTC/ETH/SOL) em paralelo
usando 3 modelos Kronos fine-tunados (savycorp/kronos-bn-{btc,eth,sol}-15m).

Uso:
    python -m trading.workflow
"""

import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.config import Config
from trading.data_fetcher import fetch_candles
from trading.signal_generator import Direction, Signal
from trading.risk_manager import OrderParams
from trading.order_executor import get_account_balance, place_order
from trading.portfolio import allocate_capital
from infra.ensemble_predictor import EnsemblePredictor
from infra.state_manager import StateManager, OpenPosition
from infra.position_watcher import sync_positions
from infra.health_check import build_ensembles_with_retry, ensemble_health_check
from infra.funding import refresh_funding_cache
from infra import monitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Construção dos ensembles (1 EnsemblePredictor por coin)
# ─────────────────────────────────────────────────────────────────────────────
def build_ensembles(cfg: Config) -> dict[str, EnsemblePredictor]:
    """
    Carrega um EnsemblePredictor para cada coin de cfg.trading.coins.

    Cada coin usa o path local em disco correspondente em
    cfg.kronos.model_paths[coin], com o layout savycorp
    (basemodel/best_model + tokenizer/best_model). O tokenizer mora no MESMO
    diretório do modelo — apenas em outro subfolder.

    Os pesos vivem em disco local (baixados 1 vez via
    `scripts/download_models.py` na VPS). Se o path não existir, levantamos
    RuntimeError com instrução de baixar antes.

    Returns:
        dict[coin, EnsemblePredictor] — ex: {"BTC": pred1, "ETH": pred2, ...}
    """
    kcfg = cfg.kronos
    ensembles: dict[str, EnsemblePredictor] = {}

    for coin in cfg.trading.coins:
        local_path = kcfg.model_paths.get(coin)
        if not local_path:
            logger.warning(
                "Coin %s não tem path em cfg.kronos.model_paths — pulando.", coin
            )
            continue

        if not Path(local_path).exists():
            raise RuntimeError(
                f"Path do modelo {coin} não encontrado: {local_path}. "
                f"Rode `HF_TOKEN=hf_xxx python3 scripts/download_models.py` "
                f"na VPS para baixar os pesos do Hugging Face Hub uma única vez."
            )

        logger.info(
            "Carregando EnsemblePredictor para %s a partir de %s", coin, local_path
        )
        ensembles[coin] = EnsemblePredictor(
            model_name=local_path,
            tokenizer_name=local_path,             # mesmo path, subfolder distinto
            model_subfolder=kcfg.model_subfolder,
            tokenizer_subfolder=kcfg.tokenizer_subfolder,
            device=kcfg.device,
            max_context=kcfg.max_context,
            sample_count=kcfg.sample_count,
            temperature=kcfg.temperature,
            top_p=kcfg.top_p,
            use_confirmation_tf=True,
            coin=coin,
            base_url=cfg.hyperliquid.base_url,
        )

    if not ensembles:
        raise RuntimeError(
            "Nenhum EnsemblePredictor foi carregado — verifique cfg.trading.coins "
            "e cfg.kronos.model_paths."
        )

    return ensembles


# ─────────────────────────────────────────────────────────────────────────────
# Geração resiliente de sinais (1 falha não derruba os outros)
# ─────────────────────────────────────────────────────────────────────────────
def generate_signals(
    ensembles: dict[str, EnsemblePredictor],
    cfg: Config,
) -> dict[str, Signal]:
    """
    Para cada coin com ensemble carregado, busca candles e gera um Signal.

    Resiliência: se um coin falhar (rede, modelo, dados), capturamos a exceção,
    logamos via monitor.alert_error e SEGUIMOS para o próximo coin. O ciclo
    nunca crasha por causa de 1 ativo isolado.

    Returns:
        dict[coin, Signal] — só inclui coins que produziram sinal com sucesso.
    """
    tcfg = cfg.trading
    kcfg = cfg.kronos
    signals: dict[str, Signal] = {}

    for coin, ensemble in ensembles.items():
        try:
            logger.info("Buscando candles %s/%s…", coin, tcfg.interval)
            df = fetch_candles(
                coin=coin,
                interval=tcfg.interval,
                lookback=kcfg.lookback,
                base_url=cfg.hyperliquid.base_url,
            )
            logger.info("[%s] close atual: %.4f", coin, df["close"].iloc[-1])

            logger.info(
                "[%s] Gerando sinal (sample_count=%d)…", coin, kcfg.sample_count
            )
            signal = ensemble.get_signal(df, kcfg.pred_len, tcfg.interval)
            logger.info(
                "Sinal %s: %s | conf=%.2f | reason=%s",
                coin,
                signal.direction.value,
                signal.confidence,
                signal.reason,
            )

            monitor.alert_signal(
                coin, signal.direction.value, signal.confidence, signal.reason
            )
            signals[coin] = signal

        except Exception as exc:
            err = traceback.format_exc()
            logger.error("Falha ao gerar sinal para %s: %s", coin, exc)
            monitor.alert_error(f"signal_{coin}", err)
            # NÃO faz raise — ciclo continua para os outros coins

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Filtragem por estado (não reabre posição já existente)
# ─────────────────────────────────────────────────────────────────────────────
def filter_already_open(
    signals: dict[str, Signal],
    state: StateManager,
) -> dict[str, Signal]:
    """
    Remove de `signals` qualquer coin que já tenha posição aberta no state.

    Loga as posições abertas existentes para debug. Sinais Direction.FLAT são
    deixados para o portfolio.allocate_capital filtrar (mais coeso lá).

    Returns:
        dict[coin, Signal] — apenas coins SEM posição aberta no momento.
    """
    open_positions = state.all_positions()
    if open_positions:
        logger.info(
            "Posições já abertas (%d): %s",
            len(open_positions),
            ", ".join(
                f"{c}={p.direction}@{p.entry_price:.2f}"
                for c, p in open_positions.items()
            ),
        )

    filtered: dict[str, Signal] = {}
    for coin, sig in signals.items():
        if state.has_open_position(coin):
            age = state.position_age_seconds(coin)
            logger.info(
                "[%s] já tem posição aberta há %.0fs — sinal descartado.",
                coin,
                age,
            )
            continue
        filtered[coin] = sig

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Execução das alocações vindas do portfolio
# ─────────────────────────────────────────────────────────────────────────────
def execute_orders(
    allocations: dict[str, OrderParams],
    signals: dict[str, Signal],
    state: StateManager,
    cfg: Config,
):
    """
    Para cada (coin, OrderParams) alocado pelo portfolio:
      1. envia ordem (place_order respeita dry_run)
      2. notifica via monitor.alert_order_placed
      3. persiste OpenPosition no state

    `signals` é passado para recuperar o entry_price (que vive no Signal,
    não no OrderParams) ao montar o OpenPosition do state.
    """
    tcfg = cfg.trading

    if not allocations:
        logger.info("Nenhuma alocação produzida pelo portfolio — nada a executar.")
        return

    for coin, params in allocations.items():
        try:
            direction_str = "LONG" if params.is_buy else "SHORT"
            logger.info(
                "[%s] Enviando ordem %s | size=$%.2f | SL=%.2f | TP=%.2f",
                coin,
                direction_str,
                params.size_usd,
                params.sl_price,
                params.tp_price,
            )
            place_order(params, cfg.hyperliquid, tcfg)

            monitor.alert_order_placed(
                coin,
                direction_str,
                params.size_usd,
                params.sl_price,
                params.tp_price,
                tcfg.dry_run,
            )

            # Entry price: extraído do Signal correspondente. Fallback para
            # o midpoint SL/TP caso (improvável) o signal não esteja presente.
            sig = signals.get(coin)
            if sig is not None:
                entry_price = sig.entry_price
            else:
                entry_price = (params.sl_price + params.tp_price) / 2

            state.set_position(
                OpenPosition(
                    coin=coin,
                    direction=direction_str,
                    entry_price=float(entry_price),
                    size_usd=float(params.size_usd),
                    sl_price=float(params.sl_price),
                    tp_price=float(params.tp_price),
                    opened_at=time.time(),
                )
            )
            logger.info("[%s] Estado de posição persistido.", coin)

        except Exception as exc:
            err = traceback.format_exc()
            logger.error("Falha ao executar ordem para %s: %s", coin, exc)
            monitor.alert_error(f"order_{coin}", err)
            # ordem de 1 coin que falha NÃO impede as demais


# ─────────────────────────────────────────────────────────────────────────────
# Ciclo principal multi-asset
# ─────────────────────────────────────────────────────────────────────────────
def run_cycle(
    ensembles: dict[str, EnsemblePredictor],
    state: StateManager,
    cfg: Config,
    cycle_num: int,
):
    """
    Um ciclo completo do bot multi-asset.

    Pipeline:
      1. notifica início
      2. gera sinais para todos os coins (resiliente a falhas isoladas)
      3. filtra coins com posição já aberta
      4. consulta saldo (1000 USDC simulado em dry_run)
      5. delega ao portfolio.allocate_capital o split entre os ativos
      6. executa as alocações resultantes
    """
    tcfg = cfg.trading

    monitor.alert_cycle_start(", ".join(tcfg.coins), tcfg.interval, cycle_num)
    logger.info(
        "── Ciclo #%d | coins=%s | interval=%s | dry_run=%s ──",
        cycle_num,
        ",".join(tcfg.coins),
        tcfg.interval,
        tcfg.dry_run,
    )

    # 0. Sync de posições — detecta SL/TP disparados ENTRE ciclos.
    # Sem isso, posições fechadas pela exchange ficam "fantasma" no state
    # e bloqueiam a reabertura via filter_already_open.
    try:
        closed = sync_positions(state, cfg.hyperliquid, tcfg.dry_run)
        if closed:
            logger.warning(
                "Sync de posições: %d fechada(s) detectada(s): %s",
                len(closed), ", ".join(closed),
            )
        else:
            logger.info("Posições em sync com a exchange.")
    except Exception as exc:
        # sync_positions já trata erros internamente, mas guard extra
        # garante que NUNCA derrube o ciclo.
        err = traceback.format_exc()
        logger.warning("Falha em sync_positions (não-fatal): %s", exc)
        monitor.alert_error("sync_positions", err)

    # 1. Sinais
    signals = generate_signals(ensembles, cfg)
    if not signals:
        logger.info("Nenhum sinal gerado neste ciclo — encerrando ciclo.")
        return

    # 2. Filtro por posições já abertas
    signals_filtered = filter_already_open(signals, state)
    if not signals_filtered:
        logger.info(
            "Todas as posições candidatas já estão abertas — nada a fazer."
        )
        return

    # 3. Saldo
    if tcfg.dry_run:
        balance = 1000.0
        logger.info("dry_run — usando saldo simulado de $%.2f", balance)
    else:
        try:
            balance = get_account_balance(cfg.hyperliquid)
            logger.info("Saldo real disponível: $%.2f", balance)
        except Exception as exc:
            err = traceback.format_exc()
            logger.error("Falha ao consultar saldo: %s", exc)
            monitor.alert_error("get_account_balance", err)
            return

    # 4a. Refresh do cache de funding ANTES da alocação. Falha aqui não é
    # fatal: penalidades caem para 1.0 (sem penalidade) e o ciclo segue
    # exatamente como antes da feature.
    try:
        refresh_funding_cache(cfg.hyperliquid.base_url)
    except Exception as exc:
        logger.warning(
            "Falha ao atualizar funding (sem penalidade neste ciclo): %s", exc
        )

    # 4. Portfolio (decisão de design do usuário — ainda NotImplementedError)
    try:
        allocations = allocate_capital(signals_filtered, balance, tcfg)
    except NotImplementedError as exc:
        logger.error(
            "allocate_capital ainda não implementado — usuário precisa codar "
            "uma das abordagens A/B/C/D do docstring de portfolio.py. "
            "Ciclo será pulado, loop NÃO crashea. Detalhe: %s",
            exc,
        )
        monitor.alert_error("portfolio.allocate_capital", str(exc))
        return
    except Exception as exc:
        err = traceback.format_exc()
        logger.error("Erro inesperado em allocate_capital: %s", exc)
        monitor.alert_error("portfolio.allocate_capital", err)
        return

    # 5. Execução
    execute_orders(allocations, signals_filtered, state, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Validação de credenciais Hyperliquid (pré-flight)
# ─────────────────────────────────────────────────────────────────────────────
def _validate_credentials(cfg: Config) -> None:
    """
    Valida credenciais Hyperliquid antes de iniciar o loop.

    - dry_run=True: skip (não precisa de credenciais reais)
    - dry_run=False:
        1. private_key e account_address presentes (env vars)
        2. private_key tem formato 0x... (64 hex chars)
        3. account_address tem formato 0x... (40 hex chars)
        4. get_account_balance retorna > $10 (saldo mínimo viável)

    Em qualquer falha em modo live: sys.exit(1) — supervisor reinicia
    se ainda assim quiser tentar (mas o erro será o mesmo até corrigir).
    """
    if cfg.trading.dry_run:
        logger.info("dry_run=True — pulando validação de credenciais Hyperliquid.")
        return

    hl = cfg.hyperliquid

    # Verificação 1: campos não vazios
    if not hl.private_key or not hl.account_address:
        logger.error(
            "FATAL: dry_run=False mas HYPERLIQUID_PRIVATE_KEY e/ou "
            "HYPERLIQUID_ACCOUNT_ADDRESS estão vazios. Configure as env vars "
            "(via systemd Environment= ou export) antes de iniciar."
        )
        sys.exit(1)

    # Verificação 2: formato (heurística simples — não valida hex)
    if not hl.private_key.startswith("0x") or len(hl.private_key) != 66:
        logger.error(
            "FATAL: HYPERLIQUID_PRIVATE_KEY parece inválida (esperado 0x + 64 hex chars). "
            "Verifique se copiou a API wallet key inteira."
        )
        sys.exit(1)

    if not hl.account_address.startswith("0x") or len(hl.account_address) != 42:
        logger.error(
            "FATAL: HYPERLIQUID_ACCOUNT_ADDRESS parece inválido (esperado 0x + 40 hex chars)."
        )
        sys.exit(1)

    # Verificação 3: saldo via API (testa conectividade + assinatura implícita)
    try:
        balance = get_account_balance(hl)
        addr_short = hl.account_address[:6] + "…" + hl.account_address[-4:]
        logger.info(
            "✓ Auth Hyperliquid OK | account=%s | balance=$%.2f",
            addr_short, balance,
        )
        if balance < 10.0:
            logger.error(
                "FATAL: balance=$%.2f < $10 mínimo. Deposite USDC na wallet principal "
                "antes de operar live.",
                balance,
            )
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error(
            "FATAL: get_account_balance falhou — credenciais ou rede. Erro: %s",
            exc,
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = Config()

    # ── Configuração ─────────────────────────────────────────────────────────
    # ⚠️ MODO MICRO-CAPITAL ($19 USDC inicial) — ajustes para ficar acima do
    # mínimo de ordem da Hyperliquid (~$10 notional). Trade-offs:
    #   - 1 coin só (BTC) — capital insuficiente para multi-asset
    #   - leverage 6x — cobre o mínimo de notional sem inflar risco bruto
    #   - risk_per_trade_pct 10% × leverage 6 × balance $19 = ~$11.4 notional
    #   - SL 1.0% / TP 2.0% — perda real ~$0.11/trade, ganho ~$0.23/trade (R:R 1:2)
    # Ao depositar mais capital ($100+), restaure: coins=["BTC","ETH","SOL"],
    # leverage=3, risk_per_trade_pct=1.0, sl_pct=1.5, tp_pct=3.0.
    cfg.trading.coins         = ["BTC"]
    cfg.trading.interval      = "15m"     # timeframe nativo dos modelos savycorp
    cfg.trading.dry_run       = True      # ⚠️ NUNCA commite com False
    cfg.trading.leverage      = 6
    cfg.trading.risk_per_trade_pct = 10.0
    cfg.trading.sl_pct        = 1.0
    cfg.trading.tp_pct        = 2.0
    cfg.trading.max_concurrent_positions = 1
    cfg.trading.loop_interval_seconds = 900   # 15min

    cfg.kronos.lookback       = 200
    cfg.kronos.pred_len       = 10
    cfg.kronos.sample_count   = 10
    # ─────────────────────────────────────────────────────────────────────────

    # Validação de credenciais (sai com exit 1 em modo live se algo errado)
    _validate_credentials(cfg)

    # Salvaguarda visível: alerta caso alguém troque dry_run para False.
    if not cfg.trading.dry_run:
        warning_banner = (
            "\n" + "!" * 78 + "\n"
            "!!  WARNING: dry_run=False — ORDENS REAIS SERÃO ENVIADAS À HYPERLIQUID  !!\n"
            "!!  Confirme: chave privada configurada, capital correto, backtest OK.  !!\n"
            + "!" * 78 + "\n"
        )
        logger.warning(warning_banner)
        monitor.alert_error("STARTUP", "dry_run=False — modo LIVE ativo")

    logger.info(
        "Carregando ensembles dos %d modelos… (pode demorar — ~111MB de pesos por coin)",
        len(cfg.trading.coins),
    )
    # Wrapper com retry: se carregamento falhar 3x consecutivos, sys.exit(1)
    # → systemd reinicia o processo. Isolamento de falhas terminais de modelo.
    ensembles = build_ensembles_with_retry(
        lambda: build_ensembles(cfg),
        max_attempts=3,
        base_delay=5.0,
    )
    logger.info("Ensembles prontos: %s", list(ensembles.keys()))

    state = StateManager()
    cycle = 0

    # Reconciliação on startup: detecta posições "fantasma" (state.json
    # desincronizado da exchange — ex: bot crashou + SL/TP disparou enquanto offline).
    logger.info("Reconciliação de startup: comparando state.json com exchange…")
    closed = sync_positions(state, cfg.hyperliquid, cfg.trading.dry_run)
    if closed:
        logger.warning(
            "Removidas %d posição(ões) fantasma do state: %s",
            len(closed), ", ".join(closed)
        )
    else:
        logger.info("State em sync com a exchange.")

    logger.info(
        "Loop multi-asset iniciado | coins=%s | %s | dry_run=%s | sample_count=%d",
        ",".join(cfg.trading.coins),
        cfg.trading.interval,
        cfg.trading.dry_run,
        cfg.kronos.sample_count,
    )

    # A cada N ciclos, valida saúde dos ensembles (smoke inference).
    # Em 15m × 4 = 1h. Se 2 health checks consecutivos falharem em todos
    # os ensembles, ensemble_health_check chama sys.exit(1) → systemd reinicia.
    HEALTH_CHECK_INTERVAL_CYCLES = 4

    while True:
        cycle += 1
        try:
            run_cycle(ensembles, state, cfg, cycle)
        except Exception as exc:
            err = traceback.format_exc()
            logger.error("Erro fatal no ciclo %d: %s", cycle, exc)
            monitor.alert_error(f"ciclo #{cycle}", err)

        # Health check periódico — pode chamar sys.exit(1) se degradação total
        if cycle % HEALTH_CHECK_INTERVAL_CYCLES == 0:
            ensemble_health_check(ensembles)

        logger.info(
            "Próximo ciclo em %ds…", cfg.trading.loop_interval_seconds
        )
        time.sleep(cfg.trading.loop_interval_seconds)


if __name__ == "__main__":
    main()
