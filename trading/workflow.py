"""
Loop principal — versão completa com todas as camadas de infraestrutura.

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
from trading.signal_generator import Direction
from trading.risk_manager import calculate_order, OrderParams
from trading.order_executor import get_account_balance, place_order
from infra.ensemble_predictor import EnsemblePredictor
from infra.state_manager import StateManager, OpenPosition
from infra import monitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_ensemble(cfg: Config) -> EnsemblePredictor:
    logger.info("Carregando EnsemblePredictor (%s)…", cfg.kronos.model_name)
    return EnsemblePredictor(
        model_name=cfg.kronos.model_name,
        tokenizer_name=cfg.kronos.tokenizer_name,
        device=cfg.kronos.device,
        max_context=cfg.kronos.max_context,
        sample_count=cfg.kronos.sample_count,
        temperature=cfg.kronos.temperature,
        top_p=cfg.kronos.top_p,
        use_confirmation_tf=True,
        coin=cfg.trading.coin,
        base_url=cfg.hyperliquid.base_url,
    )


def run_cycle(
    ensemble: EnsemblePredictor,
    state: StateManager,
    cfg: Config,
    cycle: int,
):
    tcfg = cfg.trading
    kcfg = cfg.kronos

    monitor.alert_cycle_start(tcfg.coin, tcfg.interval, cycle)

    # ── 1. Evita abrir nova posição se já há uma aberta ──────────────────────
    if state.has_open_position(tcfg.coin):
        age = state.position_age_seconds(tcfg.coin)
        pos = state.get_position(tcfg.coin)
        logger.info("Posição %s aberta há %.0fs — aguardando SL/TP.", pos.direction, age)
        return

    # ── 2. Busca candles ────────────────────────────────────────────────────
    logger.info("Buscando %s/%s…", tcfg.coin, tcfg.interval)
    df = fetch_candles(
        coin=tcfg.coin,
        interval=tcfg.interval,
        lookback=kcfg.lookback,
        base_url=cfg.hyperliquid.base_url,
    )
    logger.info("Close atual: %.4f", df["close"].iloc[-1])

    # ── 3. Sinal com ensemble + confirmação multi-TF ────────────────────────
    logger.info("Gerando sinal (sample_count=%d)…", kcfg.sample_count)
    signal = ensemble.get_signal(df, kcfg.pred_len, tcfg.interval)
    logger.info("Sinal: %s | conf=%.2f | %s",
                signal.direction.value, signal.confidence, signal.reason)

    monitor.alert_signal(tcfg.coin, signal.direction.value,
                         signal.confidence, signal.reason)

    if signal.direction == Direction.FLAT:
        logger.info("Sinal FLAT — ciclo encerrado sem ordem.")
        return

    # ── 4. Risk management ──────────────────────────────────────────────────
    balance = get_account_balance(cfg.hyperliquid) if not tcfg.dry_run else 1000.0
    order_params: OrderParams = calculate_order(signal, balance, tcfg)

    if order_params is None:
        logger.info("Risk manager bloqueou a ordem (balanço insuficiente ou FLAT).")
        return

    # ── 5. Envia ordem ──────────────────────────────────────────────────────
    result = place_order(order_params, cfg.hyperliquid, tcfg)

    monitor.alert_order_placed(
        tcfg.coin, signal.direction.value,
        order_params.size_usd, order_params.sl_price,
        order_params.tp_price, tcfg.dry_run,
    )

    # ── 6. Persiste estado ──────────────────────────────────────────────────
    state.set_position(OpenPosition(
        coin=tcfg.coin,
        direction=signal.direction.value,
        entry_price=signal.entry_price,
        size_usd=order_params.size_usd,
        sl_price=order_params.sl_price,
        tp_price=order_params.tp_price,
        opened_at=time.time(),
    ))
    logger.info("Estado de posição salvo.")


def main():
    cfg = Config()

    # ── Configuração ─────────────────────────────────────────────────────────
    cfg.trading.coin          = "BTC"
    cfg.trading.interval      = "15m"
    cfg.trading.dry_run       = True      # ⚠️ mude para False apenas após backtest
    cfg.trading.leverage      = 3         # 3x — Long e Short
    cfg.trading.risk_per_trade_pct = 1.0  # 1% do capital por trade
    cfg.trading.sl_pct        = 1.0       # 1% preço → 5% margin (com 5x)
    cfg.trading.tp_pct        = 2.0       # 2% preço → 10% margin (com 5x)
    cfg.trading.loop_interval_seconds = 900  # 15 min

    cfg.kronos.lookback       = 200
    cfg.kronos.pred_len       = 8         # 8 candles × 15m = 2h à frente
    cfg.kronos.sample_count   = 10       # aumentar = mais lento, mais confiável

    # Para usar modelo fine-tunado (após rodar fine-tune):
    # cfg.kronos.model_name     = "finetune_csv/finetuned/BN_BTC_15m/basemodel/best_model"
    # cfg.kronos.tokenizer_name = "finetune_csv/finetuned/BN_BTC_15m/tokenizer/best_model"
    # ─────────────────────────────────────────────────────────────────────────

    ensemble = build_ensemble(cfg)
    state    = StateManager()
    cycle    = 0

    logger.info("Loop iniciado | %s/%s | dry_run=%s | sample_count=%d",
                cfg.trading.coin, cfg.trading.interval,
                cfg.trading.dry_run, cfg.kronos.sample_count)

    while True:
        cycle += 1
        try:
            run_cycle(ensemble, state, cfg, cycle)
        except Exception as exc:
            err = traceback.format_exc()
            logger.error("Erro no ciclo %d: %s", cycle, exc)
            monitor.alert_error(f"ciclo #{cycle}", err)

        logger.info("Próximo ciclo em %ds…", cfg.trading.loop_interval_seconds)
        time.sleep(cfg.trading.loop_interval_seconds)


if __name__ == "__main__":
    main()
