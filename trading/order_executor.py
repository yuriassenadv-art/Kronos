"""
Executa ordens na Hyperliquid via hyperliquid-python-sdk.

Instale: pip install hyperliquid-python-sdk
"""

import logging
from typing import Optional

from .config import HyperliquidConfig, TradingConfig
from .risk_manager import OrderParams

logger = logging.getLogger(__name__)


def get_account_balance(hl_cfg: HyperliquidConfig) -> float:
    """Retorna o saldo USDC disponível na conta."""
    from hyperliquid.info import Info

    info = Info(hl_cfg.base_url, skip_ws=True)
    state = info.user_state(hl_cfg.account_address)
    return float(state["marginSummary"]["accountValue"])


def place_order(
    params: OrderParams,
    hl_cfg: HyperliquidConfig,
    trading_cfg: TradingConfig,
) -> Optional[dict]:
    """
    Envia uma ordem market + SL/TP para a Hyperliquid.

    dry_run=True → apenas loga, sem enviar.
    """
    if trading_cfg.dry_run:
        logger.info(
            "[DRY RUN] %s %s | size=$%.2f | SL=%.2f | TP=%.2f | lev=%dx",
            "LONG" if params.is_buy else "SHORT",
            params.coin,
            params.size_usd,
            params.sl_price,
            params.tp_price,
            params.leverage,
        )
        return {"status": "dry_run", "params": params}

    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    import eth_account

    account = eth_account.Account.from_key(hl_cfg.private_key)
    exchange = Exchange(
        account,
        constants.MAINNET_API_URL if hl_cfg.mainnet else constants.TESTNET_API_URL,
    )

    # Ajusta alavancagem
    exchange.update_leverage(params.leverage, params.coin, is_cross=True)

    # Tamanho em moeda base (ex: BTC)
    # Precisamos do preço atual para converter USD → qty
    from hyperliquid.info import Info
    info = Info(hl_cfg.base_url, skip_ws=True)
    mid_price = float(info.all_mids()[params.coin])
    qty = round(params.size_usd / mid_price, 5)

    # Ordem market com SL/TP
    order_result = exchange.market_open(
        params.coin,
        params.is_buy,
        qty,
        slippage=0.01,
    )
    logger.info("Ordem enviada: %s", order_result)

    # SL via TP/SL order (trigger order)
    sl_order = exchange.order(
        params.coin,
        not params.is_buy,   # lado inverso
        qty,
        params.sl_price,
        {"trigger": {"triggerPx": params.sl_price, "isMarket": True, "tpsl": "sl"}},
        reduce_only=True,
    )
    logger.info("Stop Loss configurado: %s", sl_order)

    tp_order = exchange.order(
        params.coin,
        not params.is_buy,
        qty,
        params.tp_price,
        {"trigger": {"triggerPx": params.tp_price, "isMarket": True, "tpsl": "tp"}},
        reduce_only=True,
    )
    logger.info("Take Profit configurado: %s", tp_order)

    return order_result
