"""
Executa ordens na Hyperliquid via hyperliquid-python-sdk.

Instale: pip install hyperliquid-python-sdk
"""

import logging
from typing import Optional

from hyperliquid.info import Info

from .config import HyperliquidConfig, TradingConfig
from .risk_manager import OrderParams

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Size precision por coin (BLOCKER #3)
# ─────────────────────────────────────────────────────────────────────────────
# A Hyperliquid expõe os decimais de quantidade (`szDecimals`) por ativo via
# `Info.meta()["universe"]`. Cada coin tem um valor diferente:
#   BTC=5, ETH=4, SOL=2, …
# Hardcodar `5` decimais na conversão `qty = round(size_usd / mid_price, 5)`
# quebra para coins com tick maior (ex: SOL com 0.001 SOL é INVÁLIDO e a
# exchange rejeita a ordem). Cacheamos o mapping em memória para evitar
# fetch repetido a cada ordem.
_SZ_DECIMALS_CACHE: Optional[dict[str, int]] = None


def _load_sz_decimals(base_url: str) -> dict[str, int]:
    """
    Carrega o mapping coin -> szDecimals da Hyperliquid `meta` endpoint.
    Cacheado em memória para evitar fetch repetido a cada ordem.
    """
    global _SZ_DECIMALS_CACHE
    if _SZ_DECIMALS_CACHE is not None:
        return _SZ_DECIMALS_CACHE

    info = Info(base_url, skip_ws=True)
    meta = info.meta()
    _SZ_DECIMALS_CACHE = {
        asset["name"]: int(asset["szDecimals"])
        for asset in meta.get("universe", [])
    }
    return _SZ_DECIMALS_CACHE


def _round_qty(coin: str, qty_raw: float, base_url: str) -> float:
    """
    Arredonda quantidade para o szDecimals do ativo.
    Fallback: 4 decimais (mais conservador que 5).
    """
    decimals = _load_sz_decimals(base_url).get(coin, 4)
    return round(qty_raw, decimals)


def clear_decimals_cache() -> None:
    """Reset do cache de szDecimals — útil em testes."""
    global _SZ_DECIMALS_CACHE
    _SZ_DECIMALS_CACHE = None


def get_account_balance(hl_cfg: HyperliquidConfig) -> float:
    """Retorna o saldo USDC disponível na conta."""
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
    info = Info(hl_cfg.base_url, skip_ws=True)
    mid_price = float(info.all_mids()[params.coin])
    qty = _round_qty(params.coin, params.size_usd / mid_price, hl_cfg.base_url)

    # Validação preventiva: se o size_usd é menor que 1 unidade do tick
    # (qty arredonda para 0), abortamos a ordem com warning. Sem isso, a
    # exchange rejeitaria com mensagem genérica e perderíamos o ciclo.
    if qty == 0:
        decimals_used = _load_sz_decimals(hl_cfg.base_url).get(params.coin, 4)
        logger.warning(
            "Ordem abortada: qty arredondada para 0 | coin=%s "
            "size_usd=$%.2f mid_price=%.2f szDecimals=%d. "
            "Aumente size_usd ou reduza precisão.",
            params.coin,
            params.size_usd,
            mid_price,
            decimals_used,
        )
        return None

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
