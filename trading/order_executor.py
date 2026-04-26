"""
Executa ordens na Hyperliquid via hyperliquid-python-sdk.

Instale: pip install hyperliquid-python-sdk
"""

import logging
import math
from typing import Optional

from hyperliquid.info import Info

from infra import monitor
from infra.retry import retry_on_network_error

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


@retry_on_network_error(max_attempts=3, base_delay=1.0)
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


def _round_price(price: float, sz_decimals: int) -> float:
    """
    Arredonda preço conforme regras de tick da Hyperliquid:
      - max (6 - sz_decimals) decimal places
      - max 5 significant figures

    A regra mais restritiva (menor precisão) ganha. Sem esse arredondamento
    a exchange rejeita SL/TP com `{"error": "Order has invalid price."}`
    para ativos cujo szDecimals não permite a precisão default que o
    Python `round(x, 2)` produz (ex: BTC szDecimals=5 → max 1 decimal).
    """
    if price <= 0:
        return 0.0

    max_decimals = max(0, 6 - sz_decimals)

    # Calcula precisão por sig figs: 5 sig figs total, parte inteira já
    # consome (magnitude+1) dígitos, sobra (4 - magnitude) para os decimais.
    magnitude = int(math.floor(math.log10(abs(price))))
    sig_fig_decimals = max(0, 4 - magnitude)

    # A regra mais restritiva ganha
    precision = min(max_decimals, sig_fig_decimals)
    return round(price, precision)


def clear_decimals_cache() -> None:
    """Reset do cache de szDecimals — útil em testes."""
    global _SZ_DECIMALS_CACHE
    _SZ_DECIMALS_CACHE = None


@retry_on_network_error(max_attempts=3, base_delay=1.0)
def get_account_balance(hl_cfg: HyperliquidConfig) -> float:
    """
    Retorna capital LIVRE disponível para abrir novas posições.

    Suporta dois modos da Hyperliquid (sem double-counting):
      - Classic Account: capital livre vive em `user_state.withdrawable`
        (perp account). `spot_user_state` retorna 0.
      - Unified Account sem posições abertas: spot USDC é o saldo real;
        `withdrawable` retorna 0.
      - Unified Account com posições: spot USDC continua representando o
        capital alocado (o débito real só acontece em PnL/fees), e
        `withdrawable` reflete margem após alocação.

    Estratégia: pega o MAIOR entre `withdrawable` e `spot_usdc`.
      * Em Classic, withdrawable > 0 e spot_usdc = 0 → vence withdrawable.
      * Em Unified sem posições, withdrawable = 0 e spot_usdc > 0 → vence spot.
      * Em Unified com posições, withdrawable < spot_usdc (parte da margem
        está alocada) → vence spot_usdc, que é o capital real disponível.

    Sem essa lógica, somar ambos resultaria em double-counting em Unified
    Account (smoke test: $19.98 spot + $10.84 perp = $30.82 falso).
    """
    info = Info(hl_cfg.base_url, skip_ws=True)

    state = info.user_state(hl_cfg.account_address)
    withdrawable = float(state.get("withdrawable", 0))

    # Spot USDC (Unified mode usa esse como margem)
    try:
        spot = info.spot_user_state(hl_cfg.account_address)
        spot_usdc = sum(
            float(b.get("total", 0))
            for b in spot.get("balances", [])
            if b.get("coin") == "USDC"
        )
    except Exception as exc:
        logger.warning("Falha ao consultar spot_user_state: %s", exc)
        spot_usdc = 0.0

    # Pega o maior — evita double-count em Unified Account
    available = max(withdrawable, spot_usdc)
    logger.debug(
        "Balance: withdrawable=$%.2f | spot_usdc=$%.2f | available=$%.2f",
        withdrawable, spot_usdc, available,
    )
    return available


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

    # ─── Validação de fill (Gap #4) ───────────────────────────────────────
    # Antes de enviar SL/TP ou persistir state, validamos:
    #   1. status global == "ok"
    #   2. pelo menos 1 status com "filled" no payload de retorno
    # Caso contrário, abortamos cedo e alertamos via Telegram.
    if order_result.get("status") != "ok":
        logger.error(
            "[%s] Ordem REJEITADA — status=%s | response=%s",
            params.coin, order_result.get("status"), order_result,
        )
        monitor.alert_error(
            f"order_rejected_{params.coin}",
            f"Hyperliquid rejeitou: {order_result}",
        )
        return None

    statuses = (
        order_result.get("response", {})
        .get("data", {})
        .get("statuses", [])
    )
    filled = [s for s in statuses if isinstance(s, dict) and "filled" in s]
    if not filled:
        logger.error(
            "[%s] Ordem retornou ok sem fill — statuses=%s",
            params.coin, statuses,
        )
        monitor.alert_error(
            f"order_no_fill_{params.coin}",
            f"Sem fill: {statuses}",
        )
        return None

    # Extrair preço médio realmente executado para refletir slippage real
    avg_px = float(filled[0]["filled"].get("avgPx", mid_price))
    filled_qty = float(filled[0]["filled"].get("totalSz", qty))
    logger.info(
        "[%s] Fill confirmado: avg_px=%.4f | qty=%.6f",
        params.coin, avg_px, filled_qty,
    )

    # Recalcular SL/TP sobre avg_px real (em vez de signal.entry_price)
    # para refletir slippage real:
    if params.is_buy:
        actual_sl = avg_px * (1 - trading_cfg.sl_pct / 100)
        actual_tp = avg_px * (1 + trading_cfg.tp_pct / 100)
    else:
        actual_sl = avg_px * (1 + trading_cfg.sl_pct / 100)
        actual_tp = avg_px * (1 - trading_cfg.tp_pct / 100)

    # Arredondar SL/TP segundo a regra de tick da Hyperliquid (Bug #1):
    # round(x, 2) genérico quebra para coins com szDecimals alto (BTC=5
    # → max 1 decimal). Sem isso, a exchange devolve
    # `{"error": "Order has invalid price."}` e perdemos as proteções.
    sz_decimals = _load_sz_decimals(hl_cfg.base_url).get(params.coin, 4)
    params.sl_price = _round_price(actual_sl, sz_decimals)
    params.tp_price = _round_price(actual_tp, sz_decimals)

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
