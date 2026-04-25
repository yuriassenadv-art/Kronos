from dataclasses import dataclass
from typing import Optional

from .config import TradingConfig
from .signal_generator import Direction, Signal


@dataclass
class OrderParams:
    coin: str
    is_buy: bool
    size_usd: float
    sl_price: float
    tp_price: float
    leverage: int


def calculate_order(
    signal: Signal,
    account_balance_usd: float,
    cfg: TradingConfig,
) -> Optional[OrderParams]:
    """
    Converte um Signal em parâmetros de ordem.
    Retorna None se o sinal for FLAT ou balanço insuficiente.
    """
    if signal.direction == Direction.FLAT:
        return None

    if account_balance_usd < 10:
        return None

    is_buy = signal.direction == Direction.LONG
    risk_usd = account_balance_usd * (cfg.risk_per_trade_pct / 100.0)
    size_usd = risk_usd * cfg.leverage

    if is_buy:
        sl_price = signal.entry_price * (1 - cfg.sl_pct / 100)
        tp_price = signal.entry_price * (1 + cfg.tp_pct / 100)
    else:
        sl_price = signal.entry_price * (1 + cfg.sl_pct / 100)
        tp_price = signal.entry_price * (1 - cfg.tp_pct / 100)

    return OrderParams(
        coin=cfg.coin,
        is_buy=is_buy,
        size_usd=size_usd,
        sl_price=round(sl_price, 2),
        tp_price=round(tp_price, 2),
        leverage=cfg.leverage,
    )
