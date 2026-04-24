from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KronosConfig:
    model_name: str = "NeoQuasar/Kronos-small"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    device: str = "cpu"          # "cuda:0" se tiver GPU
    max_context: int = 512
    lookback: int = 200          # candles históricos enviados ao modelo
    pred_len: int = 10           # candles a prever
    temperature: float = 1.0
    top_p: float = 0.9
    sample_count: int = 3        # média de N amostras para reduzir ruído


@dataclass
class HyperliquidConfig:
    # Deixe private_key vazio para modo dry-run (sem ordens reais)
    private_key: str = ""
    account_address: str = ""
    base_url: str = "https://api.hyperliquid.xyz"
    mainnet: bool = True


@dataclass
class TradingConfig:
    coin: str = "BTC"
    interval: str = "1h"         # 1m | 5m | 15m | 1h | 4h | 1d
    leverage: int = 3
    risk_per_trade_pct: float = 1.0   # % do saldo por trade
    sl_pct: float = 1.5               # stop loss em % do preço de entrada
    tp_pct: float = 3.0               # take profit em % do preço de entrada
    dry_run: bool = True              # True = só loga, não envia ordens
    loop_interval_seconds: int = 3600 # frequência do loop principal


@dataclass
class Config:
    kronos: KronosConfig = field(default_factory=KronosConfig)
    hyperliquid: HyperliquidConfig = field(default_factory=HyperliquidConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
