import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Raiz do projeto Kronos — usada para construir paths default dos modelos
# em disco local. `Path(__file__).parent.parent` resolve para a raiz do
# repositório independente do cwd em que o bot for invocado.
PROJECT_ROOT = Path(__file__).parent.parent


# Mapping default coin -> path local em disco dos modelos fine-tunados
# pela savycorp em 15m (BTC/ETH/SOL). Em produção (VPS AWS) os pesos são
# baixados 1 vez via `scripts/download_models.py` e a partir daí o bot
# carrega do disco — sem dependência runtime do Hugging Face Hub.
DEFAULT_MODEL_PATHS: dict[str, str] = {
    "BTC": str(PROJECT_ROOT / "models" / "BTC"),
    "ETH": str(PROJECT_ROOT / "models" / "ETH"),
    "SOL": str(PROJECT_ROOT / "models" / "SOL"),
}


@dataclass
class KronosConfig:
    """Configuração do modelo Kronos.

    Suporta dois modos:
      * Single-asset (legacy): usa `model_name` / `tokenizer_name` (NeoQuasar).
      * Multi-asset: usa `model_paths` (disco local na VPS) com
        `model_subfolder` / `tokenizer_subfolder` apontando para o layout
        savycorp (basemodel/best_model e tokenizer/best_model).
    """

    # --- Campos legacy (single-asset) ---
    model_name: str = "NeoQuasar/Kronos-small"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"

    # --- Campos multi-asset (savycorp BTC/ETH/SOL 15m) ---
    # Mapping coin -> path local em disco. Os pesos são baixados 1 vez via
    # `scripts/download_models.py` na VPS (transferência RunPod->VPS via HF).
    # Use `field(default_factory=...)` para evitar mutable default
    # compartilhado entre instâncias.
    model_paths: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_PATHS)
    )
    # Layout interno dos checkpoints savycorp: o modelo base fica em
    # `basemodel/best_model` e o tokenizer em `tokenizer/best_model`.
    # Esses subfolders são válidos tanto em paths locais quanto em repos HF.
    model_subfolder: str = "basemodel/best_model"
    tokenizer_subfolder: str = "tokenizer/best_model"

    # --- Hiperparâmetros de inferência ---
    device: str = "cpu"          # "cuda:0" se tiver GPU
    max_context: int = 512
    lookback: int = 200          # candles históricos enviados ao modelo
    pred_len: int = 10           # candles a prever
    temperature: float = 1.0
    top_p: float = 0.9
    sample_count: int = 3        # média de N amostras para reduzir ruído


@dataclass
class HyperliquidConfig:
    """Credenciais Hyperliquid carregadas via env vars (segurança VPS).

    Variáveis de ambiente esperadas em modo live:
      HYPERLIQUID_PRIVATE_KEY      — chave da API wallet (NÃO usar wallet principal!)
      HYPERLIQUID_ACCOUNT_ADDRESS  — address da wallet principal (que tem USDC)
      HYPERLIQUID_BASE_URL         — opcional, default api.hyperliquid.xyz

    Em dry_run=True estas podem ficar vazias (não há ordens reais).
    """

    private_key: str = field(
        default_factory=lambda: os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
    )
    account_address: str = field(
        default_factory=lambda: os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS", "")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv("HYPERLIQUID_BASE_URL", "https://api.hyperliquid.xyz")
    )
    mainnet: bool = True


@dataclass
class TradingConfig:
    """Parâmetros de execução de trading.

    Suporta operação single-asset (campo `coin`, legacy) e multi-asset
    (campo `coins`, lista oficial). O campo `dry_run` é mantido em True
    por padrão como salvaguarda — NUNCA mudar este default.
    """

    # --- Single-asset legacy ---
    coin: str = "BTC"            # primary coin para modo single-asset

    # --- Multi-asset (BTC/ETH/SOL em 15m) ---
    # Lista oficial de coins operados pelo bot multi-asset. Usa factory
    # para evitar lista mutável compartilhada entre instâncias.
    coins: list[str] = field(
        default_factory=lambda: ["BTC", "ETH", "SOL"]
    )
    # Limite de posições simultâneas abertas em modo multi-asset.
    max_concurrent_positions: int = 3

    # --- Parâmetros gerais ---
    # Default 15m porque os modelos savycorp foram treinados nesse timeframe;
    # qualquer interval (1m | 5m | 15m | 1h | 4h | 1d) ainda é aceito.
    interval: str = "15m"
    leverage: int = 3
    risk_per_trade_pct: float = 1.0   # % do saldo por trade
    sl_pct: float = 1.5               # stop loss em % do preço de entrada
    tp_pct: float = 3.0               # take profit em % do preço de entrada
    dry_run: bool = True              # True = só loga, não envia ordens
    loop_interval_seconds: int = 3600 # frequência do loop principal


@dataclass
class Config:
    """Container raiz que agrega todas as sub-configurações do bot."""

    kronos: KronosConfig = field(default_factory=KronosConfig)
    hyperliquid: HyperliquidConfig = field(default_factory=HyperliquidConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
