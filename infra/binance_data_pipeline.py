"""
Camada 1 — Dados (Binance Perpétuos via CCXT)

Fonte recomendada para fine-tune do Kronos:
  - 7+ anos de histórico (vs 6 meses da Hyperliquid)
  - Perpétuos USDM da Binance = mesma série que o oracle da Hyperliquid rastreia
  - Múltiplos ativos: cada símbolo salvo como CSV separado

Instalação:
    pip install ccxt

Uso:
    # Um ativo:
    python3 -m infra.binance_data_pipeline --symbols BTC --interval 1h --days 1000

    # Múltiplos ativos:
    python3 -m infra.binance_data_pipeline --symbols BTC ETH SOL --interval 1h --days 1000

    # Gera configs de fine-tune automaticamente:
    python3 -m infra.binance_data_pipeline --symbols BTC ETH SOL --gen-configs
"""

import argparse
import time
from pathlib import Path

import pandas as pd

# Binance Futures retorna no máximo 1000 candles por request (CCXT default)
BINANCE_MAX_LIMIT = 1000

# Ativos padrão: perpétuos mais líquidos da Binance (= mais relevantes p/ Hyperliquid)
DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL"]

INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}


# ─────────────────────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────────────────────

def build_exchange():
    """Cria cliente CCXT para Binance Perpétuos USDM (sem autenticação)."""
    try:
        import ccxt
    except ImportError:
        raise SystemExit("ccxt não instalado. Execute: pip install ccxt")

    exchange = ccxt.binance({
        "options": {"defaultType": "future"},   # USDM perpétuos
        "enableRateLimit": True,                # respeita rate limit automaticamente
    })
    exchange.load_markets()
    return exchange


def fetch_symbol_history(
    exchange,
    symbol: str,
    interval: str,
    days: int,
) -> pd.DataFrame:
    """
    Busca `days` dias de candles OHLCV para `symbol` na Binance perpétuos.
    Pagina automaticamente em blocos de 1500 candles.
    """
    ccxt_symbol = f"{symbol}/USDT:USDT"   # formato perpétuo CCXT

    # Verifica se o símbolo existe
    if ccxt_symbol not in exchange.markets:
        raise ValueError(f"Símbolo {ccxt_symbol} não encontrado na Binance. "
                         f"Símbolos disponíveis incluem: BTC/USDT:USDT, ETH/USDT:USDT…")

    end_ms   = int(time.time() * 1000)
    since_ms = end_ms - days * 86_400_000
    all_rows = []

    print(f"\n{'─'*50}")
    print(f"  {symbol}/USDT perp  |  {interval}  |  {days} dias")
    print(f"{'─'*50}")

    while True:
        candles = exchange.fetch_ohlcv(
            ccxt_symbol,
            timeframe=interval,
            since=since_ms,
            limit=BINANCE_MAX_LIMIT,
        )

        if not candles:
            break

        all_rows.extend(candles)
        last_ts = candles[-1][0]
        pct = min((last_ts - (int(time.time() * 1000) - days * 86_400_000)) /
                  (days * 86_400_000) * 100, 100)
        print(f"  {len(all_rows):>7,} candles  [{pct:.0f}%]…", end="\r")

        # Termina quando o último candle alcança o presente
        # (mais robusto que checar contagem, que varia por exchange)
        if last_ts >= end_ms - 1:
            break

        # Avança cursor para o candle seguinte ao último recebido
        since_ms = last_ts + 1

    print(f"  {len(all_rows):>7,} candles  [100%]   ")

    if not all_rows:
        raise RuntimeError(f"Nenhum dado retornado para {symbol}.")

    df = pd.DataFrame(all_rows, columns=["timestamps", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["timestamps"], unit="ms")

    # ── Campo obrigatório para CustomKlineDataset do Kronos ──────────────────
    # amount = volume de cotação (USDT movimentado) = base_volume × close_price
    df["amount"] = df["close"] * df["volume"]
    # ─────────────────────────────────────────────────────────────────────────

    df = df.drop_duplicates("timestamps").sort_values("timestamps").reset_index(drop=True)
    print(f"  Período: {df['timestamps'].iloc[0]}  →  {df['timestamps'].iloc[-1]}")
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]


# ─────────────────────────────────────────────────────────────────────────────
# Salvar
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    out_dir = Path(__file__).parent.parent / "finetune_csv" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"BN_{symbol}_{interval}.csv"
    df.to_csv(path, index=False)
    print(f"  Salvo → {path}  ({len(df):,} linhas)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Geração automática de configs de fine-tune
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_TEMPLATE = """\
# Fine-tune Kronos em {symbol}/USDT perpétuo da Binance
# Gerado automaticamente por binance_data_pipeline.py

data:
  data_path: "finetune_csv/data/BN_{symbol}_{interval}.csv"
  lookback_window: 512
  predict_window: 24
  max_context: 512
  clip: 5.0
  train_ratio: 0.85
  val_ratio: 0.10
  test_ratio: 0.05

training:
  tokenizer_epochs: 30
  basemodel_epochs: 20
  batch_size: 32
  log_interval: 50
  num_workers: 4
  seed: 42
  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000001
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1
  accumulation_steps: 2

model_paths:
  pretrained_tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
  pretrained_predictor: "NeoQuasar/Kronos-small"
  exp_name: "BN_{symbol}_{interval}"
  base_path: "finetune_csv/finetuned"
  base_save_path: ""
  finetuned_tokenizer: ""
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "kronos_bn_{symbol_lower}_{interval}"
  description: "Kronos fine-tuned on Binance {symbol}/USDT perpetual {interval}"
  use_comet: false
  train_tokenizer: true
  train_basemodel: true
  skip_existing: false
  pre_trained_tokenizer: true
  pre_trained_predictor: true

device:
  use_cuda: true
  device_id: 0
"""


def generate_config(symbol: str, interval: str) -> Path:
    cfg_dir = Path(__file__).parent.parent / "finetune_csv" / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"config_bn_{symbol.lower()}_{interval}.yaml"
    path.write_text(CONFIG_TEMPLATE.format(
        symbol=symbol,
        symbol_lower=symbol.lower(),
        interval=interval,
    ))
    print(f"  Config gerado → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baixa histórico de perpétuos Binance para fine-tune do Kronos"
    )
    parser.add_argument("--symbols",    nargs="+", default=DEFAULT_SYMBOLS,
                        metavar="SYM",
                        help=f"Símbolos base (padrão: {' '.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--interval",   default="1h", choices=list(INTERVAL_MAP),
                        help="Timeframe dos candles (padrão: 1h)")
    parser.add_argument("--days",       type=int, default=1000,
                        help="Dias de histórico (padrão: 1000 ≈ 2.7 anos)")
    parser.add_argument("--gen-configs", action="store_true",
                        help="Gera arquivos YAML de fine-tune automaticamente")
    args = parser.parse_args()

    print("=" * 50)
    print("  Binance Perps → Kronos Dataset Builder")
    print("=" * 50)
    print(f"  Símbolos : {', '.join(args.symbols)}")
    print(f"  Intervalo: {args.interval}")
    print(f"  Histórico: {args.days} dias")
    print()

    exchange = build_exchange()
    saved = []

    for symbol in args.symbols:
        try:
            df = fetch_symbol_history(exchange, symbol, args.interval, args.days)
            path = save_csv(df, symbol, args.interval)
            saved.append((symbol, path))

            if args.gen_configs:
                generate_config(symbol, args.interval)

        except Exception as exc:
            print(f"  ⚠ Erro em {symbol}: {exc}")

    print("\n" + "=" * 50)
    print("  Concluído!")
    for symbol, path in saved:
        print(f"  ✓ {symbol:6s} → {path.name}")

    if args.gen_configs and saved:
        print()
        print("  Para treinar (um por vez):")
        for symbol, _ in saved:
            print(f"    python3 finetune_csv/train_sequential.py "
                  f"--config finetune_csv/configs/config_bn_{symbol.lower()}_{args.interval}.yaml")
    print("=" * 50)


if __name__ == "__main__":
    main()
