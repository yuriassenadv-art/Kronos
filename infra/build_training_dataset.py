"""
Domain-adaptive training dataset builder.

Combines a long Binance history (broad pattern coverage) with a shorter
Hyperliquid history (exact target-exchange distribution) by:
  1. Loading BN_<SYMBOL>_<INTERVAL>.csv  (has `amount` column already)
  2. Loading HL_<SYMBOL>_<INTERVAL>.csv  (usually lacks `amount`)
     For intervals not natively supported by HL (e.g. 10m), a finer
     HL file is resampled: HL_<SYMBOL>_<HL_SRC>.csv → target interval.
  3. Removing Binance rows that overlap with the HL window — HL prices
     are authoritative for that period and must not be duplicated
  4. Computing amount = close * volume for any CSV that lacks it
  5. Concatenating [BN_pre_HL] + [HL] in chronological order
  6. Saving as COMBINED_<SYMBOL>_<INTERVAL>.csv
  7. Auto-generating finetune_csv/configs/config_combined_<symbol>_<interval>.yaml

Usage:
    python3 -m infra.build_training_dataset --symbol BTC --interval 1h
    python3 -m infra.build_training_dataset --symbol BTC --interval 10m
    python3 -m infra.build_training_dataset --symbol ETH --interval 1h
"""

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

REQUIRED_COLS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]

DATA_DIR = Path("finetune_csv/data")
CONFIG_DIR = Path("finetune_csv/configs")

# Hyperliquid native intervals. If the requested interval is not in this set,
# we resample from the nearest finer native interval.
HL_NATIVE = {"1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"}

# Source interval used when resampling HL data for non-native intervals
HL_RESAMPLE_SRC = {
    "10m": "5m",
    "2m":  "1m",
    "6m":  "3m",
    "20m": "5m",
    "45m": "15m",
}

# predict_window (candles) per interval — how far ahead to forecast
PREDICT_WINDOW = {
    "1m": 30, "3m": 20, "5m": 12, "10m": 6,
    "15m": 8, "30m": 6, "1h": 24, "4h": 12, "1d": 5,
}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    return df[REQUIRED_COLS]


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a OHLCV+amount DataFrame to a coarser frequency."""
    df = df.set_index("timestamps")
    resampled = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
        "amount": "sum",
    }).dropna(subset=["open"]).reset_index()
    return resampled[REQUIRED_COLS]


def _load_hl(symbol: str, interval: str) -> pd.DataFrame:
    """Load HL data, resampling from a finer interval if needed."""
    hl_path = DATA_DIR / f"HL_{symbol}_{interval}.csv"
    if hl_path.exists():
        return _load(hl_path)

    src = HL_RESAMPLE_SRC.get(interval)
    if src is None:
        sys.exit(
            f"HL CSV not found: {hl_path}\n"
            f"Interval '{interval}' is not natively supported by Hyperliquid "
            f"and has no resample source defined in HL_RESAMPLE_SRC."
        )

    src_path = DATA_DIR / f"HL_{symbol}_{src}.csv"
    if not src_path.exists():
        sys.exit(
            f"HL CSV not found: {hl_path}\n"
            f"Resample source also missing: {src_path}\n"
            f"Run first: python3 -m infra.crypto_data_pipeline --coin {symbol} --interval {src}"
        )

    print(f"  HL {interval}: resampling from {src_path.name}…")
    raw = _load(src_path)
    # pandas resample rule: "10min", "5min", etc.
    rule = interval.replace("m", "min").replace("h", "h").replace("d", "D")
    resampled = _resample_ohlcv(raw, rule)
    # Cache the resampled file for future runs
    resampled.to_csv(hl_path, index=False)
    print(f"  HL {interval}: {len(resampled):,} candles (cached → {hl_path.name})")
    return resampled


def build(symbol: str, interval: str) -> Path:
    symbol = symbol.upper()
    bn_path = DATA_DIR / f"BN_{symbol}_{interval}.csv"

    if not bn_path.exists():
        sys.exit(f"Binance CSV not found: {bn_path}")

    bn = _load(bn_path)
    hl = _load_hl(symbol, interval)

    hl_start = hl["timestamps"].min()

    # Binance rows that come strictly BEFORE the HL window
    bn_pre = bn[bn["timestamps"] < hl_start].copy()

    combined = pd.concat([bn_pre, hl], ignore_index=True)
    combined = combined.sort_values("timestamps").reset_index(drop=True)

    out_csv = DATA_DIR / f"COMBINED_{symbol}_{interval}.csv"
    combined.to_csv(out_csv, index=False)

    n_bn = len(bn_pre)
    n_hl = len(hl)
    print(
        f"[build_training_dataset] {symbol}/{interval}\n"
        f"  BN pre-HL : {n_bn:,} candles  ({bn['timestamps'].min().date()} → {bn_pre['timestamps'].max().date()})\n"
        f"  HL        : {n_hl:,} candles  ({hl_start.date()} → {hl['timestamps'].max().date()})\n"
        f"  Combined  : {len(combined):,} candles  → {out_csv}"
    )

    _write_config(symbol, interval, out_csv)
    return out_csv


def _write_config(symbol: str, interval: str, csv_path: Path):
    exp_name = f"COMBINED_{symbol}_{interval}"
    predict_window = PREDICT_WINDOW.get(interval, 24)
    cfg = {
        "data": {
            "data_path": str(csv_path),
            "lookback_window": 512,
            "predict_window": predict_window,
            "max_context": 512,
            "clip": 5.0,
            "train_ratio": 0.85,
            "val_ratio": 0.10,
            "test_ratio": 0.05,
        },
        "training": {
            "tokenizer_epochs": 30,
            "basemodel_epochs": 20,
            "batch_size": 32,
            "log_interval": 50,
            "num_workers": 4,
            "seed": 42,
            "tokenizer_learning_rate": 0.0002,
            "predictor_learning_rate": 0.000001,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_weight_decay": 0.1,
            "accumulation_steps": 2,
        },
        "model_paths": {
            "pretrained_tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
            "pretrained_predictor": "NeoQuasar/Kronos-small",
            "exp_name": exp_name,
            "base_path": "finetune_csv/finetuned",
            "base_save_path": "",
            "finetuned_tokenizer": "",
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel",
        },
        "experiment": {
            "name": f"kronos_{exp_name.lower()}",
            "description": (
                f"Kronos fine-tuned on combined Binance+Hyperliquid "
                f"{symbol} {interval} dataset (domain adaptation)"
            ),
            "use_comet": False,
            "train_tokenizer": True,
            "train_basemodel": True,
            "skip_existing": False,
            "pre_trained_tokenizer": True,
            "pre_trained_predictor": True,
        },
        "device": {
            "use_cuda": True,
            "device_id": 0,
        },
    }

    out_yaml = CONFIG_DIR / f"config_combined_{symbol.lower()}_{interval}.yaml"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        header = textwrap.dedent(f"""\
            # Auto-generated by infra/build_training_dataset.py
            # Strategy: domain adaptation — Binance (long history) + Hyperliquid (target exchange)
            # Binance rows prior to the HL window are prepended for broad pattern coverage;
            # Hyperliquid rows cover the recent period with exact target-exchange prices.
            #
            # To retrain: python3 -m infra.build_training_dataset --symbol {symbol} --interval {interval}
            #             python3 finetune_csv/train_sequential.py --config {out_yaml}
            """)
        f.write(header + "\n")
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, indent=2)

    print(f"  Config    : {out_yaml}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a domain-adaptive combined training CSV from Binance + Hyperliquid data."
    )
    parser.add_argument("--symbol",   default="BTC", help="Symbol (BTC, ETH, SOL)")
    parser.add_argument("--interval", default="1h",  help="Candlestick interval (1h, 4h, …)")
    args = parser.parse_args()
    build(args.symbol, args.interval)


if __name__ == "__main__":
    main()
