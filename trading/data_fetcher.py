import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from infra.retry import retry_on_network_error

from .config import TradingConfig

INTERVAL_TO_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


@retry_on_network_error(max_attempts=3, base_delay=1.0)
def fetch_candles(
    coin: str,
    interval: str,
    lookback: int,
    base_url: str = "https://api.hyperliquid.xyz",
) -> pd.DataFrame:
    """
    Busca os últimos `lookback` candles do par `coin` na Hyperliquid.

    Retorna DataFrame com colunas: timestamps, open, high, low, close, volume
    prontas para passar ao KronosPredictor.
    """
    interval_ms = INTERVAL_TO_MS.get(interval)
    if interval_ms is None:
        raise ValueError(f"Intervalo '{interval}' inválido. Use: {list(INTERVAL_TO_MS)}")

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (lookback + 5) * interval_ms  # +5 de margem

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }

    resp = requests.post(f"{base_url}/info", json=payload, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    if not raw:
        raise RuntimeError(f"Hyperliquid retornou resposta vazia para {coin}/{interval}")

    # Formato retornado: lista de dicts com chaves t, o, h, l, c, v, n
    df = pd.DataFrame(raw)
    df = df.rename(columns={"t": "timestamps", "o": "open", "h": "high",
                             "l": "low", "c": "close", "v": "volume"})

    df["timestamps"] = pd.to_datetime(df["timestamps"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    df = df.sort_values("timestamps").reset_index(drop=True)
    df = df.tail(lookback).reset_index(drop=True)

    return df[["timestamps", "open", "high", "low", "close", "volume"]]


def build_kronos_inputs(df: pd.DataFrame, pred_len: int, interval: str):
    """
    Divide o DataFrame em (x_df, x_timestamp, y_timestamp) para o KronosPredictor.
    y_timestamp é gerado sinteticamente como futuro imediato.
    """
    x_df = df[["open", "high", "low", "close", "volume"]].copy()
    x_timestamp = df["timestamps"].copy()

    interval_ms = INTERVAL_TO_MS[interval]
    last_ts = df["timestamps"].iloc[-1]
    y_timestamps = [
        last_ts + timedelta(milliseconds=interval_ms * (i + 1))
        for i in range(pred_len)
    ]
    y_timestamp = pd.Series(y_timestamps)

    return x_df, x_timestamp, y_timestamp
