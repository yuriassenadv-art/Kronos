"""
Camada 1 — Dados

Busca histórico da Hyperliquid e salva em CSV no formato
exato que o finetune_csv/ do Kronos espera:
  timestamps, open, high, low, close, volume

Uso:
    python3 -m infra.crypto_data_pipeline --coin BTC --interval 1h --days 180
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

INTERVAL_TO_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

# Janela segura por batch: 1000 candles evita buracos em ranges históricos
BATCH_CANDLES = 1000
BASE_URL = "https://api.hyperliquid.xyz"


def fetch_batch(coin: str, interval: str, start_ms: int, end_ms: int) -> list:
    payload = {
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval,
                "startTime": start_ms, "endTime": end_ms},
    }
    resp = requests.post(f"{BASE_URL}/info", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json() or []


def fetch_full_history(coin: str, interval: str, days: int) -> pd.DataFrame:
    """
    Busca `days` dias de candles em batches de 1000 e retorna DataFrame.

    Nota: a Hyperliquid tem histórico desde ~Set/2025.
    Para 1h recomenda-se days <= 180.
    """
    interval_ms = INTERVAL_TO_MS[interval]
    batch_window_ms = BATCH_CANDLES * interval_ms

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000

    all_rows = []
    cursor = start_ms
    empty_batches = 0
    total_batches = (days * 86_400_000) // batch_window_ms + 1

    print(f"Buscando {days} dias de {coin}/{interval} em batches de {BATCH_CANDLES}…")

    while cursor < end_ms:
        batch_end = min(cursor + batch_window_ms, end_ms)
        rows = fetch_batch(coin, interval, cursor, batch_end)

        if not rows:
            # Avança o cursor mesmo em janelas vazias (dados históricos ausentes)
            empty_batches += 1
            cursor = batch_end + interval_ms
            if empty_batches > 20:
                # Muitos batches vazios consecutivos — dados não disponíveis
                print(f"\nAtenção: {empty_batches} batches vazios consecutivos. "
                      f"Histórico disponível a partir de {all_rows[0]['t'] if all_rows else 'N/A'}.")
                break
            continue

        empty_batches = 0  # reseta contador ao encontrar dados
        all_rows.extend(rows)
        cursor = rows[-1]["t"] + interval_ms
        pct = min((cursor - start_ms) / (end_ms - start_ms) * 100, 100)
        print(f"  {len(all_rows):,} candles  [{pct:.0f}%]…", end="\r")
        time.sleep(0.15)

    print(f"\nTotal coletado: {len(all_rows):,} candles")

    if not all_rows:
        raise RuntimeError(
            f"Nenhum dado retornado para {coin}/{interval}. "
            f"Tente reduzir --days (ex: --days 180)."
        )

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"t": "timestamps", "o": "open", "h": "high",
                             "l": "low",  "c": "close", "v": "volume"})
    df["timestamps"] = pd.to_datetime(df["timestamps"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    df = df.drop_duplicates("timestamps").sort_values("timestamps").reset_index(drop=True)
    print(f"Período: {df['timestamps'].iloc[0]}  →  {df['timestamps'].iloc[-1]}")
    return df[["timestamps", "open", "high", "low", "close", "volume"]]


def save_dataset(df: pd.DataFrame, coin: str, interval: str) -> Path:
    out_dir = Path(__file__).parent.parent / "finetune_csv" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"HL_{coin}_{interval}.csv"
    df.to_csv(path, index=False)
    print(f"Salvo: {path}  ({len(df):,} linhas)")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin",     default="BTC")
    parser.add_argument("--interval", default="1h",
                        choices=list(INTERVAL_TO_MS))
    parser.add_argument("--days",     type=int, default=180,
                        help="Dias de histórico (HL disponível desde ~Set/2025; padrão: 180)")
    args = parser.parse_args()

    df = fetch_full_history(args.coin, args.interval, args.days)
    save_dataset(df, args.coin, args.interval)


if __name__ == "__main__":
    main()
