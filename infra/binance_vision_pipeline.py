"""
Camada 1 — Dados (Binance Vision: dumps oficiais públicos)

Alternativa ao CCXT que NÃO sofre geo-block.
Fonte: https://data.binance.vision/data/futures/um/  (CDN público da Binance)

Estratégia de partição:
  - Mês corrente  → ZIPs DIÁRIOS (data.binance.vision/.../daily/klines/...)
  - Demais meses  → ZIPs MENSAIS (data.binance.vision/.../monthly/klines/...)

Vantagens vs CCXT/REST:
  - Sem geo-block (funciona de US, Dubai, qualquer lugar)
  - Sem rate-limit
  - Dados oficiais Binance
  - Download paralelo (ThreadPool)

Uso:
    python3 -m infra.binance_vision_pipeline --symbols BTC --interval 15m --days 2000
    python3 -m infra.binance_vision_pipeline --symbols BTC ETH --interval 15m --days 2000 --gen-configs
"""

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/futures/um"

# Header oficial dos ZIPs de klines da Binance Vision (12 colunas)
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

VALID_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}


# ─────────────────────────────────────────────────────────────────────────────
# Partição de datas (mensal vs diário)
# ─────────────────────────────────────────────────────────────────────────────

def partition_dates(today: date, days_back: int):
    """
    Particiona o intervalo [today - days_back, today - 1] em:
      - faixa mensal: do início do mês mais antigo até o fim do mês anterior ao corrente
      - faixa diária: do dia 1 do mês corrente até ontem

    Retorna (monthly_months, daily_dates) onde:
      monthly_months = [(y, m), ...] em ordem cronológica
      daily_dates    = [date(y, m, d), ...] em ordem cronológica
    """
    start = today - timedelta(days=days_back)

    # Faixa mensal: arredonda start pra início do seu mês,
    # vai até o fim do mês ANTERIOR ao corrente.
    last_closed_month_end = date(today.year, today.month, 1) - timedelta(days=1)

    monthly_months = []
    y, m = start.year, start.month
    while (y, m) <= (last_closed_month_end.year, last_closed_month_end.month):
        monthly_months.append((y, m))
        m += 1
        if m == 13:
            m, y = 1, y + 1

    # Faixa diária: dia 1 do mês corrente até ontem.
    first_of_current = date(today.year, today.month, 1)
    yesterday = today - timedelta(days=1)
    daily_dates = []
    d = first_of_current
    while d <= yesterday:
        daily_dates.append(d)
        d += timedelta(days=1)

    return monthly_months, daily_dates


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def monthly_url(symbol: str, interval: str, year: int, month: int) -> str:
    pair = f"{symbol}USDT"
    return f"{BASE_URL}/monthly/klines/{pair}/{interval}/{pair}-{interval}-{year:04d}-{month:02d}.zip"


def daily_url(symbol: str, interval: str, d: date) -> str:
    pair = f"{symbol}USDT"
    return f"{BASE_URL}/daily/klines/{pair}/{interval}/{pair}-{interval}-{d.strftime('%Y-%m-%d')}.zip"


def fetch_zip_to_df(url: str, session: requests.Session) -> pd.DataFrame | None:
    """Baixa 1 ZIP e retorna DataFrame. Retorna None se 404 (ZIP não publicado ainda)."""
    try:
        r = session.get(url, timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                head = f.readline().decode("utf-8", errors="ignore").strip()
                f.seek(0)
                # ZIPs antigos (pré-2024) não têm header; novos têm.
                has_header = head.startswith("open_time")
                df = pd.read_csv(
                    f,
                    header=0 if has_header else None,
                    names=KLINE_COLS,
                )
        return df

    except requests.HTTPError as e:
        print(f"  ⚠ HTTP {e.response.status_code} em {url.rsplit('/', 1)[-1]}")
        return None
    except Exception as e:
        print(f"  ⚠ Falha em {url.rsplit('/', 1)[-1]}: {type(e).__name__}: {e}")
        return None


def fetch_history(symbol: str, interval: str, days: int, max_workers: int = 8) -> pd.DataFrame:
    today = date.today()
    monthly_months, daily_dates = partition_dates(today, days)

    monthly_urls = [monthly_url(symbol, interval, y, m) for y, m in monthly_months]
    daily_urls = [daily_url(symbol, interval, d) for d in daily_dates]

    print(f"  Mensais : {len(monthly_urls):>4}  "
          f"({monthly_months[0][0]}-{monthly_months[0][1]:02d} → "
          f"{monthly_months[-1][0]}-{monthly_months[-1][1]:02d})")
    print(f"  Diários : {len(daily_urls):>4}  "
          f"({daily_dates[0]} → {daily_dates[-1]})")
    print(f"  Total   : {len(monthly_urls) + len(daily_urls)} ZIPs")
    print()

    all_urls = monthly_urls + daily_urls
    dfs: list[pd.DataFrame] = []

    session = requests.Session()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_zip_to_df, u, session): u for u in all_urls}
        done = 0
        for fut in as_completed(futures):
            df = fut.result()
            done += 1
            if df is not None and not df.empty:
                dfs.append(df)
            print(f"  Baixado {done:>4}/{len(all_urls)}  "
                  f"({len(dfs)} OK, {done - len(dfs)} 404/fail)", end="\r")
    print()

    if not dfs:
        raise RuntimeError(f"Nenhum dado baixado para {symbol}/{interval}.")

    combined = pd.concat(dfs, ignore_index=True)
    combined = (combined
                .drop_duplicates("open_time")
                .sort_values("open_time")
                .reset_index(drop=True))

    # Converte para o formato exigido pelo CustomKlineDataset do Kronos
    out = pd.DataFrame({
        "timestamps": pd.to_datetime(combined["open_time"], unit="ms"),
        "open":   combined["open"].astype(float),
        "high":   combined["high"].astype(float),
        "low":    combined["low"].astype(float),
        "close":  combined["close"].astype(float),
        "volume": combined["volume"].astype(float),
    })
    # amount = volume de cotação (USDT). Mesmo cálculo dos CSVs BN_*_1h existentes.
    out["amount"] = out["close"] * out["volume"]

    print(f"  Período : {out['timestamps'].iloc[0]}  →  {out['timestamps'].iloc[-1]}")
    print(f"  Candles : {len(out):,}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Save + Config
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    out_dir = Path(__file__).parent.parent / "finetune_csv" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"BN_{symbol}_{interval}.csv"
    df.to_csv(path, index=False)
    print(f"  Salvo   : {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


# predict_window por timeframe (em barras): mantém ~24h de previsão
PREDICT_WINDOW = {
    "1m": 60, "3m": 20, "5m": 12, "15m": 96, "30m": 48,
    "1h": 24, "4h": 6, "1d": 7,
}

CONFIG_TEMPLATE = """\
# Fine-tune Kronos em {symbol}/USDT perpétuo Binance — {interval}
# Gerado por binance_vision_pipeline.py (sem geo-block, fonte: data.binance.vision)

data:
  data_path: "finetune_csv/data/BN_{symbol}_{interval}.csv"
  lookback_window: 512
  predict_window: {predict_window}
  max_context: 512
  clip: 5.0
  train_ratio: 0.85
  val_ratio: 0.10
  test_ratio: 0.05

training:
  tokenizer_epochs: 30
  basemodel_epochs: 20
  batch_size: 64
  log_interval: 50
  num_workers: 4
  seed: 42
  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000001
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1
  accumulation_steps: 1

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
  description: "Kronos fine-tuned on Binance {symbol}/USDT perpetual {interval} (Binance Vision)"
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
        predict_window=PREDICT_WINDOW.get(interval, 24),
    ))
    print(f"  Config  : {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baixa histórico de perpétuos USDM da Binance via data.binance.vision"
    )
    parser.add_argument("--symbols", nargs="+", default=["BTC"], metavar="SYM")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--days", type=int, default=2000,
                        help="Dias de histórico (padrão: 2000 ≈ 5.5 anos)")
    parser.add_argument("--gen-configs", action="store_true",
                        help="Gera arquivos YAML de fine-tune automaticamente")
    parser.add_argument("--workers", type=int, default=8,
                        help="Threads de download (padrão: 8)")
    args = parser.parse_args()

    if args.interval not in VALID_INTERVALS:
        raise SystemExit(f"Intervalo inválido. Use um de: {sorted(VALID_INTERVALS)}")

    print("=" * 60)
    print("  Binance Vision → Kronos Dataset Builder")
    print("=" * 60)
    print(f"  Símbolos : {', '.join(args.symbols)}")
    print(f"  Intervalo: {args.interval}")
    print(f"  Histórico: {args.days} dias")
    print(f"  Workers  : {args.workers}")
    print()

    saved: list[tuple[str, Path]] = []
    for symbol in args.symbols:
        print(f"── {symbol} " + "─" * (60 - len(symbol) - 4))
        try:
            df = fetch_history(symbol, args.interval, args.days, args.workers)
            path = save_csv(df, symbol, args.interval)
            saved.append((symbol, path))
            if args.gen_configs:
                generate_config(symbol, args.interval)
        except Exception as exc:
            print(f"  ❌ Erro em {symbol}: {exc}")

    print("\n" + "=" * 60)
    print("  Concluído!")
    for symbol, path in saved:
        print(f"  ✓ {symbol:6s} → {path.name}")
    if args.gen_configs and saved:
        print("\n  Para treinar:")
        for symbol, _ in saved:
            print(f"    python3 finetune_csv/train_sequential.py "
                  f"--config finetune_csv/configs/config_bn_{symbol.lower()}_{args.interval}.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
