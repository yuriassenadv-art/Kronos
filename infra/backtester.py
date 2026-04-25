"""
Camada 3 — Validação

Walk-forward backtest: simula o loop de trading no histórico,
candle a candle, sem lookahead. Serve para:
  - Calibrar o threshold de entrada (signal_generator.py)
  - Medir win rate, sharpe, drawdown
  - Decidir se vale ligar ordens reais

Uso:
    python -m infra.backtester --csv finetune_csv/data/HL_BTC_1h.csv \\
                                --lookback 200 --pred_len 10 \\
                                --threshold 0.015
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import Kronos, KronosTokenizer, KronosPredictor
from trading.signal_generator import generate_signal, Direction


@dataclass
class BacktestResult:
    trades: List[dict] = field(default_factory=list)

    @property
    def n_trades(self): return len(self.trades)

    @property
    def win_rate(self):
        if not self.trades: return 0.0
        wins = sum(1 for t in self.trades if t["pnl_pct"] > 0)
        return wins / len(self.trades)

    @property
    def total_return(self):
        r = 1.0
        for t in self.trades:
            r *= (1 + t["pnl_pct"] / 100)
        return (r - 1) * 100

    @property
    def sharpe(self):
        if len(self.trades) < 2: return 0.0
        rets = [t["pnl_pct"] for t in self.trades]
        return np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)

    @property
    def max_drawdown(self):
        equity = [1.0]
        for t in self.trades:
            equity.append(equity[-1] * (1 + t["pnl_pct"] / 100))
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak: peak = e
            dd = (peak - e) / peak
            if dd > max_dd: max_dd = dd
        return max_dd * 100

    def summary(self) -> str:
        return (
            f"Trades: {self.n_trades} | "
            f"Win Rate: {self.win_rate:.1%} | "
            f"Total Return: {self.total_return:.2f}% | "
            f"Sharpe: {self.sharpe:.2f} | "
            f"Max DD: {self.max_drawdown:.2f}%"
        )


def run_backtest(
    csv_path: str,
    lookback: int = 200,
    pred_len: int = 10,
    sl_pct: float = 1.5,
    tp_pct: float = 3.0,
    model_name: str = "NeoQuasar/Kronos-small",
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
    device: str = "cpu",
    step: int = 1,              # avança 1 candle por vez (walk-forward)
    max_steps: int = 500,       # limita duração do backtest
) -> BacktestResult:

    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    print(f"Dataset: {len(df)} candles | Walk-forward de {lookback} a {len(df)-pred_len}")

    print("Carregando modelo…")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    result = BacktestResult()
    steps_done = 0

    for i in range(lookback, len(df) - pred_len, step):
        if steps_done >= max_steps:
            break

        hist = df.iloc[i - lookback: i].reset_index(drop=True)
        future = df.iloc[i: i + pred_len].reset_index(drop=True)

        x_df = hist[["open", "high", "low", "close", "volume"]]
        x_ts  = hist["timestamps"]
        y_ts  = future["timestamps"]

        forecast_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=1.0, top_p=0.9, sample_count=3,
        )

        signal = generate_signal(hist, forecast_df)

        if signal.direction == Direction.FLAT:
            steps_done += 1
            continue

        entry = hist["close"].iloc[-1]
        is_long = signal.direction == Direction.LONG

        # Simula candle a candle até SL, TP ou fim do horizonte
        pnl_pct = 0.0
        exit_reason = "timeout"
        for _, row in future.iterrows():
            high, low = row["high"], row["low"]
            if is_long:
                if low  <= entry * (1 - sl_pct / 100):
                    pnl_pct = -sl_pct; exit_reason = "SL"; break
                if high >= entry * (1 + tp_pct / 100):
                    pnl_pct = tp_pct;  exit_reason = "TP"; break
            else:
                if high >= entry * (1 + sl_pct / 100):
                    pnl_pct = -sl_pct; exit_reason = "SL"; break
                if low  <= entry * (1 - tp_pct / 100):
                    pnl_pct = tp_pct;  exit_reason = "TP"; break

        if exit_reason == "timeout":
            exit_price = future["close"].iloc[-1]
            pnl_pct = ((exit_price / entry) - 1) * 100 * (1 if is_long else -1)

        result.trades.append({
            "timestamp":  str(hist["timestamps"].iloc[-1]),
            "direction":  signal.direction.value,
            "entry":      entry,
            "confidence": signal.confidence,
            "pnl_pct":    round(pnl_pct, 4),
            "exit_reason": exit_reason,
        })

        steps_done += 1
        if steps_done % 20 == 0:
            print(f"  [{steps_done}/{max_steps}] {result.summary()}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       required=True)
    parser.add_argument("--lookback",  type=int,   default=200)
    parser.add_argument("--pred_len",  type=int,   default=10)
    parser.add_argument("--sl",        type=float, default=1.5)
    parser.add_argument("--tp",        type=float, default=3.0)
    parser.add_argument("--max_steps", type=int,   default=300)
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    result = run_backtest(
        csv_path=args.csv,
        lookback=args.lookback,
        pred_len=args.pred_len,
        sl_pct=args.sl,
        tp_pct=args.tp,
        max_steps=args.max_steps,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(result.summary())

    out = Path("infra/backtest_results.csv")
    pd.DataFrame(result.trades).to_csv(out, index=False)
    print(f"Trades salvos em: {out}")


if __name__ == "__main__":
    main()
