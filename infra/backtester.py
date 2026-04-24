"""
Camada 3 — Validação

Walk-forward backtest com suporte a Long/Short e leverage configurável.
Simula o loop de trading no histórico, candle a candle, sem lookahead.

Uso:
    python -m infra.backtester --csv finetune_csv/data/BN_BTC_15m.csv \\
                                --lookback 200 --pred_len 8 \\
                                --sl 1.0 --tp 2.0 --leverage 5
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
    leverage: int = 1

    @property
    def n_trades(self): return len(self.trades)

    @property
    def n_long(self): return sum(1 for t in self.trades if t["direction"] == "long")

    @property
    def n_short(self): return sum(1 for t in self.trades if t["direction"] == "short")

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

    @property
    def liquidations(self):
        return sum(1 for t in self.trades if t.get("exit_reason") == "LIQ")

    def summary(self) -> str:
        liq_str = f" | Liquidações: {self.liquidations}" if self.liquidations else ""
        return (
            f"Trades: {self.n_trades} (L:{self.n_long}/S:{self.n_short}) | "
            f"Win Rate: {self.win_rate:.1%} | "
            f"Total Return: {self.total_return:.2f}% | "
            f"Sharpe: {self.sharpe:.2f} | "
            f"Max DD: {self.max_drawdown:.2f}%"
            f"{liq_str} | "
            f"Leverage: {self.leverage}x"
        )


def run_backtest(
    csv_path: str,
    lookback: int = 200,
    pred_len: int = 8,
    sl_pct: float = 1.0,
    tp_pct: float = 2.0,
    leverage: int = 3,
    model_name: str = "NeoQuasar/Kronos-small",
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
    device: str = "cpu",
    step: int = 1,
    max_steps: int = 500,
) -> BacktestResult:

    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    print(f"Dataset: {len(df)} candles | Walk-forward de {lookback} a {len(df)-pred_len}")
    print(f"Leverage: {leverage}x | SL: {sl_pct}% preço ({sl_pct*leverage:.1f}% margin) | "
          f"TP: {tp_pct}% preço ({tp_pct*leverage:.1f}% margin)")

    print("Carregando modelo…")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    result = BacktestResult(leverage=leverage)
    steps_done = 0

    # Fee como % do margin = fee_nocional × leverage
    # Hyperliquid: 0.05% taker × 2 lados = 0.10% nocional = 0.10% × leverage (margin)
    FEE_MARGIN = 0.10 * leverage

    # Liquidação: preço adverso de (100/leverage)% zera a margin (com buffer de 5%)
    LIQ_PCT = (100.0 / leverage) * 0.95

    for i in range(lookback, len(df) - pred_len, step):
        if steps_done >= max_steps:
            break

        hist   = df.iloc[i - lookback: i].reset_index(drop=True)
        future = df.iloc[i: i + pred_len].reset_index(drop=True)

        x_df = hist[["open", "high", "low", "close", "volume"]]
        x_ts = hist["timestamps"]
        y_ts = future["timestamps"]

        forecast_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=1.0, top_p=0.9, sample_count=3,
        )

        signal = generate_signal(hist, forecast_df)

        if signal.direction == Direction.FLAT:
            steps_done += 1
            continue

        entry   = hist["close"].iloc[-1]
        is_long = signal.direction == Direction.LONG

        pnl_pct    = 0.0
        exit_reason = "timeout"

        for _, row in future.iterrows():
            high, low = row["high"], row["low"]

            if is_long:
                # Liquidação (price drop > LIQ_PCT%)
                if low <= entry * (1 - LIQ_PCT / 100):
                    pnl_pct = -100.0 - FEE_MARGIN
                    exit_reason = "LIQ"
                    break
                # Stop Loss
                if low <= entry * (1 - sl_pct / 100):
                    pnl_pct = -(sl_pct * leverage) - FEE_MARGIN
                    exit_reason = "SL"
                    break
                # Take Profit
                if high >= entry * (1 + tp_pct / 100):
                    pnl_pct = (tp_pct * leverage) - FEE_MARGIN
                    exit_reason = "TP"
                    break
            else:  # SHORT
                # Liquidação (price rise > LIQ_PCT%)
                if high >= entry * (1 + LIQ_PCT / 100):
                    pnl_pct = -100.0 - FEE_MARGIN
                    exit_reason = "LIQ"
                    break
                # Stop Loss
                if high >= entry * (1 + sl_pct / 100):
                    pnl_pct = -(sl_pct * leverage) - FEE_MARGIN
                    exit_reason = "SL"
                    break
                # Take Profit
                if low <= entry * (1 - tp_pct / 100):
                    pnl_pct = (tp_pct * leverage) - FEE_MARGIN
                    exit_reason = "TP"
                    break

        if exit_reason == "timeout":
            price_change_pct = ((future["close"].iloc[-1] / entry) - 1) * 100
            if not is_long:
                price_change_pct = -price_change_pct
            pnl_pct = (price_change_pct * leverage) - FEE_MARGIN

        result.trades.append({
            "timestamp":   str(hist["timestamps"].iloc[-1]),
            "direction":   signal.direction.value,
            "entry":       entry,
            "confidence":  signal.confidence,
            "pnl_pct":     round(pnl_pct, 4),
            "exit_reason": exit_reason,
            "leverage":    leverage,
        })

        steps_done += 1
        if steps_done % 20 == 0:
            print(f"  [{steps_done}/{max_steps}] {result.summary()}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       required=True)
    parser.add_argument("--lookback",  type=int,   default=200)
    parser.add_argument("--pred_len",  type=int,   default=8)
    parser.add_argument("--sl",        type=float, default=1.0)
    parser.add_argument("--tp",        type=float, default=2.0)
    parser.add_argument("--leverage",  type=int,   default=3)
    parser.add_argument("--max_steps", type=int,   default=300)
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    result = run_backtest(
        csv_path=args.csv,
        lookback=args.lookback,
        pred_len=args.pred_len,
        sl_pct=args.sl,
        tp_pct=args.tp,
        leverage=args.leverage,
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
