"""
Camada 3 — Validação (parte 2)

EnsemblePredictor: encapsula o KronosPredictor com:
  - Multi-sample: N paths estocásticos → mediana → sinal mais estável
  - Multi-timeframe: confirma sinal em TF maior antes de agir
  - Confidence score calibrado pela concordância entre samples
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import Kronos, KronosTokenizer, KronosPredictor
from trading.data_fetcher import fetch_candles, build_kronos_inputs
from trading.signal_generator import generate_signal, Signal, Direction


# Mapeamento de TF menor → TF confirmatório maior
CONFIRMATION_TF = {
    "1m":  "15m",
    "5m":  "1h",
    "15m": "1h",
    "1h":  "4h",
    "4h":  "1d",
    "1d":  "1d",
}


class EnsemblePredictor:
    """
    Wrapper sobre KronosPredictor que produz sinais mais confiáveis
    usando múltiplas amostras e confirmação multi-timeframe.
    """

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-small",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cpu",
        max_context: int = 512,
        sample_count: int = 10,
        temperature: float = 1.0,
        top_p: float = 0.9,
        use_confirmation_tf: bool = True,
        coin: str = "BTC",
        base_url: str = "https://api.hyperliquid.xyz",
        model_subfolder: Optional[str] = None,
        tokenizer_subfolder: Optional[str] = None,
    ):
        """
        Args adicionais (multi-asset / HF Hub layout):
            model_subfolder: subfolder dentro do repo HF para o basemodel
                             (ex: "basemodel/best_model" para repos savycorp/kronos-bn-*).
            tokenizer_subfolder: subfolder dentro do repo HF para o tokenizer
                                 (ex: "tokenizer/best_model" para repos savycorp/kronos-bn-*).
            Ambos são opcionais — quando None, carrega do root do repo (compat com NeoQuasar).
        """
        self.sample_count = sample_count
        self.temperature = temperature
        self.top_p = top_p
        self.use_confirmation_tf = use_confirmation_tf
        self.coin = coin
        self.base_url = base_url

        tok_kwargs = {"subfolder": tokenizer_subfolder} if tokenizer_subfolder else {}
        mdl_kwargs = {"subfolder": model_subfolder} if model_subfolder else {}

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name, **tok_kwargs)
        model = Kronos.from_pretrained(model_name, **mdl_kwargs)
        self.predictor = KronosPredictor(
            model, tokenizer, device=device, max_context=max_context
        )

    def predict_ensemble(
        self,
        df: pd.DataFrame,
        pred_len: int,
        interval: str,
    ) -> tuple[pd.DataFrame, float]:
        """
        Roda `sample_count` forecasts independentes e retorna:
          - forecast_median: mediana dos closes previstos (mais robusto)
          - agreement: fração de samples que concordam na direção (0-1)
        """
        x_df, x_ts, y_ts = build_kronos_inputs(df, pred_len, interval)
        current_close = df["close"].iloc[-1]

        all_closes = []
        for _ in range(self.sample_count):
            fc = self.predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=pred_len, T=self.temperature,
                top_p=self.top_p, sample_count=1,
            )
            all_closes.append(fc["close"].values)

        closes_matrix = np.array(all_closes)          # shape: (sample_count, pred_len)
        median_closes = np.median(closes_matrix, axis=0)

        # Concordância: quantos samples preveem a mesma direção que a mediana?
        final_median = median_closes[-1]
        directions = (closes_matrix[:, -1] > current_close).astype(int)
        majority = 1 if final_median > current_close else 0
        agreement = np.mean(directions == majority)

        # Reconstrói DataFrame com closes medianos
        forecast_median = self.predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=self.temperature,
            top_p=self.top_p, sample_count=1,
        ).copy()
        forecast_median["close"] = median_closes

        return forecast_median, float(agreement)

    def get_signal(
        self,
        df: pd.DataFrame,
        pred_len: int,
        interval: str,
        lookback_confirm: int = 100,
    ) -> Signal:
        """
        Gera sinal com filtro de concordância e confirmação multi-TF.
        Retorna FLAT se:
          - agreement < 0.6  (samples divergem demais)
          - TF confirmatório aponta direção oposta
        """
        forecast_df, agreement = self.predict_ensemble(df, pred_len, interval)

        # Injeta agreement como fator na confidence
        base_signal = generate_signal(df, forecast_df)
        calibrated_confidence = base_signal.confidence * agreement

        # Bloqueia se amostras discordam muito
        if agreement < 0.6:
            return Signal(
                direction=Direction.FLAT,
                confidence=calibrated_confidence,
                entry_price=base_signal.entry_price,
                forecast_horizon=pred_len,
                reason=f"low_agreement={agreement:.2f}",
            )

        if base_signal.direction == Direction.FLAT:
            return base_signal

        # Confirmação multi-timeframe
        if self.use_confirmation_tf:
            confirm_interval = CONFIRMATION_TF.get(interval, interval)
            if confirm_interval != interval:
                confirmed = self._confirm_with_higher_tf(
                    base_signal.direction, confirm_interval, lookback_confirm, pred_len
                )
                if not confirmed:
                    return Signal(
                        direction=Direction.FLAT,
                        confidence=calibrated_confidence,
                        entry_price=base_signal.entry_price,
                        forecast_horizon=pred_len,
                        reason=f"rejected_by_{confirm_interval}_confirmation",
                    )

        return Signal(
            direction=base_signal.direction,
            confidence=calibrated_confidence,
            entry_price=base_signal.entry_price,
            forecast_horizon=pred_len,
            reason=f"{base_signal.reason} | agreement={agreement:.2f}",
        )

    def _confirm_with_higher_tf(
        self,
        direction: Direction,
        confirm_interval: str,
        lookback: int,
        pred_len: int,
    ) -> bool:
        """Busca TF maior e verifica se a tendência concorda."""
        try:
            df_high = fetch_candles(self.coin, confirm_interval, lookback, self.base_url)
            forecast_high, _ = self.predict_ensemble(df_high, pred_len, confirm_interval)
            signal_high = generate_signal(df_high, forecast_high)

            if signal_high.direction == Direction.FLAT:
                return True  # neutro não bloqueia

            return signal_high.direction == direction
        except Exception:
            return True   # em caso de erro, não bloqueia
