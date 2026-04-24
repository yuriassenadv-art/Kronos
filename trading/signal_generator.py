"""
Signal Generator — converte o forecast do Kronos em direção de trade.

Este é o núcleo estratégico do workflow. A implementação de `generate_signal`
é a decisão mais importante do sistema — veja o TODO abaixo.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"   # sem posição / fechar posição existente


@dataclass
class Signal:
    direction: Direction
    confidence: float          # 0.0 – 1.0
    entry_price: float         # preço atual (último close histórico)
    forecast_horizon: int      # em candles
    reason: str = ""


def generate_signal(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> Signal:
    """
    Recebe:
      historical_df — DataFrame histórico com [open, high, low, close, volume]
      forecast_df   — DataFrame de previsão do Kronos com as mesmas colunas,
                      indexado pelos timestamps futuros

    Deve retornar um Signal com direction, confidence e entry_price.

    TODO: Implemente sua lógica aqui (5-10 linhas).

    Abordagens possíveis a considerar:

    A) Retorno esperado simples
       - Compara close previsto no horizonte com close atual
       - LONG se (forecast[-1].close / current_close - 1) > threshold
       - Simples, mas sensível a ruído do modelo

    B) Mediana de trajetória
       - Usa vários sample_count e calcula a mediana de todos os forecasts
       - Mais robusto, requer sample_count > 1 no KronosPredictor

    C) Inclinação da tendência (regressão linear)
       - Ajusta uma linha nos closes previstos
       - LONG se inclinação > X%, SHORT se < -X%
       - Filtra melhor ruído de curto prazo

    D) Quebra de nível (high/low previsto vs. atual)
       - LONG se forecast.high.max() > current_close * (1 + threshold)
       - Usa informação dos campos H/L que o Kronos também prevê

    Constraints importantes:
    - confidence muito baixa (<0.4) → retorne FLAT para evitar ruído
    - O Kronos não é calibrado para cripto 24/7 sem fine-tuning — considere
      um threshold conservador inicialmente (ex: >1.5% para agir)
    """

    current_close = historical_df["close"].iloc[-1]
    horizon = len(forecast_df)

    # ─── Implemente sua lógica aqui ───────────────────────────────────────────

    predicted_close = forecast_df["close"].iloc[-1]
    expected_return = (predicted_close / current_close) - 1.0
    confidence = min(abs(expected_return) / 0.03, 1.0)  # normaliza em 3%

    if expected_return > 0.015 and confidence > 0.4:
        direction = Direction.LONG
    elif expected_return < -0.015 and confidence > 0.4:
        direction = Direction.SHORT
    else:
        direction = Direction.FLAT

    # ──────────────────────────────────────────────────────────────────────────

    return Signal(
        direction=direction,
        confidence=confidence,
        entry_price=current_close,
        forecast_horizon=horizon,
        reason=f"expected_return={expected_return:.3%}",
    )
