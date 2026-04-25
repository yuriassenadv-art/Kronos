"""
Sistema de autoverificação do bot multi-asset Kronos.

Estratégia:
1. Startup: build_ensembles_with_retry() tenta carregar modelos N vezes.
   Se falhar, sai com sys.exit(1) — systemd / supervisord reinicia.
2. Runtime: ensemble_health_check() roda smoke inference periódica.
   Cada chamada incrementa um contador de falhas consecutivas; se exceder
   o threshold, sai com sys.exit(1).

Nada disso depende de Telegram — só logs estruturados e exit codes.
"""
import logging
import sys
import time
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Estado do health check (module-level, simples como o resto do bot) ────────
_consecutive_failures: int = 0
MAX_CONSECUTIVE_HEALTH_FAILURES = 2


def build_ensembles_with_retry(
    builder: Callable[[], dict],
    max_attempts: int = 3,
    base_delay: float = 5.0,
) -> dict:
    """
    Wrapper sobre `build_ensembles(cfg)` (passado como `builder`) com retry.
    Após `max_attempts`, chama sys.exit(1) — supervisor decide se reinicia.

    Args:
        builder: callable que retorna dict[coin, EnsemblePredictor]
        max_attempts: tentativas antes de desistir
        base_delay: delay base entre tentativas (exponencial: 5s, 10s, 20s…)

    Returns:
        dict[coin, EnsemblePredictor]
    """
    for attempt in range(max_attempts):
        try:
            ensembles = builder()
            logger.info(
                "Ensembles carregados com sucesso na tentativa %d/%d",
                attempt + 1, max_attempts,
            )
            return ensembles
        except Exception as exc:
            delay = base_delay * (2 ** attempt)
            if attempt + 1 == max_attempts:
                logger.error(
                    "FATAL: build_ensembles falhou após %d tentativas (%s) — "
                    "saindo com exit 1 para supervisor reiniciar",
                    max_attempts, exc,
                )
                sys.exit(1)

            logger.warning(
                "build_ensembles tentativa %d/%d falhou: %s — retry em %.0fs",
                attempt + 1, max_attempts, exc, delay,
            )
            time.sleep(delay)

    # Unreachable, mas explicit para mypy/linters
    sys.exit(1)


def ensemble_health_check(
    ensembles: dict,
    smoke_df: Optional[pd.DataFrame] = None,
) -> bool:
    """
    Roda uma smoke inference em cada ensemble para garantir que estão
    funcionais. Não custa muito — sample_count=1, pred_len pequeno.

    Estratégia de exit:
    - Cada chamada que detecta TODOS os ensembles falhando incrementa um
      contador module-level.
    - Se contador >= MAX_CONSECUTIVE_HEALTH_FAILURES, chama sys.exit(1).
    - Se ao menos 1 ensemble responde, contador é resetado.

    Returns:
        True se ao menos 1 ensemble está saudável; False caso contrário
        (mas pode ter saído antes).
    """
    global _consecutive_failures

    # Smoke df default: candles falsos suficientes para ensemble não crashar
    if smoke_df is None:
        smoke_df = _make_smoke_df()

    healthy = []
    failed = []
    for coin, ensemble in ensembles.items():
        try:
            # Smoke inference: 1 sample, pred_len pequeno
            ensemble.get_signal(smoke_df, pred_len=2, interval="15m")
            healthy.append(coin)
        except Exception as exc:
            logger.warning("[%s] health check falhou: %s", coin, exc)
            failed.append(coin)

    logger.info(
        "Health check: saudáveis=%s | falhos=%s",
        healthy or "(nenhum!)", failed,
    )

    if healthy:
        # Pelo menos 1 ensemble responde → reset contador
        _consecutive_failures = 0
        return True

    # NENHUM ensemble respondeu — incrementa
    _consecutive_failures += 1
    logger.error(
        "Health check: TODOS os %d ensembles falharam "
        "(consecutivo: %d/%d)",
        len(ensembles), _consecutive_failures,
        MAX_CONSECUTIVE_HEALTH_FAILURES,
    )

    if _consecutive_failures >= MAX_CONSECUTIVE_HEALTH_FAILURES:
        logger.error(
            "FATAL: %d health checks consecutivos falharam — "
            "saindo com exit 1 para supervisor reiniciar",
            _consecutive_failures,
        )
        sys.exit(1)

    return False


def reset_health_state():
    """Reseta o contador (útil em testes)."""
    global _consecutive_failures
    _consecutive_failures = 0


def _make_smoke_df() -> pd.DataFrame:
    """DataFrame OHLCV mínimo para smoke inference. 200 candles mockados."""
    import numpy as np
    n = 200
    base = 50000.0
    closes = base + np.random.randn(n).cumsum() * 100
    return pd.DataFrame({
        "timestamps": pd.date_range("2026-01-01", periods=n, freq="15min"),
        "open":   closes + np.random.randn(n) * 50,
        "high":   closes + np.abs(np.random.randn(n)) * 100,
        "low":    closes - np.abs(np.random.randn(n)) * 100,
        "close":  closes,
        "volume": np.abs(np.random.randn(n)) * 1000,
    })
