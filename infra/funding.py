"""
Consulta de funding rate atual na Hyperliquid e ajuste de pesos por sinal.

A Hyperliquid expõe funding via Info.meta_and_asset_ctxs():
    [
        {"universe": [...]},   # mesma estrutura de meta()
        [
            {"funding": "0.0000125", "openInterest": "...", ...},
            ...
        ]
    ]

A ordem do segundo array corresponde à ordem de `universe` no primeiro.
"funding" é a taxa horária em formato decimal (0.0000125 = 0.00125% / hora).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cache module-level (atualiza a cada ciclo via refresh_funding_cache).
_FUNDING_CACHE: dict[str, float] = {}

# Threshold default: funding diário absoluto acima do qual reduzimos peso
# 0.0001/h × 24h = 0.24%/dia — começa a virar tax relevante
FUNDING_THRESHOLD_HOURLY = 0.0001  # 0.01% / hora


def refresh_funding_cache(base_url: str) -> dict[str, float]:
    """
    Atualiza cache de funding por coin via Info.meta_and_asset_ctxs().
    Chamado uma vez por ciclo (em workflow.py:run_cycle, antes do
    allocate_capital).

    Returns:
        dict[coin, funding_hourly] — funding em decimal (0.0001 = 0.01%/h).
        Em caso de erro, retorna o cache anterior (ou {} na primeira falha).
    """
    global _FUNDING_CACHE

    try:
        from hyperliquid.info import Info
        info = Info(base_url, skip_ws=True)
        meta_and_ctx = info.meta_and_asset_ctxs()
        # Estrutura: [{"universe": [...]}, [ctx0, ctx1, ...]]
        universe = meta_and_ctx[0]["universe"]
        ctxs = meta_and_ctx[1]

        new_cache: dict[str, float] = {}
        for asset, ctx in zip(universe, ctxs):
            try:
                new_cache[asset["name"]] = float(ctx.get("funding", 0))
            except (TypeError, ValueError):
                continue

        _FUNDING_CACHE = new_cache
        logger.debug(
            "Funding cache atualizado: %d coins | exemplo BTC=%.5f%%/h",
            len(new_cache),
            (new_cache.get("BTC", 0) * 100),
        )
    except Exception as exc:
        logger.warning(
            "Falha ao atualizar funding cache: %s (mantendo cache anterior)",
            exc,
        )

    return _FUNDING_CACHE


def get_funding(coin: str) -> float:
    """Retorna o funding hourly cacheado, 0.0 se não houver dado."""
    return _FUNDING_CACHE.get(coin, 0.0)


def funding_penalty(coin: str, is_long: bool, threshold: float = FUNDING_THRESHOLD_HOURLY) -> float:
    """
    Retorna um multiplicador 0..1 para reduzir peso da alocação quando o
    funding é desfavorável à direção do trade:
      - LONG paga funding positivo → penaliza se funding > +threshold
      - SHORT paga funding negativo → penaliza se funding < -threshold

    Modelo simples: penalidade linear até 50% de redução em funding 5x o
    threshold. Floor em 0.5 — nunca zera o sinal só por funding.

    Returns:
        float entre 0.5 e 1.0 (multiplicador a aplicar no peso/tamanho).
    """
    f = get_funding(coin)

    # Funding desfavorável?
    if is_long and f > threshold:
        excess_ratio = (f - threshold) / (4 * threshold)  # 0 em threshold, 1 em 5x
    elif (not is_long) and f < -threshold:
        excess_ratio = (-f - threshold) / (4 * threshold)
    else:
        return 1.0  # sem penalidade

    # Penalidade linear, clamped em [0.5, 1.0]
    excess_ratio = max(0.0, min(1.0, excess_ratio))
    return 1.0 - 0.5 * excess_ratio


def clear_cache():
    """Reset do cache (útil em testes)."""
    global _FUNDING_CACHE
    _FUNDING_CACHE = {}
