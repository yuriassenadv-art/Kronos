"""
Portfolio Manager — Camada de alocação multi-asset.

Decide como dividir o capital disponível entre múltiplos sinais
direcionais simultâneos (BTC, ETH, SOL).

Implementação ativa: Filosofia B — Confidence-weighted.
Usa o `signal.confidence` (já calibrado pelo agreement_score do ensemble)
como peso de alocação proporcional. O coin com sinal mais forte recebe
fatia maior do risco; sinais fracos recebem fatia menor.
"""
from dataclasses import replace
from typing import Optional

from .config import TradingConfig
from .signal_generator import Signal, Direction
from .risk_manager import OrderParams, calculate_order


def allocate_capital(
    signals: dict[str, Signal],
    balance: float,
    cfg: TradingConfig,
) -> dict[str, OrderParams]:
    """
    Recebe um dict de sinais por coin e retorna um dict de OrderParams
    indicando quanto alocar em cada um.

    Args:
        signals: dict mapeando coin -> Signal. Pode conter Direction.FLAT
                 (sinais que não devem virar ordem).
        balance: saldo disponível em USDC na conta.
        cfg: TradingConfig com risk_per_trade_pct, leverage,
             max_concurrent_positions, etc.

    Returns:
        dict[coin, OrderParams] — apenas coins que receberam alocação.
        Coins com Direction.FLAT ou que perderam na competição NÃO devem
        aparecer no dict de retorno.

    ════════════════════════════════════════════════════════════════════
    TODO (USUÁRIO): Implementar a lógica de alocação.
    Existem 4 abordagens canônicas — escolha uma (ou combine):
    ════════════════════════════════════════════════════════════════════

    A) **Equal-weight (mais simples)**
       - Ignora confidence: cada sinal não-FLAT recebe 1/N do risco total.
       - Pro: previsível, baixa concentração, fácil debugar.
       - Con: desperdiça informação de confidence; aloca igual a sinais
              fracos e fortes.
       - Implementação: filtra FLATs, divide cfg.risk_per_trade_pct por
         len(signals_active), chama calculate_order para cada com cfg
         modificado.

    B) **Confidence-weighted (informação-aware)**
       - Aloca proporcional a signal.confidence.
       - Pro: usa o sinal mais informativo do ensemble (agreement score).
       - Con: pode concentrar muito em 1 ativo se confidence dispara.
       - Implementação: weights = [s.confidence for s in active]; normaliza;
         chama calculate_order com risk_pct ajustado por peso.

    C) **Sharpe-weighted (histórico-aware)**
       - Aloca proporcional ao Sharpe ratio histórico de cada coin
         (medido em backtests do infra/backtester.py).
       - Pro: prioriza ativos onde o modelo demonstrou edge real.
       - Con: requer backtests recentes; passado não garante futuro.
       - Implementação: precisa de cfg.sharpe_per_coin: dict[str, float]
         (ou hardcoded como ponto de partida). Sugere-se fallback para
         equal-weight se faltar histórico.

    D) **Best-only (winner-takes-all)**
       - Pega só o sinal com maior confidence em cada ciclo. Os outros
         viram FLAT.
       - Pro: máxima simplicidade, aloca capital cheio ao sinal mais forte.
       - Con: descarta diversificação; muito sensível a outliers de
              confidence.
       - Implementação: max(active, key=lambda s: s.confidence);
         calculate_order só para esse coin com cfg.risk_per_trade_pct cheio.

    Constraints obrigatórios em qualquer implementação:
    - Filtrar Direction.FLAT antes de alocar (sinais FLAT viram skip).
    - Respeitar cfg.max_concurrent_positions: se > N coins ativos,
      pegar top-N por confidence ou descartar excedente.
    - Cada chamada interna deve usar trading.risk_manager.calculate_order
      passando uma cópia ajustada de cfg (NÃO mutar cfg original).
    - Return type sempre dict[str, OrderParams] — coin nunca aparece
      com None.

    ────────────────────────────────────────────────────────────────────
    IMPLEMENTAÇÃO ATIVA: Filosofia B (Confidence-weighted)
    ────────────────────────────────────────────────────────────────────
    O risco total `cfg.risk_per_trade_pct` é distribuído entre os coins
    ativos proporcionalmente ao `signal.confidence` (que já incorpora
    o agreement_score do ensemble). Sinais mais convictos recebem
    fatia maior; sinais fracos, fatia menor.
    """
    # 1. Filtra FLATs e sinais sem confidence positiva (evita div/0).
    active = {
        coin: sig
        for coin, sig in signals.items()
        if sig.direction != Direction.FLAT and sig.confidence > 0
    }
    if not active:
        return {}

    # 2. Respeita max_concurrent_positions: top-N por confidence.
    if len(active) > cfg.max_concurrent_positions:
        active = dict(
            sorted(
                active.items(),
                key=lambda kv: kv[1].confidence,
                reverse=True,
            )[: cfg.max_concurrent_positions]
        )

    # 3. Pesos normalizados pela soma das confidences.
    total_conf = sum(s.confidence for s in active.values())
    weights = {coin: sig.confidence / total_conf for coin, sig in active.items()}

    # 4. Para cada coin: cfg.copy() ajustado com risco fatiado pelo peso.
    allocations: dict[str, OrderParams] = {}
    for coin, sig in active.items():
        coin_cfg = replace(
            cfg,
            coin=coin,
            risk_per_trade_pct=cfg.risk_per_trade_pct * weights[coin],
        )
        params = calculate_order(sig, balance, coin_cfg)
        if params is not None:
            allocations[coin] = params

    return allocations
