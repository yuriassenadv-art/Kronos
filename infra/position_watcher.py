"""
Camada 4 — Execução (sincronização de posições)

Reconciliação entre `state.json` (visão local do bot) e a Hyperliquid
(verdade de mercado). Detecta quando uma posição foi fechada na exchange
fora do controle do bot — tipicamente:

  * Stop Loss disparado (trigger order executada)
  * Take Profit disparado
  * Liquidação por margem
  * Fechamento manual via UI da Hyperliquid

Esses eventos NÃO geram callbacks no bot; sem este watcher, o state.json
fica "fantasma" e o `filter_already_open` impede a abertura de novas
posições no mesmo coin para sempre.

Uso:
  * Chamado no início de cada `run_cycle` para detectar SL/TP entre ciclos
  * Chamado no startup do bot (`main`) para reconciliar após crash/restart

Em `dry_run=True` é no-op (não há ordens reais para sincronizar).
"""

import logging
from typing import Any

from infra.state_manager import StateManager
from infra import monitor
from trading.config import HyperliquidConfig

logger = logging.getLogger(__name__)

# Importa Info preguiçosamente: o pacote hyperliquid pode não estar
# instalado em ambientes de dev/test. Em runtime (VPS) está sempre presente.
# Em testes, o stub de `sys.modules["hyperliquid.info"].Info` é suficiente
# para que `patch("infra.position_watcher.Info")` funcione.
try:
    from hyperliquid.info import Info  # type: ignore
except Exception:  # pragma: no cover — só dispara em ambientes sem o SDK
    Info = None  # type: ignore


def _exchange_open_coins(user_state: dict[str, Any]) -> set[str]:
    """
    Extrai o conjunto de coins com posição aberta (size != 0) do payload
    de `Info.user_state`.

    Layout esperado (Hyperliquid SDK):
        {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "0.001", ...}},
                {"position": {"coin": "ETH", "szi": "0",     ...}},  # fechada
            ],
            "marginSummary": {...},
        }

    Considera fechada qualquer posição com `szi == 0` ou ausente da lista.
    """
    open_coins: set[str] = set()
    for entry in user_state.get("assetPositions", []) or []:
        pos = (entry or {}).get("position") or {}
        coin = pos.get("coin")
        if not coin:
            continue
        try:
            szi = float(pos.get("szi", "0"))
        except (TypeError, ValueError):
            # Campo malformado — trata como fechada por segurança
            szi = 0.0
        if szi != 0.0:
            open_coins.add(coin)
    return open_coins


def sync_positions(
    state: StateManager,
    hl_cfg: HyperliquidConfig,
    dry_run: bool,
) -> list[str]:
    """
    Reconcilia state.json com a verdade da Hyperliquid.

    Para cada posição em `state.all_positions()`, verifica se o ativo ainda
    está aberto na exchange. Se NÃO estiver (size == 0 ou ausente), assume
    que SL/TP/liquidação disparou e:
      1. Limpa a posição do state via `state.clear_position(coin)`
      2. Notifica via `monitor.alert_position_closed(...)`
      3. Loga warning

    Args:
        state:    StateManager com as posições locais
        hl_cfg:   Credenciais/endpoint da Hyperliquid
        dry_run:  Se True, não consulta a exchange (não há ordens reais)

    Returns:
        Lista de coins que foram fechados neste sync (vazia se tudo em sync).
        Em `dry_run=True` sempre retorna [].

    Resiliência:
        Se a chamada à Hyperliquid falhar (rede, rate-limit, etc.), captura
        a exceção, loga warning e retorna []. NUNCA derruba o bot.
    """
    if dry_run:
        return []

    local_positions = state.all_positions()
    if not local_positions:
        return []

    if Info is None:
        logger.warning(
            "sync_positions: pacote hyperliquid não disponível — "
            "reconciliação saltada."
        )
        return []

    try:
        info = Info(hl_cfg.base_url, skip_ws=True)
        user_state = info.user_state(hl_cfg.account_address)
    except Exception as exc:
        logger.warning(
            "sync_positions: falha ao consultar Hyperliquid user_state: %s. "
            "Mantendo state local intocado neste ciclo.",
            exc,
        )
        return []

    exchange_open = _exchange_open_coins(user_state)
    closed: list[str] = []

    for coin, pos in local_positions.items():
        if coin in exchange_open:
            continue  # posição segue aberta na exchange — nada a fazer

        # Posição local sem contraparte na exchange → fechada externamente
        logger.warning(
            "sync_positions: posição %s ausente na exchange — "
            "assumindo SL/TP/liquidação disparou. Limpando state.",
            coin,
        )
        state.clear_position(coin)
        try:
            monitor.alert_position_closed(
                coin,
                pos.direction,
                "SL/TP ou liquidação detectado via sync",
            )
        except Exception as exc:
            # Falha no Telegram NUNCA pode derrubar a reconciliação
            logger.warning(
                "sync_positions: falha ao notificar fechamento de %s: %s",
                coin, exc,
            )
        closed.append(coin)

    return closed
