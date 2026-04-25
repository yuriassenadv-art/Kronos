"""
Camada 4 — Execução (alertas)

Envia notificações Telegram a cada evento relevante do workflow.

Setup:
  1. Crie um bot: https://t.me/BotFather → /newbot → copie o token
  2. Abra conversa com o bot e mande /start
  3. Pegue seu chat_id: https://api.telegram.org/bot<TOKEN>/getUpdates
  4. Defina as variáveis de ambiente:
       TELEGRAM_BOT_TOKEN=...
       TELEGRAM_CHAT_ID=...
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")


def _send(text: str):
    if not _BOT_TOKEN or not _CHAT_ID:
        logger.debug("Telegram não configurado — mensagem suprimida: %s", text)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception as exc:
        logger.warning("Falha ao enviar Telegram: %s", exc)


def alert_signal(coin: str, direction: str, confidence: float, reason: str):
    _send(
        f"📡 *Sinal Kronos*\n"
        f"Par: `{coin}`\n"
        f"Direção: `{direction}`\n"
        f"Confiança: `{confidence:.1%}`\n"
        f"Motivo: `{reason}`"
    )


def alert_order_placed(coin: str, direction: str, size_usd: float,
                        sl: float, tp: float, dry_run: bool):
    tag = "🔵 DRY RUN" if dry_run else "✅ ORDEM ENVIADA"
    _send(
        f"{tag}\n"
        f"Par: `{coin}`  |  `{direction}`\n"
        f"Tamanho: `${size_usd:.2f}`\n"
        f"SL: `{sl:.2f}`  |  TP: `{tp:.2f}`"
    )


def alert_position_closed(coin: str, direction: str, reason: str):
    _send(
        f"🟢 *Posição fechada*\n"
        f"Par: `{coin}`\n"
        f"Direção: `{direction}`\n"
        f"Motivo: `{reason}`"
    )


def alert_error(context: str, error: str):
    _send(f"❌ *Erro — {context}*\n```\n{error[:400]}\n```")


def alert_cycle_start(coin: str, interval: str, cycle: int):
    _send(f"🔄 Ciclo #{cycle} | `{coin}/{interval}`")
