"""
First-contact smoke test: valida fim-a-fim o pipeline live na Hyperliquid
SEM iniciar o loop principal. Abre uma ordem MÍNIMA e fecha imediatamente.

Use UMA VEZ antes do primeiro deploy live para confirmar:
  - API wallet assina corretamente
  - Size precision funciona com size mínimo
  - market_open + close fluem
  - Balance reflete corretamente após o ciclo

Custo esperado: ~$0.10 em fees (taker round-trip 0.10%).

Uso:
    HYPERLIQUID_PRIVATE_KEY=0x... \\
    HYPERLIQUID_ACCOUNT_ADDRESS=0x... \\
    python3 scripts/first_contact_test.py [COIN]

COIN default = BTC. Use o coin com menor minimum order na sua conta.
"""
import logging
import sys
import time
from pathlib import Path

# Adiciona o root ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.config import Config
from trading.order_executor import (
    get_account_balance,
    place_order,
    _round_qty,
    _load_sz_decimals,
)
from trading.risk_manager import OrderParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Tamanho mínimo da ordem em USD (a Hyperliquid exige >= $10 em geral).
# Ajustado para $10.5 — cabe em conta com $19 USDC e mantém buffer para fees.
TEST_SIZE_USD = 10.5
TEST_SL_PCT = 5.0   # SL/TP largos pra não disparar durante o teste
TEST_TP_PCT = 5.0
LEVERAGE = 1


def main():
    coin = sys.argv[1] if len(sys.argv) > 1 else "BTC"

    cfg = Config()
    cfg.trading.dry_run = False    # FORÇA modo live para o teste
    cfg.trading.coin = coin
    cfg.trading.leverage = LEVERAGE
    cfg.trading.sl_pct = TEST_SL_PCT
    cfg.trading.tp_pct = TEST_TP_PCT

    if not cfg.hyperliquid.private_key or not cfg.hyperliquid.account_address:
        logger.error(
            "Defina HYPERLIQUID_PRIVATE_KEY e HYPERLIQUID_ACCOUNT_ADDRESS "
            "antes de rodar o smoke test."
        )
        sys.exit(1)

    # Banner visual
    logger.warning("=" * 70)
    logger.warning("FIRST-CONTACT SMOKE TEST")
    logger.warning("Coin: %s | Size: $%.2f | Lev: %dx", coin, TEST_SIZE_USD, LEVERAGE)
    logger.warning("ESTE TESTE ENVIA ORDEM REAL — custo esperado ~$0.10 em fees")
    logger.warning("=" * 70)

    # Etapa 1: balance
    logger.info("[1/4] Consultando balance…")
    try:
        balance = get_account_balance(cfg.hyperliquid)
        logger.info("Balance: $%.2f", balance)
    except Exception as exc:
        logger.error("FAIL — get_account_balance: %s", exc)
        sys.exit(1)

    if balance < TEST_SIZE_USD * 1.1:
        logger.error(
            "FAIL — balance $%.2f < $%.2f (size + buffer). Deposite mais USDC.",
            balance, TEST_SIZE_USD * 1.1,
        )
        sys.exit(1)

    # Etapa 2: size decimals
    logger.info("[2/4] Consultando szDecimals para %s…", coin)
    try:
        decimals = _load_sz_decimals(cfg.hyperliquid.base_url).get(coin)
        if decimals is None:
            logger.error("FAIL — coin %s não está no universe da Hyperliquid", coin)
            sys.exit(1)
        logger.info("szDecimals[%s] = %d", coin, decimals)
    except Exception as exc:
        logger.error("FAIL — _load_sz_decimals: %s", exc)
        sys.exit(1)

    # Etapa 3: place_order LONG mínimo
    logger.info("[3/4] Enviando ordem LONG %s $%.2f…", coin, TEST_SIZE_USD)

    # Calcula SL/TP a partir do mid_price atual (estimativa)
    from hyperliquid.info import Info
    info = Info(cfg.hyperliquid.base_url, skip_ws=True)
    mid = float(info.all_mids().get(coin, 0))
    if mid == 0:
        logger.error("FAIL — mid price=0 para %s", coin)
        sys.exit(1)

    params = OrderParams(
        coin=coin,
        is_buy=True,
        size_usd=TEST_SIZE_USD,
        sl_price=round(mid * (1 - TEST_SL_PCT / 100), 2),
        tp_price=round(mid * (1 + TEST_TP_PCT / 100), 2),
        leverage=LEVERAGE,
    )

    try:
        result = place_order(params, cfg.hyperliquid, cfg.trading)
    except Exception as exc:
        logger.error("FAIL — place_order: %s", exc)
        sys.exit(1)

    if result is None:
        logger.error("FAIL — place_order retornou None (ordem rejeitada ou sem fill)")
        sys.exit(1)

    logger.info("Ordem enviada: %s", result)

    # Etapa 4: aguardar 5s e fechar
    logger.info("[4/4] Aguardando 5s e fechando posição…")
    time.sleep(5)

    try:
        from hyperliquid.exchange import Exchange
        from hyperliquid.utils import constants
        import eth_account
        account = eth_account.Account.from_key(cfg.hyperliquid.private_key)
        exchange = Exchange(
            account,
            constants.MAINNET_API_URL if cfg.hyperliquid.mainnet else constants.TESTNET_API_URL,
        )

        # Bug #2: market_close retornou None silenciosamente no smoke test.
        # Buscamos o size atual da posição via user_state e passamos
        # explicitamente. Se ainda assim falhar, fallback para ordem reverse
        # com reduce_only=True.
        state = info.user_state(cfg.hyperliquid.account_address)
        positions = state.get("assetPositions", [])
        coin_pos = next(
            (p["position"] for p in positions
             if p.get("position", {}).get("coin") == coin),
            None,
        )

        if coin_pos is None:
            logger.warning(
                "Posição %s não consta em assetPositions — pode já estar fechada.",
                coin,
            )
            close_result = {"status": "ok", "note": "no_position"}
        else:
            szi = float(coin_pos["szi"])
            pos_size = abs(szi)
            logger.info("Fechando posição %s size=%.6f (szi=%.6f)…",
                        coin, pos_size, szi)

            close_result = exchange.market_close(coin, sz=pos_size)
            logger.info("Resultado market_close: %s", close_result)

            if (close_result is None
                    or (isinstance(close_result, dict)
                        and close_result.get("status") != "ok")):
                logger.warning(
                    "market_close falhou (resp=%s); tentando ordem reverse "
                    "com reduce_only=True…",
                    close_result,
                )
                # Se a posição é SHORT (szi < 0), reverse é BUY; se LONG, SELL.
                is_buy_inverse = szi < 0
                close_result = exchange.market_open(
                    coin,
                    is_buy_inverse,
                    pos_size,
                    slippage=0.02,
                    reduce_only=True,
                )
                logger.info("Resultado fallback reverse: %s", close_result)
    except Exception as exc:
        logger.error(
            "WARN — falha ao fechar a posição automaticamente: %s. "
            "FECHE MANUALMENTE em app.hyperliquid.xyz!", exc,
        )
        sys.exit(1)

    # Verificação pós-close: confirma que a posição não está mais aberta
    time.sleep(3)
    try:
        state_after = info.user_state(cfg.hyperliquid.account_address)
        positions_after = state_after.get("assetPositions", [])
        still_open = any(
            p.get("position", {}).get("coin") == coin
            and abs(float(p.get("position", {}).get("szi", 0))) > 0
            for p in positions_after
        )
        if still_open:
            logger.error(
                "POSIÇÃO %s AINDA ABERTA — feche MANUALMENTE em "
                "app.hyperliquid.xyz!", coin,
            )
            sys.exit(1)
        logger.info("Posição %s confirmada FECHADA na exchange.", coin)
    except SystemExit:
        raise
    except Exception as exc:
        logger.warning("Falha ao verificar fechamento: %s", exc)

    # Confere balance final
    try:
        balance_after = get_account_balance(cfg.hyperliquid)
        logger.info(
            "Balance final: $%.2f | delta=%.2f (esperado ~ -$0.05 a -$0.20 em fees)",
            balance_after, balance_after - balance,
        )
    except Exception:
        pass

    logger.info("=" * 70)
    logger.info("SMOKE TEST OK — pipeline live validado.")
    logger.info("Próximo passo: editar trading/workflow.py:main para dry_run=False")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
