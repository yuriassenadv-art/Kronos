"""
Testes do sistema multi-asset do bot Kronos (BTC/ETH/SOL).

Cobertura:
  * generate_signals (resilience a falhas isoladas, mocks dos predictors)
  * filter_already_open + StateManager (isolamento de posições por coin)
  * portfolio.allocate_capital (NotImplementedError + mock equal-weight local)
  * run_cycle integrado (NotImplementedError não derruba o loop)

Sem rede, sem GPU — todos os predictors / fetch_candles / place_order são mockados.
"""

import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Garante que o pacote raiz Kronos está no path mesmo se rodado fora da raiz
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# Stub de dependências pesadas (torch / model / infra.ensemble_predictor)
# ─────────────────────────────────────────────────────────────────────────────
# Estes testes são puramente de lógica de orquestração (workflow, state,
# portfolio). Não tocam em GPU, não baixam pesos do HF, não fazem inferência
# real. Sem torch instalado, precisamos pré-popular sys.modules com módulos
# fake ANTES de importar trading.workflow (que faz `from infra.ensemble_predictor
# import EnsemblePredictor`, que por sua vez importa torch indiretamente).
def _ensure_stub(modname: str) -> types.ModuleType:
    if modname not in sys.modules:
        sys.modules[modname] = types.ModuleType(modname)
    return sys.modules[modname]


# Stub de torch (apenas o que `model.kronos` toca em import-time não deve
# crashar — testes não chamam métodos do torch).
_torch_stub = _ensure_stub("torch")
# Submódulos comumente acessados no topo de model/kronos.py
_ensure_stub("torch.nn")
_ensure_stub("torch.nn.functional")
# huggingface_hub também é importado por model.module — stubamos por segurança
_ensure_stub("huggingface_hub")
sys.modules["huggingface_hub"].PyTorchModelHubMixin = type(
    "PyTorchModelHubMixin", (), {}
)
# einops (também pode ser importado por model)
_ensure_stub("einops")

# Stub do pacote model (usado por infra.ensemble_predictor)
_model_stub = _ensure_stub("model")
_model_stub.Kronos = type("Kronos", (), {})
_model_stub.KronosTokenizer = type("KronosTokenizer", (), {})
_model_stub.KronosPredictor = type("KronosPredictor", (), {})

# Stub do EnsemblePredictor — não queremos a implementação real
_ep_stub = _ensure_stub("infra.ensemble_predictor")


class EnsemblePredictor:  # placeholder usado só para spec dos MagicMocks
    def __init__(self, *args, **kwargs):
        pass

    def get_signal(self, df, pred_len, interval):
        raise NotImplementedError("stub")


_ep_stub.EnsemblePredictor = EnsemblePredictor

# Stub do hyperliquid.info.Info — necessário pelos testes de size precision
# do order_executor (e idempotente para outros agentes que stubam o mesmo).
_hl_pkg = _ensure_stub("hyperliquid")
_hl_info_mod = _ensure_stub("hyperliquid.info")
if not hasattr(_hl_info_mod, "Info"):
    class _StubInfo:  # placeholder; testes substituem via patch()
        def __init__(self, *args, **kwargs):
            pass

        def meta(self):
            return {"universe": []}

        def all_mids(self):
            return {}

        def user_state(self, *args, **kwargs):
            return {"marginSummary": {"accountValue": "0"}}

    _hl_info_mod.Info = _StubInfo

# Agora podemos importar com segurança o resto da stack
from trading.config import Config, TradingConfig
from trading.signal_generator import Signal, Direction
from trading.risk_manager import OrderParams, calculate_order
from trading.workflow import (
    generate_signals,
    filter_already_open,
    execute_orders,
    run_cycle,
)
from trading.portfolio import allocate_capital
from infra.state_manager import StateManager, OpenPosition


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def cfg():
    """Config básico com 3 coins (BTC/ETH/SOL) em dry_run."""
    cfg = Config()
    cfg.trading.coins = ["BTC", "ETH", "SOL"]
    cfg.trading.interval = "15m"
    cfg.trading.dry_run = True
    cfg.trading.leverage = 3
    cfg.trading.risk_per_trade_pct = 1.0
    cfg.trading.sl_pct = 1.5
    cfg.trading.tp_pct = 3.0
    cfg.trading.max_concurrent_positions = 3
    cfg.kronos.lookback = 200
    cfg.kronos.pred_len = 10
    cfg.kronos.sample_count = 3
    return cfg


@pytest.fixture
def state(tmp_path):
    """StateManager isolado em tmpdir — não interfere com .state.json real."""
    state_file = tmp_path / "state_test.json"
    return StateManager(state_file=state_file)


@pytest.fixture
def fake_candles_df():
    """DataFrame OHLCV mínimo para satisfazer fetch_candles mockado."""
    return pd.DataFrame(
        {
            "open":   [100.0, 101.0, 102.0],
            "high":   [101.5, 102.5, 103.5],
            "low":    [ 99.5, 100.5, 101.5],
            "close":  [101.0, 102.0, 103.0],
            "volume": [10.0,  12.0,  11.0],
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestEnsembleSignals — generate_signals com predictors mockados
# ─────────────────────────────────────────────────────────────────────────────
class TestEnsembleSignals:
    """Cobre o pipeline de geração de sinais com mocks dos predictors."""

    def test_generate_signals_returns_three_signals_when_all_succeed(
        self, cfg, fake_candles_df
    ):
        # Mocks de 3 EnsemblePredictors retornando 3 sinais distintos
        btc_signal = Signal(
            direction=Direction.LONG,
            confidence=0.85,
            entry_price=50000.0,
            forecast_horizon=10,
            reason="expected_return=+2.5%",
        )
        eth_signal = Signal(
            direction=Direction.SHORT,
            confidence=0.70,
            entry_price=3000.0,
            forecast_horizon=10,
            reason="expected_return=-1.8%",
        )
        sol_signal = Signal(
            direction=Direction.FLAT,
            confidence=0.20,
            entry_price=150.0,
            forecast_horizon=10,
            reason="confidence baixa",
        )

        ens_btc = MagicMock()
        ens_btc.get_signal.return_value = btc_signal
        ens_eth = MagicMock()
        ens_eth.get_signal.return_value = eth_signal
        ens_sol = MagicMock()
        ens_sol.get_signal.return_value = sol_signal

        ensembles = {"BTC": ens_btc, "ETH": ens_eth, "SOL": ens_sol}

        # Mock de fetch_candles e do monitor (ambos importados pelo workflow)
        with patch(
            "trading.workflow.fetch_candles", return_value=fake_candles_df
        ) as mock_fetch, patch("trading.workflow.monitor") as mock_monitor:
            signals = generate_signals(ensembles, cfg)

        # 3 chaves no dict de retorno
        assert set(signals.keys()) == {"BTC", "ETH", "SOL"}
        # Cada Signal preserva seus atributos
        assert signals["BTC"].direction == Direction.LONG
        assert signals["BTC"].confidence == 0.85
        assert signals["BTC"].entry_price == 50000.0
        assert signals["ETH"].direction == Direction.SHORT
        assert signals["SOL"].direction == Direction.FLAT
        # fetch_candles foi chamado 3 vezes (1 por coin)
        assert mock_fetch.call_count == 3
        # monitor.alert_signal chamado para cada coin
        assert mock_monitor.alert_signal.call_count == 3

    def test_generate_signals_resilient_to_single_coin_failure(
        self, cfg, fake_candles_df
    ):
        # ETH explode no get_signal — BTC e SOL devem continuar funcionando
        btc_signal = Signal(
            direction=Direction.LONG,
            confidence=0.80,
            entry_price=50000.0,
            forecast_horizon=10,
            reason="ok",
        )
        sol_signal = Signal(
            direction=Direction.LONG,
            confidence=0.65,
            entry_price=150.0,
            forecast_horizon=10,
            reason="ok",
        )

        ens_btc = MagicMock()
        ens_btc.get_signal.return_value = btc_signal
        ens_eth = MagicMock()
        ens_eth.get_signal.side_effect = RuntimeError("Falha simulada no ETH")
        ens_sol = MagicMock()
        ens_sol.get_signal.return_value = sol_signal

        ensembles = {"BTC": ens_btc, "ETH": ens_eth, "SOL": ens_sol}

        with patch(
            "trading.workflow.fetch_candles", return_value=fake_candles_df
        ), patch("trading.workflow.monitor") as mock_monitor:
            signals = generate_signals(ensembles, cfg)

        # BTC e SOL presentes; ETH ausente
        assert "BTC" in signals
        assert "SOL" in signals
        assert "ETH" not in signals
        # Verifica que monitor.alert_error foi chamado para ETH
        error_calls = [
            call for call in mock_monitor.alert_error.call_args_list
            if "ETH" in str(call)
        ]
        assert len(error_calls) >= 1, (
            "monitor.alert_error deveria ter sido chamado para o ETH"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestStateIsolation — StateManager isola posições por coin
# ─────────────────────────────────────────────────────────────────────────────
class TestStateIsolation:
    """Garante que StateManager não confunde estado entre BTC/ETH/SOL."""

    def test_state_manager_isolates_positions_per_coin(self, state):
        # Adiciona posição BTC
        btc_pos = OpenPosition(
            coin="BTC",
            direction="LONG",
            entry_price=50000.0,
            size_usd=30.0,
            sl_price=49250.0,
            tp_price=51500.0,
            opened_at=time.time(),
        )
        state.set_position(btc_pos)

        # BTC presente, ETH ausente
        assert state.has_open_position("BTC") is True
        assert state.has_open_position("ETH") is False
        assert state.count_open() == 1

        # Adiciona posição ETH
        eth_pos = OpenPosition(
            coin="ETH",
            direction="SHORT",
            entry_price=3000.0,
            size_usd=30.0,
            sl_price=3045.0,
            tp_price=2910.0,
            opened_at=time.time(),
        )
        state.set_position(eth_pos)

        # 2 posições abertas, ambos coins listados
        assert state.count_open() == 2
        assert set(state.list_open_coins()) == {"BTC", "ETH"}

        # Remove só BTC; ETH permanece
        state.clear_position("BTC")
        assert state.has_open_position("BTC") is False
        assert state.has_open_position("ETH") is True
        all_pos = state.all_positions()
        assert set(all_pos.keys()) == {"ETH"}
        assert all_pos["ETH"].direction == "SHORT"

    def test_filter_already_open_removes_signals_for_open_coins(self, state):
        # 3 sinais simultâneos
        signals = {
            "BTC": Signal(Direction.LONG, 0.8, 50000.0, 10, "btc"),
            "ETH": Signal(Direction.SHORT, 0.7, 3000.0, 10, "eth"),
            "SOL": Signal(Direction.LONG, 0.6, 150.0, 10, "sol"),
        }
        # Adiciona posição BTC já aberta
        state.set_position(
            OpenPosition(
                coin="BTC",
                direction="LONG",
                entry_price=49000.0,
                size_usd=30.0,
                sl_price=48000.0,
                tp_price=50500.0,
                opened_at=time.time(),
            )
        )

        filtered = filter_already_open(signals, state)

        # BTC removido (já tem posição); ETH e SOL preservados
        assert "BTC" not in filtered
        assert "ETH" in filtered
        assert "SOL" in filtered
        assert len(filtered) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestPortfolio — allocate_capital stub + mock local equal-weight
# ─────────────────────────────────────────────────────────────────────────────
def _mock_equal_weight_allocate(
    signals: dict[str, Signal],
    balance: float,
    cfg: TradingConfig,
) -> dict[str, OrderParams]:
    """
    Mock LOCAL de allocate_capital — abordagem A (equal-weight) do docstring
    de portfolio.allocate_capital. NÃO sobrescreve o módulo, é só pra testar
    o contrato esperado.

    Filtra Direction.FLAT, divide risk_per_trade_pct igualmente entre os
    sinais ativos e chama calculate_order para cada um com cfg modificado.
    """
    from dataclasses import replace

    active = {c: s for c, s in signals.items() if s.direction != Direction.FLAT}
    if not active:
        return {}

    # Cópia ajustada de cfg sem mutar o original (per-coin risk = total / N)
    n = len(active)
    per_coin_risk = cfg.risk_per_trade_pct / n

    out: dict[str, OrderParams] = {}
    for coin, sig in active.items():
        cfg_coin = replace(cfg, risk_per_trade_pct=per_coin_risk, coin=coin)
        params = calculate_order(sig, balance, cfg_coin)
        if params is not None:
            out[coin] = params
    return out


class TestPortfolio:
    """Cobre allocate_capital real (Filosofia B confidence-weighted) + mock equal-weight."""

    def test_allocate_capital_confidence_weighted_filters_flat_and_weights(self, cfg):
        """Implementação ativa (Filosofia B): aloca proporcional a confidence,
        filtra Direction.FLAT, respeita max_concurrent_positions."""
        signals = {
            "BTC": Signal(Direction.LONG, 0.80, 50000.0, 10, "btc forte"),
            "ETH": Signal(Direction.SHORT, 0.40, 3000.0, 10, "eth medio"),
            "SOL": Signal(Direction.FLAT, 0.10, 150.0, 10, "sol flat"),
        }
        result = allocate_capital(signals, balance=1000.0, cfg=cfg.trading)

        # SOL FLAT removido; BTC + ETH presentes
        assert set(result.keys()) == {"BTC", "ETH"}

        # Pesos confidence-weighted: BTC=0.8/(0.8+0.4)=0.667, ETH=0.4/1.2=0.333
        # risk_per_trade_pct=1.0%, balance=1000, leverage=3
        # BTC size = 1000 × (1.0 × 0.667 / 100) × 3 = $20.00
        # ETH size = 1000 × (1.0 × 0.333 / 100) × 3 = $10.00
        assert result["BTC"].size_usd == pytest.approx(20.0, rel=1e-3)
        assert result["ETH"].size_usd == pytest.approx(10.0, rel=1e-3)

        # BTC peso > ETH peso (alocação reflete confidence relativa)
        assert result["BTC"].size_usd > result["ETH"].size_usd

        # Direções preservadas
        assert result["BTC"].is_buy is True
        assert result["ETH"].is_buy is False

        # Total alocado ~= risk × leverage = $30 (full budget consumido)
        total = sum(p.size_usd for p in result.values())
        assert total == pytest.approx(30.0, rel=1e-3)

    def test_allocate_capital_returns_empty_when_all_flat(self, cfg):
        """Quando todos os sinais são FLAT, retorna dict vazio (sem erro)."""
        signals = {
            "BTC": Signal(Direction.FLAT, 0.10, 50000.0, 10, "flat"),
            "ETH": Signal(Direction.FLAT, 0.05, 3000.0, 10, "flat"),
        }
        result = allocate_capital(signals, balance=1000.0, cfg=cfg.trading)
        assert result == {}

    def test_allocate_capital_caps_at_max_concurrent_positions(self, cfg):
        """Se mais coins ativos que max_concurrent_positions, top-N por confidence vence."""
        cfg.trading.max_concurrent_positions = 2
        signals = {
            "BTC": Signal(Direction.LONG, 0.90, 50000.0, 10, "btc top"),
            "ETH": Signal(Direction.LONG, 0.70, 3000.0, 10, "eth meio"),
            "SOL": Signal(Direction.LONG, 0.30, 150.0, 10, "sol fraco"),
        }
        result = allocate_capital(signals, balance=1000.0, cfg=cfg.trading)

        # SOL (menor confidence) deve ter sido cortado pelo cap
        assert set(result.keys()) == {"BTC", "ETH"}
        assert "SOL" not in result

    def test_mock_equal_weight_allocation_divides_capital(self, cfg):
        # 3 sinais: 2 não-FLAT (BTC/ETH) + 1 FLAT (SOL, deve ser filtrado)
        signals = {
            "BTC": Signal(Direction.LONG, 0.8, 50000.0, 10, "btc"),
            "ETH": Signal(Direction.SHORT, 0.7, 3000.0, 10, "eth"),
            "SOL": Signal(Direction.FLAT, 0.2, 150.0, 10, "flat"),
        }

        result = _mock_equal_weight_allocate(
            signals, balance=1000.0, cfg=cfg.trading
        )

        # SOL filtrado — só BTC e ETH
        assert set(result.keys()) == {"BTC", "ETH"}
        assert len(result) == 2

        # Cada size_usd = 1000 × (1.0 / 2 / 100) × 3 = 15.0
        # (risk_per_trade_pct=1.0 dividido por 2 coins = 0.5%; balance × 0.5% × leverage 3)
        expected_size = 1000.0 * (0.5 / 100.0) * 3
        assert result["BTC"].size_usd == pytest.approx(expected_size)
        assert result["ETH"].size_usd == pytest.approx(expected_size)

        # Total alocado = 2 × 15.0 = 30.0
        total = sum(p.size_usd for p in result.values())
        assert total == pytest.approx(2 * expected_size)

        # Direções corretas
        assert result["BTC"].is_buy is True   # LONG
        assert result["ETH"].is_buy is False  # SHORT


# ─────────────────────────────────────────────────────────────────────────────
# TestSizePrecision — _load_sz_decimals e _round_qty (BLOCKER #3)
# ─────────────────────────────────────────────────────────────────────────────
class TestSizePrecision:
    """Cobre _load_sz_decimals e _round_qty para size precision por coin."""

    def setup_method(self):
        # Limpa cache entre testes
        from trading.order_executor import clear_decimals_cache
        clear_decimals_cache()

    def test_round_qty_uses_per_coin_decimals(self):
        from trading.order_executor import _round_qty
        from unittest.mock import patch

        fake_meta = {
            "universe": [
                {"name": "BTC", "szDecimals": 5},
                {"name": "ETH", "szDecimals": 4},
                {"name": "SOL", "szDecimals": 2},
            ]
        }
        with patch("trading.order_executor.Info") as mock_info_cls:
            mock_info_cls.return_value.meta.return_value = fake_meta

            # BTC permite 5 decimais
            assert _round_qty("BTC", 0.123456789, "https://x") == 0.12346
            # ETH limita a 4
            assert _round_qty("ETH", 0.123456789, "https://x") == 0.1235
            # SOL limita a 2
            assert _round_qty("SOL", 1.234567, "https://x") == 1.23

    def test_round_qty_fallback_for_unknown_coin(self):
        from trading.order_executor import _round_qty
        from unittest.mock import patch

        with patch("trading.order_executor.Info") as mock_info_cls:
            mock_info_cls.return_value.meta.return_value = {"universe": []}
            # Coin desconhecido cai no fallback (4 decimais)
            assert _round_qty("XYZ", 0.123456, "https://x") == 0.1235


# ─────────────────────────────────────────────────────────────────────────────
# TestPositionWatcher — sync_positions reconcilia state com a exchange
# ─────────────────────────────────────────────────────────────────────────────
class TestPositionWatcher:
    """Cobre sync_positions: fecha state quando exchange não tem mais a posição."""

    def test_sync_positions_dry_run_returns_empty(self, state):
        from trading.config import HyperliquidConfig
        from infra.position_watcher import sync_positions
        # dry_run=True nunca consulta a exchange
        closed = sync_positions(state, HyperliquidConfig(), dry_run=True)
        assert closed == []

    def test_sync_positions_clears_state_when_position_closed_on_exchange(
        self, state
    ):
        from trading.config import HyperliquidConfig
        from infra.position_watcher import sync_positions

        # Pré-popula state com posição BTC
        state.set_position(OpenPosition(
            coin="BTC", direction="LONG", entry_price=50000.0,
            size_usd=30.0, sl_price=49250.0, tp_price=51500.0,
            opened_at=time.time(),
        ))

        # Mock: exchange retorna nenhuma posição aberta
        fake_user_state = {"assetPositions": [], "marginSummary": {}}
        with patch(
            "infra.position_watcher.Info"
        ) as mock_info_cls, patch(
            "infra.position_watcher.monitor"
        ):
            mock_info_cls.return_value.user_state.return_value = fake_user_state
            closed = sync_positions(
                state, HyperliquidConfig(), dry_run=False
            )

        # State deve ter sido limpo
        assert "BTC" in closed
        assert state.has_open_position("BTC") is False

    def test_sync_positions_keeps_state_when_position_still_open(
        self, state
    ):
        from trading.config import HyperliquidConfig
        from infra.position_watcher import sync_positions

        state.set_position(OpenPosition(
            coin="BTC", direction="LONG", entry_price=50000.0,
            size_usd=30.0, sl_price=49250.0, tp_price=51500.0,
            opened_at=time.time(),
        ))

        # Mock: exchange ainda mostra BTC com size > 0
        fake_user_state = {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "0.001", "entryPx": "50000"}}
            ],
            "marginSummary": {},
        }
        with patch(
            "infra.position_watcher.Info"
        ) as mock_info_cls, patch(
            "infra.position_watcher.monitor"
        ):
            mock_info_cls.return_value.user_state.return_value = fake_user_state
            closed = sync_positions(
                state, HyperliquidConfig(), dry_run=False
            )

        # Posição segue aberta, nada limpo
        assert closed == []
        assert state.has_open_position("BTC") is True


# ─────────────────────────────────────────────────────────────────────────────
# TestRetryDecorator — retry_on_network_error com backoff exponencial
# ─────────────────────────────────────────────────────────────────────────────
class TestRetryDecorator:
    """Cobre retry_on_network_error: retry em network error, fail fast em 4xx."""

    def test_retry_succeeds_after_transient_failure(self):
        from infra.retry import retry_on_network_error
        import requests
        calls = []

        @retry_on_network_error(max_attempts=3, base_delay=0.01)
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise requests.exceptions.ConnectionError("transient")
            return "ok"

        assert flaky() == "ok"
        assert len(calls) == 3

    def test_retry_gives_up_after_max_attempts(self):
        from infra.retry import retry_on_network_error
        import requests

        @retry_on_network_error(max_attempts=2, base_delay=0.01)
        def always_fails():
            raise requests.exceptions.ConnectionError("nope")

        with pytest.raises(requests.exceptions.ConnectionError):
            always_fails()

    def test_retry_does_not_retry_on_value_error(self):
        from infra.retry import retry_on_network_error
        calls = []

        @retry_on_network_error(max_attempts=3, base_delay=0.01)
        def bad_input():
            calls.append(1)
            raise ValueError("4xx-like")

        with pytest.raises(ValueError):
            bad_input()
        assert len(calls) == 1  # nenhum retry


# ─────────────────────────────────────────────────────────────────────────────
# TestOrderFillVerification — validação de status / fills no place_order
# ─────────────────────────────────────────────────────────────────────────────
class TestOrderFillVerification:
    """Cobre validação de status / fills no place_order."""

    def setup_method(self):
        # Garante que cada teste reinicializa cache de szDecimals
        from trading.order_executor import clear_decimals_cache
        clear_decimals_cache()

    def _stub_hyperliquid_modules(self):
        """
        Garante que os módulos lazy-imported por place_order existem em
        sys.modules antes do patch (Exchange, hyperliquid.utils.constants,
        eth_account). Sem isso, `from hyperliquid.exchange import Exchange`
        dentro da função estoura ImportError.
        """
        _ensure_stub("hyperliquid.exchange")
        if not hasattr(sys.modules["hyperliquid.exchange"], "Exchange"):
            sys.modules["hyperliquid.exchange"].Exchange = type(
                "Exchange", (), {}
            )
        _ensure_stub("hyperliquid.utils")
        _ensure_stub("hyperliquid.utils.constants")
        sys.modules["hyperliquid.utils"].constants = sys.modules[
            "hyperliquid.utils.constants"
        ]
        sys.modules["hyperliquid.utils.constants"].MAINNET_API_URL = "x"
        sys.modules["hyperliquid.utils.constants"].TESTNET_API_URL = "y"
        _ensure_stub("eth_account")
        if not hasattr(sys.modules["eth_account"], "Account"):
            class _StubAccount:
                @staticmethod
                def from_key(_key):
                    return None
            sys.modules["eth_account"].Account = _StubAccount

    def test_place_order_returns_none_when_status_err(self, cfg):
        from trading.order_executor import place_order
        from trading.risk_manager import OrderParams
        from unittest.mock import patch, MagicMock

        self._stub_hyperliquid_modules()

        params = OrderParams(
            coin="BTC", is_buy=True, size_usd=30.0,
            sl_price=49000.0, tp_price=51000.0, leverage=3,
        )
        cfg.trading.dry_run = False  # força fluxo live para validar

        # Info / Exchange / eth_account / constants — mockados nas suas origens
        # (Exchange e eth_account são lazy-imported dentro de place_order).
        mock_info = MagicMock()
        mock_info.all_mids.return_value = {"BTC": "50000"}
        mock_info.meta.return_value = {
            "universe": [{"name": "BTC", "szDecimals": 5}]
        }
        mock_exchange = MagicMock()
        mock_exchange.market_open.return_value = {
            "status": "err", "response": "rejected"
        }
        mock_account = MagicMock()

        with patch("trading.order_executor.Info", return_value=mock_info), \
             patch(
                 "hyperliquid.exchange.Exchange", return_value=mock_exchange
             ), \
             patch(
                 "eth_account.Account.from_key", return_value=mock_account
             ), \
             patch("trading.order_executor.monitor"):
            result = place_order(params, cfg.hyperliquid, cfg.trading)

        # Status err → return None, NÃO envia SL/TP
        assert result is None
        mock_exchange.order.assert_not_called()  # SL/TP nunca enviado


# ─────────────────────────────────────────────────────────────────────────────
# TestHealthCheck — build_ensembles_with_retry + ensemble_health_check
# ─────────────────────────────────────────────────────────────────────────────
class TestHealthCheck:
    """Cobre build_ensembles_with_retry e ensemble_health_check."""

    def setup_method(self):
        from infra.health_check import reset_health_state
        reset_health_state()

    def test_build_ensembles_with_retry_succeeds_after_transient_failure(self):
        from infra.health_check import build_ensembles_with_retry
        from unittest.mock import MagicMock
        import time as time_mod

        attempts = []

        def flaky_builder():
            attempts.append(1)
            if len(attempts) < 2:
                raise RuntimeError("disco lento")
            return {"BTC": MagicMock()}

        # Patch sleep para não esperar de verdade no teste
        with patch.object(time_mod, "sleep"):
            from infra import health_check
            with patch.object(health_check, "time") as mock_time:
                ensembles = build_ensembles_with_retry(
                    flaky_builder, max_attempts=3, base_delay=0.01,
                )
        assert "BTC" in ensembles
        assert len(attempts) == 2

    def test_build_ensembles_with_retry_exits_after_max_attempts(self):
        from infra.health_check import build_ensembles_with_retry

        def always_fails():
            raise RuntimeError("modelo corrompido")

        with patch("infra.health_check.time"):
            with pytest.raises(SystemExit) as exc_info:
                build_ensembles_with_retry(
                    always_fails, max_attempts=2, base_delay=0.01,
                )
        assert exc_info.value.code == 1

    def test_ensemble_health_check_passes_when_at_least_one_ok(self):
        from infra.health_check import ensemble_health_check, reset_health_state

        reset_health_state()

        ok_ensemble = MagicMock()
        ok_ensemble.get_signal.return_value = MagicMock()

        bad_ensemble = MagicMock()
        bad_ensemble.get_signal.side_effect = RuntimeError("OOM")

        result = ensemble_health_check(
            {"BTC": ok_ensemble, "ETH": bad_ensemble}
        )
        assert result is True

    def test_ensemble_health_check_exits_after_consecutive_total_failures(self):
        from infra.health_check import ensemble_health_check, reset_health_state

        reset_health_state()

        bad_ensemble = MagicMock()
        bad_ensemble.get_signal.side_effect = RuntimeError("OOM")

        # Primeira falha: não sai (< 2 consecutivos)
        result = ensemble_health_check({"BTC": bad_ensemble})
        assert result is False

        # Segunda falha total → sys.exit(1)
        with pytest.raises(SystemExit) as exc_info:
            ensemble_health_check({"BTC": bad_ensemble})
        assert exc_info.value.code == 1


# ─────────────────────────────────────────────────────────────────────────────
# TestFundingPenalty — funding_penalty: penaliza LONG/SHORT em funding
# adverso, sem penalidade quando funding está abaixo do threshold.
# ─────────────────────────────────────────────────────────────────────────────
class TestFundingPenalty:
    """Cobre funding_penalty: penaliza LONG em funding alto e SHORT em funding negativo."""

    def setup_method(self):
        from infra.funding import clear_cache
        clear_cache()

    def test_no_penalty_when_funding_neutral(self):
        from infra.funding import funding_penalty, _FUNDING_CACHE
        _FUNDING_CACHE["BTC"] = 0.00005  # abaixo do threshold padrão
        assert funding_penalty("BTC", is_long=True) == 1.0
        assert funding_penalty("BTC", is_long=False) == 1.0

    def test_long_penalty_when_funding_high_positive(self):
        from infra.funding import funding_penalty, _FUNDING_CACHE
        _FUNDING_CACHE["BTC"] = 0.0005  # 5x o threshold
        # LONG paga funding positivo → penalidade máxima (0.5)
        assert funding_penalty("BTC", is_long=True) == pytest.approx(0.5)
        # SHORT recebe → sem penalidade
        assert funding_penalty("BTC", is_long=False) == 1.0

    def test_short_penalty_when_funding_high_negative(self):
        from infra.funding import funding_penalty, _FUNDING_CACHE
        _FUNDING_CACHE["BTC"] = -0.0005  # 5x abaixo do threshold negativo
        # SHORT paga funding negativo → penalidade máxima
        assert funding_penalty("BTC", is_long=False) == pytest.approx(0.5)
        # LONG recebe → sem penalidade
        assert funding_penalty("BTC", is_long=True) == 1.0

    def test_penalty_floor_at_05(self):
        from infra.funding import funding_penalty, _FUNDING_CACHE
        _FUNDING_CACHE["BTC"] = 1.0  # absurdamente alto
        # Mesmo em valores extremos, floor em 0.5
        assert funding_penalty("BTC", is_long=True) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# TestPortfolioWithFunding — integração de funding_penalty no allocate_capital
# ─────────────────────────────────────────────────────────────────────────────
class TestPortfolioWithFunding:
    """Cobre integração de funding_penalty no allocate_capital."""

    def setup_method(self):
        from infra.funding import clear_cache
        clear_cache()

    def test_allocate_capital_reduces_long_weight_when_funding_high(self, cfg):
        from infra.funding import _FUNDING_CACHE
        from trading.portfolio import allocate_capital

        # Funding alto positivo no BTC penaliza LONG
        _FUNDING_CACHE["BTC"] = 0.0005   # 5x threshold
        _FUNDING_CACHE["ETH"] = 0.0       # neutro

        signals = {
            "BTC": Signal(Direction.LONG, 0.8, 50000.0, 10, "btc"),
            "ETH": Signal(Direction.LONG, 0.4, 3000.0, 10, "eth"),
        }
        result = allocate_capital(signals, balance=1000.0, cfg=cfg.trading)

        # BTC ainda deve receber maior alocação (confidence 0.8 > 0.4),
        # mas com penalidade aplicada — proporção BTC/ETH deve ser menor
        # que confidence pura (0.8/0.4 = 2.0)
        ratio = result["BTC"].size_usd / result["ETH"].size_usd
        # Com penalty 0.5 em BTC: peso BTC = 0.8 × 0.5 = 0.4, ETH = 0.4 × 1.0 = 0.4
        # Ratio esperado ~= 1.0 (igualdade após penalidade)
        assert ratio < 2.0  # penalidade reduz a vantagem do BTC
        assert ratio == pytest.approx(1.0, rel=1e-2)


# ─────────────────────────────────────────────────────────────────────────────
# TestCredentialValidation — _validate_credentials no workflow
# ─────────────────────────────────────────────────────────────────────────────
class TestCredentialValidation:
    """Cobre _validate_credentials em workflow.main."""

    def test_validate_skips_in_dry_run(self, cfg):
        from trading.workflow import _validate_credentials
        cfg.trading.dry_run = True
        cfg.hyperliquid.private_key = ""
        cfg.hyperliquid.account_address = ""
        # Não deve raise nem sys.exit
        _validate_credentials(cfg)

    def test_validate_exits_when_credentials_missing_in_live(self, cfg):
        from trading.workflow import _validate_credentials
        cfg.trading.dry_run = False
        cfg.hyperliquid.private_key = ""
        cfg.hyperliquid.account_address = ""
        with pytest.raises(SystemExit) as exc_info:
            _validate_credentials(cfg)
        assert exc_info.value.code == 1

    def test_validate_exits_on_malformed_private_key(self, cfg):
        from trading.workflow import _validate_credentials
        cfg.trading.dry_run = False
        cfg.hyperliquid.private_key = "not_hex"
        cfg.hyperliquid.account_address = "0x" + "a" * 40
        with pytest.raises(SystemExit) as exc_info:
            _validate_credentials(cfg)
        assert exc_info.value.code == 1

    def test_validate_exits_on_low_balance(self, cfg):
        from trading.workflow import _validate_credentials
        cfg.trading.dry_run = False
        cfg.hyperliquid.private_key = "0x" + "a" * 64
        cfg.hyperliquid.account_address = "0x" + "b" * 40
        with patch("trading.workflow.get_account_balance", return_value=5.0):
            with pytest.raises(SystemExit) as exc_info:
                _validate_credentials(cfg)
        assert exc_info.value.code == 1

    def test_validate_passes_with_valid_credentials_and_balance(self, cfg):
        from trading.workflow import _validate_credentials
        cfg.trading.dry_run = False
        cfg.hyperliquid.private_key = "0x" + "a" * 64
        cfg.hyperliquid.account_address = "0x" + "b" * 40
        with patch("trading.workflow.get_account_balance", return_value=500.0):
            # Não deve raise
            _validate_credentials(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# TestPriceRounding — _round_price (Bug #1: SL/TP price tick rounding)
# ─────────────────────────────────────────────────────────────────────────────
class TestPriceRounding:
    """Cobre _round_price para regras de tick da Hyperliquid.

    Regras:
      - max (6 - sz_decimals) decimal places
      - max 5 significant figures
    A regra mais restritiva (menor precisão) ganha.
    """

    def test_btc_high_price_rounds_to_integer(self):
        from trading.order_executor import _round_price
        # BTC szDecimals=5 → max 1 decimal AND 5 sig figs.
        # 77485.55: magnitude=4 → sig_fig_decimals = 4-4 = 0 → integer.
        # min(1, 0) = 0 → arredonda para inteiro
        result = _round_price(77485.55, 5)
        assert result == 77486 or result == 77485

    def test_eth_mid_price_one_decimal(self):
        from trading.order_executor import _round_price
        # ETH szDecimals=4 → max 2 decimals AND 5 sig figs.
        # 3500.567: magnitude=3 → sig_fig_decimals = 4-3 = 1.
        # min(2, 1) = 1 decimal → 3500.6
        result = _round_price(3500.567, 4)
        assert result == pytest.approx(3500.6, abs=0.01)

    def test_sol_low_price_two_decimals(self):
        from trading.order_executor import _round_price
        # SOL szDecimals=2 → max 4 decimals AND 5 sig figs.
        # 145.327: magnitude=2 → sig_fig_decimals = 4-2 = 2.
        # min(4, 2) = 2 decimal → 145.33
        result = _round_price(145.327, 2)
        assert result == pytest.approx(145.33, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# TestUnifiedAccountBalance — get_account_balance (Bug #3: double-count fix)
# ─────────────────────────────────────────────────────────────────────────────
class TestUnifiedAccountBalance:
    """Cobre get_account_balance em modo Unified vs Classic.

    Garante que NÃO há double-count quando há posição aberta em conta
    Unified (smoke test mostrou $19.98 + $10.84 = $30.82 falso).
    """

    def test_classic_account_uses_withdrawable(self):
        from trading.order_executor import get_account_balance
        from trading.config import HyperliquidConfig

        with patch("trading.order_executor.Info") as mock_info_cls:
            mock_info = MagicMock()
            mock_info.user_state.return_value = {
                "marginSummary": {"accountValue": "100.0"},
                "withdrawable": "100.0",
            }
            mock_info.spot_user_state.return_value = {"balances": []}
            mock_info_cls.return_value = mock_info

            balance = get_account_balance(HyperliquidConfig())
            assert balance == pytest.approx(100.0)

    def test_unified_account_no_position_uses_spot_usdc(self):
        from trading.order_executor import get_account_balance
        from trading.config import HyperliquidConfig

        with patch("trading.order_executor.Info") as mock_info_cls:
            mock_info = MagicMock()
            mock_info.user_state.return_value = {
                "marginSummary": {"accountValue": "0.0"},
                "withdrawable": "0.0",
            }
            mock_info.spot_user_state.return_value = {
                "balances": [{"coin": "USDC", "total": "19.98"}]
            }
            mock_info_cls.return_value = mock_info

            balance = get_account_balance(HyperliquidConfig())
            assert balance == pytest.approx(19.98)

    def test_unified_account_with_position_no_double_count(self):
        from trading.order_executor import get_account_balance
        from trading.config import HyperliquidConfig

        # Cenário do incidente real: $19.98 spot, $10 alocado em posição perp.
        # Soma = $29.98 (errado); max = $19.98 (correto — capital real).
        with patch("trading.order_executor.Info") as mock_info_cls:
            mock_info = MagicMock()
            mock_info.user_state.return_value = {
                "marginSummary": {"accountValue": "10.0"},
                "withdrawable": "9.98",  # margem após alocação
            }
            mock_info.spot_user_state.return_value = {
                "balances": [{"coin": "USDC", "total": "19.98"}]
            }
            mock_info_cls.return_value = mock_info

            balance = get_account_balance(HyperliquidConfig())
            # max(9.98, 19.98) = 19.98 — NÃO 29.98
            assert balance == pytest.approx(19.98)
            assert balance < 25.0


# ─────────────────────────────────────────────────────────────────────────────
# TestWorkflowIntegration — run_cycle não crashea quando portfolio falha
# ─────────────────────────────────────────────────────────────────────────────
class TestWorkflowIntegration:
    """High-level: integração end-to-end de run_cycle com Filosofia B."""

    def test_run_cycle_executes_orders_when_signals_active(
        self, cfg, state, fake_candles_df
    ):
        """Com allocate_capital implementado (Filosofia B), run_cycle deve
        gerar sinais → alocar → executar ordem em dry_run."""
        btc_signal = Signal(
            direction=Direction.LONG,
            confidence=0.80,
            entry_price=50000.0,
            forecast_horizon=10,
            reason="long forte",
        )
        ens_btc = MagicMock()
        ens_btc.get_signal.return_value = btc_signal
        ensembles = {"BTC": ens_btc}
        cfg.trading.coins = ["BTC"]

        with patch(
            "trading.workflow.fetch_candles", return_value=fake_candles_df
        ), patch("trading.workflow.monitor") as mock_monitor, patch(
            "trading.workflow.place_order"
        ) as mock_place_order:
            run_cycle(ensembles, state, cfg, cycle_num=1)

        # place_order foi chamado pra BTC (dry_run só loga, mas a função roda)
        assert mock_place_order.call_count == 1

        # Estado persistido: posição BTC aberta com direction LONG
        assert state.has_open_position("BTC") is True
        pos = state.get_position("BTC")
        assert pos.direction == "LONG"
        assert pos.entry_price == 50000.0

        # Telegram: alert_signal e alert_order_placed chamados (cycle_start +1)
        assert mock_monitor.alert_signal.call_count == 1
        assert mock_monitor.alert_order_placed.call_count == 1
        assert mock_monitor.alert_cycle_start.call_count == 1

    def test_run_cycle_skips_when_position_already_open(
        self, cfg, state, fake_candles_df
    ):
        """Se BTC já tem posição aberta, sinal é descartado e nada é executado."""
        # Pré-popula state com posição BTC
        state.set_position(
            OpenPosition(
                coin="BTC",
                direction="LONG",
                entry_price=49000.0,
                size_usd=30.0,
                sl_price=48000.0,
                tp_price=50500.0,
                opened_at=time.time(),
            )
        )

        btc_signal = Signal(
            direction=Direction.LONG,
            confidence=0.80,
            entry_price=50000.0,
            forecast_horizon=10,
            reason="long",
        )
        ens_btc = MagicMock()
        ens_btc.get_signal.return_value = btc_signal
        ensembles = {"BTC": ens_btc}
        cfg.trading.coins = ["BTC"]

        with patch(
            "trading.workflow.fetch_candles", return_value=fake_candles_df
        ), patch("trading.workflow.monitor"), patch(
            "trading.workflow.place_order"
        ) as mock_place_order:
            run_cycle(ensembles, state, cfg, cycle_num=2)

        # Nenhuma nova ordem enviada (posição já estava aberta)
        mock_place_order.assert_not_called()
