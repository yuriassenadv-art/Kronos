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
