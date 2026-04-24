"""
Camada 4 — Execução (controle de estado)

Persiste o estado de posições abertas em JSON para que o loop
sobreviva a reinicializações sem criar ordens duplicadas.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

STATE_FILE = Path(__file__).parent.parent / "trading" / ".state.json"


@dataclass
class OpenPosition:
    coin: str
    direction: str        # "LONG" | "SHORT"
    entry_price: float
    size_usd: float
    sl_price: float
    tp_price: float
    opened_at: float      # unix timestamp


class StateManager:
    """
    Lê/escreve estado de posição em disco.
    Thread-safe para uso em loop único.
    """

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if not self.state_file.exists():
            return {}
        with open(self.state_file) as f:
            return json.load(f)

    def _save(self, data: dict):
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_position(self, coin: str) -> Optional[OpenPosition]:
        data = self._load()
        pos = data.get(coin)
        if pos is None:
            return None
        return OpenPosition(**pos)

    def set_position(self, pos: OpenPosition):
        data = self._load()
        data[pos.coin] = asdict(pos)
        self._save(data)

    def clear_position(self, coin: str):
        data = self._load()
        data.pop(coin, None)
        self._save(data)

    def has_open_position(self, coin: str) -> bool:
        return self.get_position(coin) is not None

    def position_age_seconds(self, coin: str) -> float:
        pos = self.get_position(coin)
        if pos is None:
            return 0.0
        return time.time() - pos.opened_at
