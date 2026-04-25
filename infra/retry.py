"""
Decorator de retry com exponential backoff para chamadas de rede flakey
contra a Hyperliquid (Info / Exchange) ou outras APIs.

Uso:
    @retry_on_network_error(max_attempts=3, base_delay=1.0)
    def fetch_data(...): ...

Política:
- Retry SOMENTE em exceções de rede (requests.RequestException, TimeoutError,
  ConnectionError) ou HTTP 5xx.
- NÃO retry em 4xx (input ruim — falhar rápido).
- NÃO usar em operações não-idempotentes (place_order!) — pode causar dupla
  execução. Aplicar apenas em operações de leitura.
"""
import functools
import logging
import time
from typing import Callable

import requests

logger = logging.getLogger(__name__)


def retry_on_network_error(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exponential: bool = True,
) -> Callable:
    """
    Decorator que retenta a função em falhas de rede.
    Backoff: base_delay * (2 ** attempt) se exponential=True, senão constante.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except (
                    requests.exceptions.RequestException,
                    TimeoutError,
                    ConnectionError,
                ) as exc:
                    last_exc = exc
                    if attempt + 1 == max_attempts:
                        logger.error(
                            "%s falhou após %d tentativas: %s",
                            fn.__name__, max_attempts, exc,
                        )
                        raise

                    delay = base_delay * (2 ** attempt) if exponential else base_delay
                    logger.warning(
                        "%s tentativa %d/%d falhou (%s) — retrying em %.1fs",
                        fn.__name__, attempt + 1, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
                except Exception:
                    # 4xx ou erro de aplicação — não retry, propaga.
                    raise
            raise RuntimeError(f"Unreachable: {last_exc}")  # placebo
        return wrapper
    return decorator
