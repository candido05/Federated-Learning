"""Funções utilitárias para experimentos de Aprendizado Federado.

Este módulo contém funções auxiliares para logging, cálculo de métricas,
visualização e outras operações comuns.
"""

from .helpers import (
    format_device_info,
    get_device_info,
    replace_keys,
    safe_run_simulation,
    sanitize_filename,
    validate_config,
)
from .logging_utils import ExperimentLogger
from .metrics import MetricsCalculator

__all__ = [
    # Métricas
    "MetricsCalculator",
    # Logging
    "ExperimentLogger",
    # Helpers
    "replace_keys",
    "safe_run_simulation",
    "get_device_info",
    "format_device_info",
    "validate_config",
    "sanitize_filename",
]
