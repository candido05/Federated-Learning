"""Módulo de configuração para o projeto TCC de Federated Learning.

Este módulo contém dataclasses de configuração e hiperparâmetros dos modelos.
"""

from .config import GlobalConfig, LoggingConfig
from .model_params import XGBOOST_PARAMS, CATBOOST_PARAMS, LIGHTGBM_PARAMS

__all__ = [
    "GlobalConfig",
    "LoggingConfig",
    "XGBOOST_PARAMS",
    "CATBOOST_PARAMS",
    "LIGHTGBM_PARAMS",
]
