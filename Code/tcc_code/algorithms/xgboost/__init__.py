"""
Módulo XGBoost para Federated Learning
"""

from .client import XGBoostClient
from .server import create_server_fn, get_evaluate_fn, config_func
from .runner import run_xgboost_experiment

__all__ = [
    'XGBoostClient',
    'create_server_fn',
    'get_evaluate_fn',
    'config_func',
    'run_xgboost_experiment'
]
