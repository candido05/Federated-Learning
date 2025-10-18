"""
Módulo de modelos para Federated Learning.

Exporta classes base e clientes específicos para cada framework.
"""

from .base_client import BaseFLClient
from .xgboost_client import XGBoostClient
from .catboost_client import CatBoostClient
from .lightgbm_client import LightGBMClient

__all__ = [
    'BaseFLClient',
    'XGBoostClient',
    'CatBoostClient',
    'LightGBMClient',
]
