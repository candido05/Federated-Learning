"""
Algoritmos de Federated Learning
XGBoost, LightGBM, CatBoost
"""

from .xgboost_fl import run_xgboost_experiment
from .lightgbm_fl import run_lightgbm_experiment
from .catboost_fl import run_catboost_experiment

__all__ = [
    'run_xgboost_experiment',
    'run_lightgbm_experiment',
    'run_catboost_experiment'
]
