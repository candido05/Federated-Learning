"""Implementações de modelos para Aprendizado Federado.

Este módulo contém wrappers e implementações para modelos baseados em árvores
(XGBoost, LightGBM, CatBoost) adaptados para aprendizado federado.
"""

from .base_client import BaseFLClient

__all__ = [
    "BaseFLClient",
]
