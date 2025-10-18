"""
Módulo de estratégias de agregação para Federated Learning.

Exporta classe base e estratégias específicas.
"""

from .base_strategy import BaseStrategy
from .bagging_strategy import FedBagging
from .cyclic_strategy import FedCyclic

__all__ = [
    'BaseStrategy',
    'FedBagging',
    'FedCyclic',
]
