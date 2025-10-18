"""
Módulo do servidor Federated Learning.

Exporta gerenciador de servidor e funções de avaliação.
"""

from .server_manager import FederatedServer
from .evaluation import get_evaluate_fn

__all__ = [
    'FederatedServer',
    'get_evaluate_fn',
]
