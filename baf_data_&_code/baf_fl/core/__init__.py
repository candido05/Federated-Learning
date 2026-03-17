"""
Componentes centrais do treinamento federado: cliente, runner e serializacao.

Nota: runner nao e importado aqui para evitar import circular
(runner importa strategies, que importa core.serialization).
Use: from baf_fl.core.runner import run_federated_training
"""

from .client import TreeModelClient, ModelFactory
from .serialization import serialize_model, deserialize_model_from_ndarrays

__all__ = [
    'TreeModelClient',
    'ModelFactory',
    'serialize_model',
    'deserialize_model_from_ndarrays',
]
