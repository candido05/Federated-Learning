"""
Serializacao e deserializacao de modelos para comunicacao Flower.
"""

import pickle
from typing import List, Optional

import numpy as np
from flwr.common import Parameters, parameters_to_ndarrays


def serialize_model(model) -> List[np.ndarray]:
    """
    Serializa modelo para formato compativel com Flower.
    Retorna lista de ndarrays (formato esperado pelo NumPyClient).
    """
    if model is None:
        return [np.array([], dtype=np.uint8)]

    model_bytes = pickle.dumps(model)
    model_array = np.frombuffer(model_bytes, dtype=np.uint8)
    return [model_array]


def deserialize_model_from_ndarrays(ndarrays: List[np.ndarray]) -> Optional[object]:
    """Deserializa modelo a partir de lista de ndarrays."""
    if ndarrays is None or len(ndarrays) == 0:
        return None
    if len(ndarrays[0]) == 0:
        return None

    try:
        model_bytes = ndarrays[0].tobytes()
        return pickle.loads(model_bytes)
    except Exception as e:
        print(f"    Erro deserializando modelo: {e}")
        return None


def deserialize_model_from_parameters(params: Parameters) -> Optional[object]:
    """
    Deserializa modelo a partir de objeto Parameters do Flower.
    Usa parameters_to_ndarrays para conversao correta.
    """
    try:
        ndarrays = parameters_to_ndarrays(params)
        return deserialize_model_from_ndarrays(ndarrays)
    except Exception as e:
        print(f"    Erro convertendo Parameters: {e}")
        return None
