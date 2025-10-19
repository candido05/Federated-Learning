"""Hiperparâmetros dos modelos XGBoost, CatBoost e LightGBM.

Este módulo define hiperparâmetros padrão para modelos baseados em árvores
usados em experimentos de aprendizado federado.
"""

from typing import Any, Dict, List, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Hiperparâmetros base do XGBoost (sem tree_method, pois é detectado dinamicamente)
_XGBOOST_BASE_PARAMS: Dict[str, Union[str, float, int, List[str]]] = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "error"],
    "eta": 0.1,
    "max_depth": 6,
    "seed": 42,
}

# Parâmetros XGBoost com detecção de GPU (para compatibilidade com código legado)
XGBOOST_PARAMS: Dict[str, Union[str, float, int, List[str]]] = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "error"],
    "eta": 0.1,
    "max_depth": 6,
    "tree_method": "hist",  # CPU por padrão
    "seed": 42,
}

# Hiperparâmetros do CatBoost
CATBOOST_PARAMS: Dict[str, Union[str, float, int]] = {
    "iterations": 20,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.1,
    "depth": 6,
    "random_seed": 42,
}

# Hiperparâmetros do LightGBM
LIGHTGBM_PARAMS: Dict[str, Union[str, float, int]] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "seed": 42,
}


def get_xgboost_params_with_gpu_detection() -> Dict[str, Any]:
    """Retorna parâmetros XGBoost com detecção automática de GPU.

    Detecta se há GPU disponível via torch.cuda.is_available() e configura
    automaticamente tree_method e predictor para usar GPU quando disponível.

    Returns:
        Dicionário com parâmetros XGBoost otimizados para o hardware disponível.

    Note:
        - GPU disponível: tree_method='gpu_hist', predictor='gpu_predictor'
        - Somente CPU: tree_method='hist'
    """
    # Detecta se há GPU disponível
    use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
    tree_method = "gpu_hist" if use_gpu else "hist"

    # Cria cópia dos parâmetros base
    params = _XGBOOST_BASE_PARAMS.copy()
    params["tree_method"] = tree_method

    # Adiciona predictor GPU se disponível
    if use_gpu:
        params["predictor"] = "gpu_predictor"

    return params


def get_model_params(model_name: str, enable_gpu_detection: bool = True) -> Dict[str, Any]:
    """Obtém hiperparâmetros para um modelo específico.

    Args:
        model_name: Nome do modelo ('xgboost', 'catboost' ou 'lightgbm').
        enable_gpu_detection: Se True, detecta GPU automaticamente para XGBoost.
                             Se False, usa parâmetros estáticos (padrão: True).

    Returns:
        Dicionário contendo os hiperparâmetros do modelo.

    Raises:
        ValueError: Se o nome do modelo não for reconhecido.

    Note:
        Para XGBoost com enable_gpu_detection=True, a função detecta automaticamente
        a disponibilidade de GPU e configura tree_method adequadamente.
    """
    model_name_lower = model_name.lower()

    if model_name_lower == "xgboost":
        if enable_gpu_detection:
            return get_xgboost_params_with_gpu_detection()
        else:
            return XGBOOST_PARAMS.copy()
    elif model_name_lower == "catboost":
        return CATBOOST_PARAMS.copy()
    elif model_name_lower == "lightgbm":
        return LIGHTGBM_PARAMS.copy()
    else:
        raise ValueError(
            f"Nome de modelo desconhecido: {model_name}. "
            f"Esperado um de: xgboost, catboost, lightgbm"
        )
