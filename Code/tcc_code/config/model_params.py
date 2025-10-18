"""Hiperparâmetros dos modelos XGBoost, CatBoost e LightGBM.

Este módulo define hiperparâmetros padrão para modelos baseados em árvores
usados em experimentos de aprendizado federado.
"""

from typing import Any, Dict, List, Union

# Hiperparâmetros do XGBoost
XGBOOST_PARAMS: Dict[str, Union[str, float, int, List[str]]] = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "error"],
    "eta": 0.1,
    "max_depth": 6,
    "tree_method": "hist",
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


def get_model_params(model_name: str) -> Dict[str, Any]:
    """Obtém hiperparâmetros para um modelo específico.

    Args:
        model_name: Nome do modelo ('xgboost', 'catboost' ou 'lightgbm').

    Returns:
        Dicionário contendo os hiperparâmetros do modelo.

    Raises:
        ValueError: Se o nome do modelo não for reconhecido.
    """
    model_name_lower = model_name.lower()

    if model_name_lower == "xgboost":
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
