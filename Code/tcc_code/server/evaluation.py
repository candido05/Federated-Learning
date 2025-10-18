"""
Funções de avaliação centralizada para Federated Learning.

Fornece funções de avaliação compatíveis com Flower para diferentes frameworks.
"""

import numpy as np
import tempfile
import os
from typing import Callable, Dict, Tuple, Optional
from datetime import datetime

from flwr.common import NDArrays, Scalar

from utils.metrics import MetricsCalculator


def get_evaluate_fn(
    test_data: Tuple[np.ndarray, np.ndarray],
    model_type: str,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """
    Cria função de avaliação centralizada compatível com Flower.

    Args:
        test_data: Tupla (X_test, y_test) para avaliação.
        model_type: Tipo do modelo ('xgboost', 'catboost', 'lightgbm').

    Returns:
        Função evaluate_fn compatível com estratégias Flower.

    Raises:
        ValueError: Se model_type não for suportado.
    """
    X_test, y_test = test_data

    # Valida tipo de modelo
    supported_models = ['xgboost', 'catboost', 'lightgbm']
    if model_type.lower() not in supported_models:
        raise ValueError(
            f"Tipo de modelo '{model_type}' não suportado. "
            f"Use um de: {supported_models}"
        )

    # Cria calculator de métricas
    metrics_calculator = MetricsCalculator()

    def evaluate_fn(
        server_round: int,
        parameters_ndarrays: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Avalia modelo global no servidor usando dataset de teste.

        Args:
            server_round: Número do round atual.
            parameters_ndarrays: Parâmetros do modelo global como lista de arrays.
            config: Configuração adicional (não usado).

        Returns:
            Tupla (loss, metrics) ou None se avaliação falhar.
        """
        try:
            # Converte NDArrays para bytes
            model_bytes = parameters_ndarrays[0].tobytes()

            # Carrega modelo baseado no tipo
            if model_type.lower() == 'xgboost':
                model = _load_xgboost_model(model_bytes)
                predictions = _predict_xgboost(model, X_test)

            elif model_type.lower() == 'catboost':
                model = _load_catboost_model(model_bytes)
                predictions = _predict_catboost(model, X_test)

            elif model_type.lower() == 'lightgbm':
                model = _load_lightgbm_model(model_bytes)
                predictions = _predict_lightgbm(model, X_test)

            # Calcula métricas comprehensivas
            metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_true=y_test,
                y_pred_proba=predictions
            )

            # Loss é 1 - accuracy (quanto menor, melhor)
            loss = 1.0 - metrics['accuracy']

            # Prepara métricas para retorno
            metrics_dict = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
            }

            return loss, metrics_dict

        except Exception as e:
            # Log erro mas não interrompe treinamento
            print(f"[ERRO] Avaliação centralizada falhou no round {server_round}: {e}")
            return None

    return evaluate_fn


def _load_xgboost_model(model_bytes: bytes):
    """
    Carrega modelo XGBoost a partir de bytes.

    Args:
        model_bytes: Bytes do modelo serializado.

    Returns:
        xgb.Booster carregado.
    """
    import xgboost as xgb

    temp_path = None
    try:
        # Cria arquivo temporário
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"eval_xgb_{timestamp}.json")

        # Escreve bytes
        with open(temp_path, 'wb') as f:
            f.write(model_bytes)

        # Carrega modelo
        bst = xgb.Booster()
        bst.load_model(temp_path)

        return bst

    finally:
        # Limpa arquivo temporário
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _predict_xgboost(model, X: np.ndarray) -> np.ndarray:
    """
    Realiza predição com modelo XGBoost.

    Args:
        model: xgb.Booster.
        X: Features para predição.

    Returns:
        Probabilidades preditas [n_samples, n_classes].
    """
    import xgboost as xgb

    dtest = xgb.DMatrix(X)
    predictions = model.predict(dtest)

    # Converte para formato [n_samples, n_classes]
    if len(predictions.shape) == 1:
        prob_class_1 = predictions.reshape(-1, 1)
        prob_class_0 = 1 - prob_class_1
        predictions = np.hstack([prob_class_0, prob_class_1])

    return predictions


def _load_catboost_model(model_bytes: bytes):
    """
    Carrega modelo CatBoost a partir de bytes.

    Args:
        model_bytes: Bytes do modelo serializado.

    Returns:
        CatBoost carregado.
    """
    from catboost import CatBoost

    temp_path = None
    try:
        # Cria arquivo temporário
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"eval_cb_{timestamp}.cbm")

        # Escreve bytes
        with open(temp_path, 'wb') as f:
            f.write(model_bytes)

        # Carrega modelo
        model = CatBoost()
        model.load_model(temp_path, format="cbm")

        return model

    finally:
        # Limpa arquivo temporário
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _predict_catboost(model, X: np.ndarray) -> np.ndarray:
    """
    Realiza predição com modelo CatBoost.

    Args:
        model: CatBoost.
        X: Features para predição.

    Returns:
        Probabilidades preditas [n_samples, n_classes].
    """
    predictions = model.predict(X, prediction_type='Probability')
    return predictions


def _load_lightgbm_model(model_bytes: bytes):
    """
    Carrega modelo LightGBM a partir de bytes.

    Args:
        model_bytes: Bytes do modelo serializado.

    Returns:
        lgb.Booster carregado.
    """
    import lightgbm as lgb

    temp_path = None
    try:
        # Cria arquivo temporário
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"eval_lgb_{timestamp}.txt")

        # Escreve bytes
        with open(temp_path, 'wb') as f:
            f.write(model_bytes)

        # Carrega modelo
        bst = lgb.Booster(model_file=temp_path)

        return bst

    finally:
        # Limpa arquivo temporário
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _predict_lightgbm(model, X: np.ndarray) -> np.ndarray:
    """
    Realiza predição com modelo LightGBM.

    Args:
        model: lgb.Booster.
        X: Features para predição.

    Returns:
        Probabilidades preditas [n_samples, n_classes].
    """
    predictions = model.predict(X)

    # Converte para formato [n_samples, n_classes]
    if len(predictions.shape) == 1:
        prob_class_1 = predictions.reshape(-1, 1)
        prob_class_0 = 1 - prob_class_1
        predictions = np.hstack([prob_class_0, prob_class_1])

    return predictions
