"""
Cliente Flower para modelos de arvore com suporte a Warm Start.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import flwr as fl

from ..config import RANDOM_STATE
from ..data.metrics import calc_tpr_at_fpr, evaluate_model_standard
from .serialization import serialize_model, deserialize_model_from_ndarrays


class ModelFactory:
    """Factory para criar e treinar modelos com suporte a Warm Start."""

    @staticmethod
    def create_and_train(
        model_type: str,
        params: Dict,
        scale_pos_weight: float,
        X_train: np.ndarray,
        y_train: np.ndarray,
        warm_start_model=None,
        client_id: int = 0,
    ):
        """Cria e treina modelo do tipo especificado."""
        if model_type == 'xgboost':
            return ModelFactory._train_xgboost(
                params, scale_pos_weight, X_train, y_train, warm_start_model, client_id,
            )
        elif model_type == 'lightgbm':
            return ModelFactory._train_lightgbm(
                params, scale_pos_weight, X_train, y_train, warm_start_model, client_id,
            )
        elif model_type == 'catboost':
            return ModelFactory._train_catboost(
                params, X_train, y_train, warm_start_model, client_id,
            )
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

    @staticmethod
    def _train_xgboost(params, scale_pos_weight, X_train, y_train, warm_start_model, client_id):
        full_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': RANDOM_STATE,
            'scale_pos_weight': scale_pos_weight,
            **params,
        }
        model = xgb.XGBClassifier(**full_params)

        if warm_start_model is not None:
            print(f"    [Cliente {client_id}] WARM START XGBoost")
            model.fit(X_train, y_train, xgb_model=warm_start_model.get_booster(), verbose=False)
        else:
            print(f"    [Cliente {client_id}] XGBoost - do zero")
            model.fit(X_train, y_train, verbose=False)

        return model

    @staticmethod
    def _train_lightgbm(params, scale_pos_weight, X_train, y_train, warm_start_model, client_id):
        full_params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'random_state': RANDOM_STATE,
            'scale_pos_weight': scale_pos_weight,
            **params,
        }
        model = lgb.LGBMClassifier(**full_params)

        if warm_start_model is not None:
            print(f"    [Cliente {client_id}] WARM START LightGBM")
            model.fit(X_train, y_train, init_model=warm_start_model)
        else:
            print(f"    [Cliente {client_id}] LightGBM - do zero")
            model.fit(X_train, y_train)

        return model

    @staticmethod
    def _train_catboost(params, X_train, y_train, warm_start_model, client_id):
        full_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': RANDOM_STATE,
            'auto_class_weights': 'Balanced',
            'verbose': False,
            **params,
        }
        model = CatBoostClassifier(**full_params)

        if warm_start_model is not None:
            print(f"    [Cliente {client_id}] WARM START CatBoost")
            model.fit(X_train, y_train, init_model=warm_start_model, verbose=False)
        else:
            print(f"    [Cliente {client_id}] CatBoost - do zero")
            model.fit(X_train, y_train, verbose=False)

        return model


class TreeModelClient(fl.client.NumPyClient):
    """Cliente Flower para modelos de arvore com suporte a WARM START."""

    def __init__(
        self,
        client_id: int,
        train_data: Dict,
        val_data: Dict,
        model_type: str,
        model_params: Dict,
        scale_pos_weight: float,
    ):
        self.client_id = client_id
        self.model = None

        self.X_train = train_data['X']
        self.y_train = train_data['y']
        self.age_train = train_data['age']

        self.X_val = val_data['X']
        self.y_val = val_data['y']
        self.age_val = val_data['age']

        self.model_type = model_type
        self.model_params = model_params.get(model_type, {})
        self.scale_pos_weight = scale_pos_weight

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Retorna modelo serializado."""
        return serialize_model(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Treina modelo localmente com suporte a WARM START."""
        server_round = config.get("server_round", 0)
        use_warm_start = config.get("warm_start", False)
        lr_factor = config.get("lr_factor", 1.0)

        print(f"\n    [Cliente {self.client_id}] Round {server_round} | warm_start={use_warm_start} | lr_factor={lr_factor:.3f}")

        warm_start_model = None
        if use_warm_start:
            warm_start_model = deserialize_model_from_ndarrays(parameters)
            if warm_start_model is not None:
                print(f"    [Cliente {self.client_id}] Modelo recebido para warm start!")
            else:
                print(f"    [Cliente {self.client_id}] AVISO: Falha ao receber modelo, treinando do zero")

        # Aplicar decay no learning rate se fornecido pela strategy
        train_params = dict(self.model_params)
        if lr_factor < 1.0:
            lr_key = 'learning_rate' if self.model_type != 'catboost' else 'learning_rate'
            if lr_key in train_params:
                original_lr = train_params[lr_key]
                train_params[lr_key] = original_lr * lr_factor
                print(f"    [Cliente {self.client_id}] LR decay: {original_lr:.4f} -> {train_params[lr_key]:.4f}")

        t_start = time.time()
        self.model = ModelFactory.create_and_train(
            model_type=self.model_type,
            params=train_params,
            scale_pos_weight=self.scale_pos_weight,
            X_train=self.X_train,
            y_train=self.y_train,
            warm_start_model=warm_start_model,
            client_id=self.client_id,
        )
        training_time = time.time() - t_start

        y_prob_train = self.model.predict_proba(self.X_train)[:, 1]
        train_tpr = calc_tpr_at_fpr(self.y_train, y_prob_train)

        y_prob_val = self.model.predict_proba(self.X_val)[:, 1]
        val_tpr = calc_tpr_at_fpr(self.y_val, y_prob_val)

        print(f"    [Cliente {self.client_id}] Treino TPR={train_tpr:.4f} | Val TPR={val_tpr:.4f} | Tempo={training_time:.1f}s")

        serialized = serialize_model(self.model)
        model_size_bytes = sum(arr.nbytes for arr in serialized)

        return serialized, len(self.X_train), {
            "client_id": self.client_id,
            "train_tpr": float(train_tpr),
            "val_tpr": float(val_tpr),
            "training_time_sec": float(training_time),
            "model_size_bytes": int(model_size_bytes),
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        if self.model is None:
            return 1.0, 0, {}

        y_prob = self.model.predict_proba(self.X_val)[:, 1]
        metrics = evaluate_model_standard(self.y_val, y_prob, self.age_val)

        return 1.0 - metrics['tpr_at_5fpr'], len(self.X_val), {
            "tpr_at_5fpr": metrics['tpr_at_5fpr'],
            "roc_auc": metrics['roc_auc'],
        }


