"""
Estrategia Bagging: todos os clientes treinam, predicoes agregadas por media.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import (
    Parameters, Scalar, FitRes, EvaluateRes, FitIns, EvaluateIns,
)
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from ..data.metrics import evaluate_model_standard, evaluate_model_fair
from ..core.serialization import (
    serialize_model, deserialize_model_from_parameters,
)
from flwr.common import ndarrays_to_parameters


class BaggingStrategy(Strategy):
    """Bagging: Todos os clientes treinam, predicoes agregadas por media."""

    def __init__(self, num_clients: int = 3, val_data: Optional[Dict] = None,
                 recorder=None):
        self.num_clients = num_clients
        self.client_models: Dict[int, object] = {}
        self.round_metrics: List[Dict] = []
        self.val_data = val_data or {}
        self.recorder = recorder
        self.best_model = None

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        print(f"\n{'='*60}")
        print(f"BAGGING - Round {server_round}")
        print(f"{'='*60}")

        if self.recorder:
            self.recorder.mark_round_start(server_round)

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )

        if self.best_model is not None:
            model_ndarrays = serialize_model(self.best_model)
            fit_params = ndarrays_to_parameters(model_ndarrays)
            config = {"server_round": server_round, "warm_start": True}
            print(f"  [Servidor] Enviando melhor modelo para {len(clients)} clientes (warm start)")
        else:
            fit_params = Parameters(tensors=[], tensor_type="")
            config = {"server_round": server_round, "warm_start": False}
            print(f"  [Servidor] Clientes iniciarao do zero")

        fit_ins = FitIns(parameters=fit_params, config=config)

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self, server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"  [Servidor] Agregando {len(results)} modelos...")

        self.client_models = {}
        round_total_bytes = 0

        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("client_id", 0)
            model = deserialize_model_from_parameters(fit_res.parameters)

            if model is not None:
                self.client_models[client_id] = model
                print(f"  [Servidor] Modelo do Cliente {client_id} OK")
            else:
                print(f"  [Servidor] Falha ao carregar modelo do Cliente {client_id}")

            if self.recorder:
                model_bytes = fit_res.metrics.get("model_size_bytes", 0)
                round_total_bytes += model_bytes
                self.recorder.record_client_round(
                    round_num=server_round,
                    client_id=int(client_id),
                    train_tpr=float(fit_res.metrics.get("train_tpr", 0.0)),
                    val_tpr=float(fit_res.metrics.get("val_tpr", 0.0)),
                    training_time_sec=float(fit_res.metrics.get("training_time_sec", 0.0)),
                    model_size_bytes=int(model_bytes),
                    num_train_samples=int(fit_res.num_examples),
                )

        self._round_total_bytes = round_total_bytes

        # Selecionar melhor modelo para warm start na proxima rodada
        best_metric = -1
        for cid, model in self.client_models.items():
            metric = 0
            for client_proxy, fit_res in results:
                if fit_res.metrics.get("client_id", -1) == cid:
                    metric = float(fit_res.metrics.get("val_tpr", fit_res.metrics.get("accuracy", 0)))
                    break
            if metric > best_metric:
                best_metric = metric
                self.best_model = model

        if self.best_model is not None:
            print(f"  [Servidor] Melhor modelo selecionado (metric={best_metric:.4f})")

        return None, {"num_models": len(self.client_models)}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return []

    def aggregate_evaluate(
        self, server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return None, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if not self.client_models:
            return None

        X_val = self.val_data['X']
        y_val = self.val_data['y']
        age_val = self.val_data['age']

        predictions = []
        for model in self.client_models.values():
            pred = model.predict_proba(X_val)[:, 1]
            predictions.append(pred)

        y_prob_ensemble = np.mean(predictions, axis=0)

        metrics_std = evaluate_model_standard(y_val, y_prob_ensemble, age_val)
        metrics_fair = evaluate_model_fair(y_val, y_prob_ensemble, age_val)

        print(f"\n  [Servidor] BAGGING Round {server_round}:")
        print(f"    Threshold Unico:     TPR={metrics_std['tpr_at_5fpr']:.4f}, Fairness={metrics_std['fairness_ratio']:.4f}")
        print(f"    Threshold por Grupo: TPR={metrics_fair['tpr_at_5fpr']:.4f}, Fairness={metrics_fair['fairness_ratio']:.4f}")

        self.round_metrics.append({
            'round': server_round,
            'tpr_standard': metrics_std['tpr_at_5fpr'],
            'fairness_standard': metrics_std['fairness_ratio'],
            'tpr_fair': metrics_fair['tpr_at_5fpr'],
            'fairness_fair': metrics_fair['fairness_ratio'],
        })

        if self.recorder:
            self.recorder.record_server_round(
                round_num=server_round,
                tpr_standard=metrics_std['tpr_at_5fpr'],
                fairness_standard=metrics_std['fairness_ratio'],
                tpr_fair=metrics_fair['tpr_at_5fpr'],
                fairness_fair=metrics_fair['fairness_ratio'],
                num_client_models=len(self.client_models),
                total_bytes_this_round=getattr(self, '_round_total_bytes', 0),
            )

        return 1.0 - metrics_std['tpr_at_5fpr'], {"tpr_at_5fpr": metrics_std['tpr_at_5fpr']}

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Predicao ensemble (media das probabilidades de todos os modelos)."""
        predictions = [m.predict_proba(X)[:, 1] for m in self.client_models.values()]
        return np.mean(predictions, axis=0)
