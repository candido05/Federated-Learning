"""
Estrategia Cycling com Warm Start: treinamento sequencial com transferencia de conhecimento.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import (
    Parameters, Scalar, FitRes, EvaluateRes, FitIns, EvaluateIns,
    ndarrays_to_parameters,
)
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from ..data.metrics import evaluate_model_standard, evaluate_model_fair
from ..core.serialization import (
    serialize_model, deserialize_model_from_parameters,
)


class CyclingStrategy(Strategy):
    """Cycling com WARM START: Treinamento sequencial com transferencia de conhecimento."""

    def __init__(self, num_clients: int = 3, val_data: Optional[Dict] = None,
                 recorder=None, lr_decay: float = 0.85):
        self.num_clients = num_clients
        self.current_client_idx = 0
        self.current_model: Optional[object] = None
        self.round_metrics: List[Dict] = []
        self.val_data = val_data or {}
        self.recorder = recorder
        self.lr_decay = lr_decay

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        print(f"\n{'='*60}")
        print(f"CYCLING - Round {server_round} - Cliente {self.current_client_idx}")
        print(f"{'='*60}")

        if self.recorder:
            self.recorder.mark_round_start(server_round)

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )
        # Ordenar por cid para garantir mapeamento correto
        clients = sorted(clients, key=lambda c: int(c.cid))

        selected_client = clients[self.current_client_idx]

        # Decay do learning rate: lr_factor = decay^(round-1)
        lr_factor = self.lr_decay ** (server_round - 1)

        if self.current_model is not None:
            model_ndarrays = serialize_model(self.current_model)
            fit_params = ndarrays_to_parameters(model_ndarrays)
            config = {"server_round": server_round, "warm_start": True, "lr_factor": lr_factor}
            print(f"  [Servidor] Enviando modelo para Cliente {self.current_client_idx} (warm start, lr_factor={lr_factor:.3f})")
        else:
            fit_params = Parameters(tensors=[], tensor_type="")
            config = {"server_round": server_round, "warm_start": False, "lr_factor": lr_factor}
            print(f"  [Servidor] Cliente {self.current_client_idx} iniciara do zero (lr_factor={lr_factor:.3f})")

        fit_ins = FitIns(parameters=fit_params, config=config)

        return [(selected_client, fit_ins)]

    def aggregate_fit(
        self, server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            print(f"  [Servidor] AVISO: Nenhum resultado recebido!")
            return None, {}

        client_proxy, fit_res = results[0]
        client_id = fit_res.metrics.get("client_id", self.current_client_idx)

        model = deserialize_model_from_parameters(fit_res.parameters)

        if model is not None:
            self.current_model = model
            print(f"  [Servidor] Modelo do Cliente {client_id} salvo para proximo round")
        else:
            print(f"  [Servidor] ERRO: Falha ao deserializar modelo do Cliente {client_id}")

        if self.recorder:
            model_bytes = fit_res.metrics.get("model_size_bytes", 0)
            self._round_total_bytes = model_bytes
            self.recorder.record_client_round(
                round_num=server_round,
                client_id=int(client_id),
                train_tpr=float(fit_res.metrics.get("train_tpr", 0.0)),
                val_tpr=float(fit_res.metrics.get("val_tpr", 0.0)),
                training_time_sec=float(fit_res.metrics.get("training_time_sec", 0.0)),
                model_size_bytes=int(model_bytes),
                num_train_samples=int(fit_res.num_examples),
            )

        prev_client = self.current_client_idx
        self.current_client_idx = (self.current_client_idx + 1) % self.num_clients

        return None, {"trained_client": prev_client}

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
        if self.current_model is None:
            return None

        X_val = self.val_data['X']
        y_val = self.val_data['y']
        age_val = self.val_data['age']

        y_prob = self.current_model.predict_proba(X_val)[:, 1]

        metrics_std = evaluate_model_standard(y_val, y_prob, age_val)
        metrics_fair = evaluate_model_fair(y_val, y_prob, age_val)

        trained_client = (self.current_client_idx - 1) % self.num_clients

        print(f"\n  [Servidor] CYCLING Round {server_round} (Cliente {trained_client}):")
        print(f"    Threshold Unico:     TPR={metrics_std['tpr_at_5fpr']:.4f}, Fairness={metrics_std['fairness_ratio']:.4f}")
        print(f"    Threshold por Grupo: TPR={metrics_fair['tpr_at_5fpr']:.4f}, Fairness={metrics_fair['fairness_ratio']:.4f}")

        self.round_metrics.append({
            'round': server_round,
            'trained_client': trained_client,
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
                trained_client=trained_client,
                total_bytes_this_round=getattr(self, '_round_total_bytes', 0),
            )

        return 1.0 - metrics_std['tpr_at_5fpr'], {
            "tpr_at_5fpr": metrics_std['tpr_at_5fpr'],
            "trained_client": trained_client,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicao usando o modelo atual (ultimo cliente treinado)."""
        if self.current_model is None:
            raise ValueError("Nenhum modelo disponivel")
        return self.current_model.predict_proba(X)[:, 1]
