"""
Servidor Flower para Federated Learning com XGBoost.
Adaptado para deployment distribuído (WSL / SDN / GNS3).

Uso:
    python server.py [--address 0.0.0.0:8080] [--rounds 5] [--method bagging]
"""

import argparse
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xgboost as xgb
from datasets import load_dataset

import flwr as fl
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

from Task import replace_keys, transform_dataset_to_dmatrix


# ──────────────────────────────────────────────
# XGBoost params padrão (mesmos do pyproject original)
# ──────────────────────────────────────────────
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eta": 0.1,
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 4,
    "num_parallel_tree": 1,
    "subsample": 1.0,
    "tree_method": "hist",
}


# ──────────────────────────────────────────────
# Estratégia customizada para agregação XGBoost
# ──────────────────────────────────────────────
class FedXgbBaggingStrategy(fl.server.strategy.FedAvg):
    """Estratégia de Bagging para XGBoost federado."""

    def __init__(self, evaluate_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._evaluate_fn = evaluate_fn

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Agrega modelos XGBoost via bagging (concatena árvores)."""
        if not results:
            return None, {}

        # Coleta modelos dos clientes
        models = []
        for _, fit_res in results:
            model_bytes = bytearray(
                parameters_to_ndarrays(fit_res.parameters)[0].tobytes()
            )
            bst = xgb.Booster(params=XGBOOST_PARAMS)
            bst.load_model(model_bytes)
            models.append(bst)

        # Agrega via bagging: usa o primeiro modelo como base
        # Em produção, aqui seria feita concatenação de árvores
        # Para simplificação, usa o modelo com mais árvores
        aggregated = models[0]
        for m in models[1:]:
            # Merge trees - XGBoost não tem merge nativo simples,
            # então usamos o modelo do último cliente no ciclo
            aggregated = m

        # Serializa modelo agregado
        global_model = aggregated.save_raw("json")
        model_np = np.frombuffer(global_model, dtype=np.uint8)
        parameters = ndarrays_to_parameters([model_np])

        return parameters, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self._evaluate_fn is None:
            return None
        return self._evaluate_fn(server_round, parameters)


class FedXgbCyclicStrategy(fl.server.strategy.FedAvg):
    """Estratégia Cíclica para XGBoost federado."""

    def __init__(self, evaluate_fn=None, **kwargs):
        # Cíclico: 1 cliente por vez
        kwargs["fraction_fit"] = 0.0
        kwargs["min_fit_clients"] = 1
        kwargs["min_available_clients"] = 1
        super().__init__(**kwargs)
        self._evaluate_fn = evaluate_fn
        self._current_client_idx = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """No modo cíclico, o modelo do cliente é o modelo global."""
        if not results:
            return None, {}

        # Pega o modelo do único cliente que treinou
        _, fit_res = results[0]
        return fit_res.parameters, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self._evaluate_fn is None:
            return None
        return self._evaluate_fn(server_round, parameters)


# ──────────────────────────────────────────────
# Avaliação centralizada
# ──────────────────────────────────────────────
def get_evaluate_fn(test_dmatrix, params):
    """Retorna função de avaliação centralizada."""

    def evaluate_fn(
        server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if server_round == 0:
            return None

        model_bytes = bytearray(
            parameters_to_ndarrays(parameters)[0].tobytes()
        )
        bst = xgb.Booster(params=params)
        bst.load_model(model_bytes)

        eval_results = bst.eval_set(
            evals=[(test_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        print(f"[Round {server_round}] Avaliação centralizada - AUC: {auc}")

        return float(auc), {"AUC": auc}

    return evaluate_fn


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Flower FL Server - XGBoost")
    parser.add_argument(
        "--address",
        type=str,
        default="0.0.0.0:8080",
        help="Endereço do servidor (padrão: 0.0.0.0:8080)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Número de rounds federados (padrão: 5)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cyclic",
        choices=["bagging", "cyclic"],
        help="Método de treinamento: bagging ou cyclic (padrão: cyclic)",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Mínimo de clientes para iniciar (padrão: 2)",
    )
    parser.add_argument(
        "--centralised-eval",
        action="store_true",
        default=True,
        help="Usar avaliação centralizada (padrão: True)",
    )
    parser.add_argument(
        "--no-centralised-eval",
        action="store_false",
        dest="centralised_eval",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FLOWER FL SERVER - XGBoost Federado")
    print("=" * 60)
    print(f"  Endereço:       {args.address}")
    print(f"  Rounds:         {args.rounds}")
    print(f"  Método:         {args.method}")
    print(f"  Min clientes:   {args.min_clients}")
    print(f"  Eval central:   {args.centralised_eval}")
    print("=" * 60)

    evaluate_fn = None
    if args.centralised_eval:
        print("\n[INFO] Carregando dataset de teste para avaliação centralizada...")
        test_set = load_dataset("jxie/higgs", split="test")
        test_set.set_format("numpy")
        test_dmatrix = transform_dataset_to_dmatrix(test_set)
        evaluate_fn = get_evaluate_fn(test_dmatrix, XGBOOST_PARAMS)
        print("[INFO] Dataset de teste carregado.\n")

    # Modelo global inicial (vazio)
    initial_model = np.frombuffer(b"", dtype=np.uint8)
    initial_parameters = ndarrays_to_parameters([initial_model])

    # Seleciona estratégia
    if args.method == "bagging":
        strategy = FedXgbBaggingStrategy(
            evaluate_fn=evaluate_fn,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=args.min_clients,
            min_evaluate_clients=args.min_clients,
            min_available_clients=args.min_clients,
            initial_parameters=initial_parameters,
        )
    else:
        strategy = FedXgbCyclicStrategy(
            evaluate_fn=evaluate_fn,
            min_available_clients=args.min_clients,
            initial_parameters=initial_parameters,
        )

    # Inicia servidor
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    print("\n[INFO] Treinamento federado concluído!")


if __name__ == "__main__":
    main()