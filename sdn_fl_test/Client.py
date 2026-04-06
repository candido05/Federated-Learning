"""
Cliente Flower para Federated Learning com XGBoost.
Adaptado para deployment distribuído (WSL / SDN / GNS3).

Uso:
    python client.py --server 10.0.0.10:8080 --partition-id 0 --num-partitions 2
"""

import argparse
import warnings

import numpy as np
import xgboost as xgb

import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

from Task import load_data

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────
# XGBoost params padrão
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


def _local_boost(bst_input, num_local_round, train_dmatrix, train_method):
    """Treinamento local: atualiza árvores com dados locais."""
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    # Bagging: extrai as últimas N árvores para agregação
    # Cyclic: retorna o modelo inteiro
    bst = (
        bst_input[
            bst_input.num_boosted_rounds()
            - num_local_round : bst_input.num_boosted_rounds()
        ]
        if train_method == "bagging"
        else bst_input
    )

    return bst


class XgbClient(fl.client.Client):
    """Cliente Flower para XGBoost federado."""

    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        train_method,
        params,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.train_method = train_method
        self.params = params

    def fit(self, ins: FitIns) -> FitRes:
        """Treinamento local."""
        global_model_bytes = parameters_to_ndarrays(ins.parameters)[0]

        if len(global_model_bytes) == 0:
            # Primeiro round: treina do zero
            print("[CLIENT] Primeiro round - treinamento inicial")
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
            )
        else:
            # Rounds subsequentes: carrega modelo global e treina localmente
            print("[CLIENT] Carregando modelo global e treinando localmente...")
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(global_model_bytes.tobytes())
            bst.load_model(global_model)
            bst = _local_boost(
                bst,
                self.num_local_round,
                self.train_dmatrix,
                self.train_method,
            )

        # Serializa modelo local
        local_model = bst.save_raw("json")
        model_np = np.frombuffer(local_model, dtype=np.uint8)
        parameters = ndarrays_to_parameters([model_np])

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=parameters,
            num_examples=self.num_train,
            metrics={"num-examples": self.num_train},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Avaliação local."""
        global_model_bytes = parameters_to_ndarrays(ins.parameters)[0]

        if len(global_model_bytes) == 0:
            return EvaluateRes(
                status=Status(code=Code.OK, message="Skip"),
                loss=0.0,
                num_examples=self.num_val,
                metrics={"auc": 0.0},
            )

        bst = xgb.Booster(params=self.params)
        global_model = bytearray(global_model_bytes.tobytes())
        bst.load_model(global_model)

        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = float(eval_results.split("\t")[1].split(":")[1])
        print(f"[CLIENT] Avaliação local - AUC: {auc:.4f}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=1.0 - auc,
            num_examples=self.num_val,
            metrics={"auc": auc, "num-examples": self.num_val},
        )


def main():
    parser = argparse.ArgumentParser(description="Flower FL Client - XGBoost")
    parser.add_argument(
        "--server",
        type=str,
        default="127.0.0.1:8080",
        help="Endereço do servidor Flower (padrão: 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        required=True,
        help="ID da partição deste cliente (0, 1, 2, ...)",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=2,
        help="Número total de partições/clientes (padrão: 2)",
    )
    parser.add_argument(
        "--num-local-rounds",
        type=int,
        default=1,
        help="Rounds locais por round global (padrão: 1)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cyclic",
        choices=["bagging", "cyclic"],
        help="Método de treinamento (padrão: cyclic)",
    )
    parser.add_argument(
        "--partitioner-type",
        type=str,
        default="uniform",
        choices=["uniform", "linear", "square", "exponential"],
        help="Tipo de particionamento dos dados (padrão: uniform)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fração de teste local (padrão: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade (padrão: 42)",
    )
    parser.add_argument(
        "--centralised-eval-client",
        action="store_true",
        default=False,
        help="Usar dataset centralizado para avaliação no cliente",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FLOWER FL CLIENT - XGBoost Federado")
    print("=" * 60)
    print(f"  Servidor:       {args.server}")
    print(f"  Partition ID:   {args.partition_id}")
    print(f"  Num partitions: {args.num_partitions}")
    print(f"  Local rounds:   {args.num_local_rounds}")
    print(f"  Método:         {args.method}")
    print(f"  Partitioner:    {args.partitioner_type}")
    print("=" * 60)

    # Carrega dados
    print("\n[INFO] Carregando dados da partição...")
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
        partitioner_type=args.partitioner_type,
        partition_id=args.partition_id,
        num_partitions=args.num_partitions,
        centralised_eval_client=args.centralised_eval_client,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"[INFO] Dados carregados: {num_train} treino, {num_val} validação\n")

    # Cria cliente
    client = XgbClient(
        train_dmatrix=train_dmatrix,
        valid_dmatrix=valid_dmatrix,
        num_train=num_train,
        num_val=num_val,
        num_local_round=args.num_local_rounds,
        train_method=args.method,
        params=XGBOOST_PARAMS,
    )

    # Conecta ao servidor
    fl.client.start_client(
        server_address=args.server,
        client=client,
    )

    print("\n[INFO] Cliente finalizado!")


if __name__ == "__main__":
    main()