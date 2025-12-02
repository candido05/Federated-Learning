"""
Cliente Federated Learning para CatBoost
"""

import os
from catboost import CatBoost
from flwr.client import Client
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Status
from flwr.common.logger import log
from logging import INFO, WARNING

from common import calculate_comprehensive_metrics, print_metrics_summary


class CatBoostClient(Client):
    """Cliente Federated Learning para CatBoost"""

    def __init__(self, train_pool, valid_pool, num_train, num_val,
                 num_local_round, params, train_method, cid: int,
                 X_valid=None, y_valid=None):
        self.train_pool = train_pool
        self.valid_pool = valid_pool
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.train_method = train_method
        self.cid = cid
        self.X_valid = X_valid
        self.y_valid = y_valid

    def fit(self, ins: FitIns) -> FitRes:
        log(INFO, f"[Client {self.cid}] fit, config: {ins.config}")
        global_round = int(ins.config.get("global_round", "0"))

        print(f"\n{'─'*80}")
        print(f"[Cliente {self.cid}] INICIANDO TREINAMENTO - Round {global_round}")
        print(f"  Amostras treino: {self.num_train} | Amostras validação: {self.num_val}")
        print(f"  Épocas locais: {self.num_local_round}")
        print(f"{'─'*80}")

        if global_round <= 1 or not ins.parameters.tensors:
            model = CatBoost(self.params)
            model.fit(self.train_pool, eval_set=self.valid_pool, verbose=10)
        else:
            try:
                global_model_bytes = bytearray(ins.parameters.tensors[0])
                temp_model_path = f"/tmp/catboost_global_model_{self.cid}_{global_round}.cbm"
                with open(temp_model_path, 'wb') as f:
                    f.write(global_model_bytes)

                model = CatBoost(self.params)
                model.load_model(temp_model_path)
                model.fit(self.train_pool, eval_set=self.valid_pool, verbose=10, init_model=model)

                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)

            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Falha ao carregar modelo global: {e}; treinando do zero.")
                model = CatBoost(self.params)
                model.fit(self.train_pool, eval_set=self.valid_pool, verbose=10)

        print(f"{'─'*80}")
        print(f"[Cliente {self.cid}] TREINAMENTO CONCLUÍDO - Round {global_round}")
        print(f"{'─'*80}\n")

        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = model.predict(self.valid_pool, prediction_type='Probability')
                if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                metrics = calculate_comprehensive_metrics(self.y_valid, y_pred_proba)
                print_metrics_summary(metrics, client_id=self.cid, round_num=global_round)
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Erro ao calcular métricas: {e}")

        temp_save_path = f"/tmp/catboost_local_model_{self.cid}_{global_round}.cbm"
        model.save_model(temp_save_path, format='cbm')

        with open(temp_save_path, 'rb') as f:
            local_model_bytes = f.read()

        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        log(INFO, f"[Client {self.cid}] evaluate, config: {ins.config}")
        if not ins.parameters.tensors:
            return EvaluateRes(
                status=Status(code=Code.OK, message="No model"),
                loss=0.0,
                num_examples=self.num_val,
                metrics={"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "auc": 0.5}
            )

        temp_model_path = f"/tmp/catboost_eval_model_{self.cid}.cbm"
        model_bytes = bytearray(ins.parameters.tensors[0])

        with open(temp_model_path, 'wb') as f:
            f.write(model_bytes)

        model = CatBoost(self.params)
        model.load_model(temp_model_path)

        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = model.predict(self.valid_pool, prediction_type='Probability')
                if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                comprehensive_metrics = calculate_comprehensive_metrics(self.y_valid, y_pred_proba)

                return_metrics = {k: v for k, v in comprehensive_metrics.items() if k != 'confusion_matrix'}

                return EvaluateRes(
                    status=Status(code=Code.OK, message="OK"),
                    loss=float(1.0 - comprehensive_metrics.get('accuracy', 0.5)),
                    num_examples=self.num_val,
                    metrics=return_metrics,
                )
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Erro ao calcular métricas: {e}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=0.693147,
            num_examples=self.num_val,
            metrics={"accuracy": 0.5, "auc": 0.5, "precision": 0.5, "recall": 0.5, "f1_score": 0.5},
        )
