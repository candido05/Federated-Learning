"""
Cliente Federated Learning para XGBoost
"""

import xgboost as xgb
from flwr.client import Client
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Status
from flwr.common.logger import log
from logging import INFO, WARNING

from common import calculate_comprehensive_metrics, print_metrics_summary


class XGBoostClient(Client):
    """Cliente Federated Learning para XGBoost"""

    def __init__(self, train_dmatrix, valid_dmatrix, num_train, num_val,
                 num_local_round, params, train_method, cid: int,
                 X_valid=None, y_valid=None):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
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

        if global_round <= 1 or not ins.parameters.tensors:
            # Primeira rodada: treinar do zero
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
                verbose_eval=False,
            )
        else:
            # Carregar modelo global e continuar treinamento
            try:
                global_model = bytearray(ins.parameters.tensors[0])
                global_bst = xgb.Booster(params=self.params)
                global_bst.load_model(global_model)
                bst = xgb.train(
                    self.params,
                    self.train_dmatrix,
                    num_boost_round=self.num_local_round,
                    evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
                    xgb_model=global_bst,
                    verbose_eval=False,
                )
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Falha ao carregar modelo global: {e}; treinando do zero.")
                bst = xgb.train(
                    self.params,
                    self.train_dmatrix,
                    num_boost_round=self.num_local_round,
                    evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
                    verbose_eval=False,
                )

        # Calcular métricas avançadas
        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = bst.predict(self.valid_dmatrix)
                metrics = calculate_comprehensive_metrics(self.y_valid, y_pred_proba)
                print_metrics_summary(metrics, client_id=self.cid, round_num=global_round)
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Erro ao calcular métricas: {e}")

        # Salvar modelo como bytes
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

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

        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Calcular métricas avançadas
        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = bst.predict(self.valid_dmatrix)
                comprehensive_metrics = calculate_comprehensive_metrics(self.y_valid, y_pred_proba)

                return_metrics = {k: v for k, v in comprehensive_metrics.items() if k != 'confusion_matrix'}

                return EvaluateRes(
                    status=Status(code=Code.OK, message="OK"),
                    loss=float(comprehensive_metrics.get('auc', 0.5)),
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
