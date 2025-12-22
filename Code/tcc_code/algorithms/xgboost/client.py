"""
Cliente Federated Learning para XGBoost
"""

import numpy as np
import xgboost as xgb
from xgboost.callback import TrainingCallback
from flwr.client import Client
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Status
from flwr.common.logger import log
from logging import INFO, WARNING

from common import (
    calculate_comprehensive_metrics, print_metrics_summary,
    ClassBalancingHelper, CurriculumLearning
)


class VerboseCallback(TrainingCallback):
    """Callback para mostrar progresso detalhado do treinamento no terminal"""

    def __init__(self, client_id: int, global_round: int):
        self.client_id = client_id
        self.global_round = global_round

    def after_iteration(self, model, epoch: int, evals_log: dict):
        if evals_log:
            train_metrics = evals_log.get('train', {})
            val_metrics = evals_log.get('validate', {})

            train_metric = list(train_metrics.values())[0][-1] if train_metrics else 0.0
            val_metric = list(val_metrics.values())[0][-1] if val_metrics else 0.0

            print(f"  [Cliente {self.client_id}] Round {self.global_round} | Época {epoch+1:2d} | "
                  f"Train: {train_metric:.4f} | Valid: {val_metric:.4f}")

        return False


class XGBoostClient(Client):
    """Cliente Federated Learning para XGBoost"""

    def __init__(self, train_dmatrix, valid_dmatrix, num_train, num_val,
                 num_local_round, params, train_method, cid: int,
                 X_valid=None, y_valid=None, train_y=None,
                 advanced_config=None):
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
        self.train_y = train_y
        self.advanced_config = advanced_config or {}

        self.class_balancer = None
        self.curriculum = None
        if self.advanced_config.get('use_class_weights') or self.advanced_config.get('use_sample_weights'):
            num_classes = len(np.unique(train_y)) if train_y is not None else 3
            max_weight = self.advanced_config.get('max_class_weight', 10.0)
            self.class_balancer = ClassBalancingHelper(num_classes=num_classes, max_weight=max_weight)

        if self.advanced_config.get('use_curriculum'):
            num_rounds = self.advanced_config.get('num_server_rounds', 50)
            warmup = self.advanced_config.get('curriculum_warmup', 5)
            self.curriculum = CurriculumLearning(num_rounds=num_rounds, warmup_rounds=warmup)

    def fit(self, ins: FitIns) -> FitRes:
        log(INFO, f"[Client {self.cid}] fit, config: {ins.config}")
        global_round = int(ins.config.get("global_round", "0"))

        print(f"\n{'─'*80}")
        print(f"[Cliente {self.cid}] INICIANDO TREINAMENTO - Round {global_round}")
        print(f"  Amostras treino: {self.num_train} | Amostras validação: {self.num_val}")
        print(f"  Épocas locais: {self.num_local_round}")

        sample_weights = None
        if self.class_balancer and self.train_y is not None:
            if self.advanced_config.get('use_class_weights'):
                class_weights = self.class_balancer.compute_class_weights(self.train_y)

                if self.curriculum:
                    class_weights = self.curriculum.adjust_class_weights_by_round(
                        class_weights, global_round
                    )
                    multiplier = self.curriculum.get_round_weights_multiplier(global_round)
                    print(f"  [CURRICULUM] Multiplicador de pesos: {multiplier:.2f}x")

                sample_weights = np.array([class_weights.get(label, 1.0) for label in self.train_y])
                print(f"  [CLASS WEIGHTS] Aplicados: {class_weights}")

            elif self.advanced_config.get('use_sample_weights'):
                sample_weights = self.class_balancer.compute_sample_weights(self.train_y)
                print(f"  [SAMPLE WEIGHTS] Aplicados: min={sample_weights.min():.2f}, "
                      f"max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")

        if sample_weights is not None:
            self.train_dmatrix.set_weight(sample_weights)

        print(f"{'─'*80}")

        verbose_callback = VerboseCallback(self.cid, global_round)

        # FASE 3: Early Stopping
        early_stopping_rounds = None
        if self.advanced_config.get('use_early_stopping', False):
            early_stopping_rounds = self.advanced_config.get('early_stopping_rounds', 5)
            print(f"  [EARLY STOPPING] Ativo com {early_stopping_rounds} rounds de paciência")


        if global_round <= 1 or not ins.parameters.tensors:
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
                callbacks=[verbose_callback],
            )
        else:
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
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False,
                    callbacks=[verbose_callback],
                )
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Falha ao carregar modelo global: {e}; treinando do zero.")
                bst = xgb.train(
                    self.params,
                    self.train_dmatrix,
                    num_boost_round=self.num_local_round,
                    evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False,
                    callbacks=[verbose_callback],
                )

        # FASE 3: Log early stopping info
        actual_iterations = bst.best_iteration + 1 if hasattr(bst, 'best_iteration') and bst.best_iteration is not None else self.num_local_round
        saved_iterations = self.num_local_round - actual_iterations

        print(f"{'─'*80}")
        print(f"[Cliente {self.cid}] TREINAMENTO CONCLUÍDO - Round {global_round}")
        if early_stopping_rounds and saved_iterations > 0:
            print(f"  [EARLY STOPPING] Parou em {actual_iterations}/{self.num_local_round} iterações (economizou {saved_iterations})")
        print(f"{'─'*80}\n")

        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = bst.predict(self.valid_dmatrix)
                metrics = calculate_comprehensive_metrics(self.y_valid, y_pred_proba)
                print_metrics_summary(metrics, client_id=self.cid, round_num=global_round)
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Erro ao calcular métricas: {e}")

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
