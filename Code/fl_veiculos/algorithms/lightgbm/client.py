"""
Cliente Federated Learning para LightGBM
"""

import os
import numpy as np
import lightgbm as lgb
from flwr.client import Client
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Status
from flwr.common.logger import log
from logging import INFO, WARNING

from common import (
    calculate_comprehensive_metrics, print_metrics_summary,
    ClassBalancingHelper, CurriculumLearning
)


class VerboseCallback:
    """Callback para mostrar progresso detalhado do treinamento no terminal"""

    def __init__(self, client_id: int, global_round: int, num_rounds: int):
        self.client_id = client_id
        self.global_round = global_round
        self.num_rounds = num_rounds

    def __call__(self, env):
        """Chamado após cada iteração de boosting"""
        iteration = env.iteration

        if env.evaluation_result_list:
            result = env.evaluation_result_list[0]
            metric_value = result[2]

            print(f"  [Cliente {self.client_id}] Round {self.global_round} | Época {iteration+1:2d}/{self.num_rounds:2d} | "
                  f"Valid: {metric_value:.4f}")


class LightGBMClient(Client):
    """Cliente Federated Learning para LightGBM"""

    def __init__(self, train_data, valid_data, num_train, num_val,
                 num_local_round, params, train_method, cid: int,
                 X_valid=None, y_valid=None, train_y=None,
                 advanced_config=None):
        self.train_data = train_data
        self.valid_data = valid_data
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
                self.train_data.set_weight(sample_weights)
                print(f"  [CLASS WEIGHTS] Aplicados: {class_weights}")

            elif self.advanced_config.get('use_sample_weights'):
                sample_weights = self.class_balancer.compute_sample_weights(self.train_y)
                self.train_data.set_weight(sample_weights)
                print(f"  [SAMPLE WEIGHTS] Aplicados: min={sample_weights.min():.2f}, "
                      f"max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")

        print(f"{'─'*80}")

        verbose_callback = VerboseCallback(self.cid, global_round, self.num_local_round)





        if global_round <= 1 or not ins.parameters.tensors:
            bst = lgb.train(
                self.params,
                self.train_data,
                valid_sets=[self.valid_data],
                valid_names=['valid'],
                callbacks=[verbose_callback],
            )
        else:
            try:
                global_model_bytes = bytearray(ins.parameters.tensors[0])
                temp_model_path = f"/tmp/lgb_global_model_{self.cid}_{global_round}.txt"

                with open(temp_model_path, 'wb') as f:
                    f.write(global_model_bytes)

                bst = lgb.train(
                    self.params,
                    self.train_data,
                    num_boost_round=self.num_local_round,
                    init_model=temp_model_path,
                    valid_sets=[self.valid_data],
                    valid_names=['valid'],
                    fobj=fobj,
                    callbacks=[verbose_callback],
                )

                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)

            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Falha ao carregar modelo global: {e}; treinando do zero.")
                bst = lgb.train(
                    self.params,
                    self.train_data,
                    num_boost_round=self.num_local_round,
                    valid_names=['valid'],
                    callbacks=[verbose_callback],
                )

        print(f"{'─'*80}")
        print(f"[Cliente {self.cid}] TREINAMENTO CONCLUÍDO - Round {global_round}")
        print(f"{'─'*80}\n")

        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = bst.predict(self.X_valid)
                metrics = calculate_comprehensive_metrics(self.y_valid, y_pred_proba)
                print_metrics_summary(metrics, client_id=self.cid, round_num=global_round)
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Erro ao calcular métricas: {e}")

        temp_save_path = f"/tmp/lgb_local_model_{self.cid}_{global_round}.txt"
        bst.save_model(temp_save_path)

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

        temp_model_path = f"/tmp/lgb_eval_model_{self.cid}.txt"
        model_bytes = bytearray(ins.parameters.tensors[0])

        with open(temp_model_path, 'wb') as f:
            f.write(model_bytes)

        bst = lgb.Booster(model_file=temp_model_path)

        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = bst.predict(self.X_valid)
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
