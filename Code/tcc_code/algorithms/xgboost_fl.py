"""
Módulo para Federated Learning com XGBoost
Baseado no código funcional de archive/xgboost.py
"""

import os
import inspect
import xgboost as xgb
import torch
from typing import Dict
from flwr.client import Client, ClientApp
from flwr.common import (
    Code, EvaluateIns, EvaluateRes, FitIns, FitRes,
    Parameters, Status, Scalar
)
from flwr.common.config import unflatten_dict
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
from flwr.simulation import run_simulation
from logging import INFO, WARNING

from common import (
    DataProcessor, replace_keys, calculate_comprehensive_metrics,
    print_metrics_summary, evaluate_metrics_aggregation
)


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


def create_client_fn(data_processor: DataProcessor, num_local_round: int, params: Dict):
    """Factory function para criar função de cliente"""

    def client_fn(context: Context):
        node_cfg = context.node_config or {}
        partition_id = int(node_cfg.get("partition-id", 0))

        cfg = replace_keys(unflatten_dict(context.run_config))
        test_fraction = float(cfg.get("test_fraction", 0.2))
        centralised_eval_client = cfg.get("centralised_eval_client", False)

        train_X, train_y, valid_X, valid_y = data_processor.get_client_data(
            partition_id, test_fraction, centralised_eval_client
        )

        train_dmatrix = xgb.DMatrix(train_X, label=train_y)
        valid_dmatrix = xgb.DMatrix(valid_X, label=valid_y)
        num_train = train_X.shape[0]
        num_val = valid_X.shape[0]

        return XGBoostClient(
            train_dmatrix, valid_dmatrix, num_train, num_val,
            num_local_round, params, cfg.get("train_method", "cyclic"),
            partition_id, X_valid=valid_X, y_valid=valid_y
        ).to_client()

    return client_fn


def get_evaluate_fn(data_processor: DataProcessor, params: Dict):
    """Cria função de avaliação centralizada do servidor"""

    def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Scalar]):
        if server_round == 0:
            return 0.693147, {"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "auc": 0.5}

        try:
            bst = xgb.Booster(params=params)
            if parameters.tensors:
                para_b = bytearray(parameters.tensors[-1])
                bst.load_model(para_b)

                test_dmatrix = xgb.DMatrix(data_processor.X_test_all, label=data_processor.y_test_all)
                y_pred_proba = bst.predict(test_dmatrix)
                comprehensive_metrics = calculate_comprehensive_metrics(data_processor.y_test_all, y_pred_proba)
                print_metrics_summary(comprehensive_metrics, round_num=server_round)

                return_metrics = {k: v for k, v in comprehensive_metrics.items() if k != 'confusion_matrix'}
                return float(1.0 - comprehensive_metrics['accuracy']), return_metrics

        except Exception as e:
            log(WARNING, f"Erro na avaliação do servidor: {e}")

        return 0.693147, {"accuracy": 0.5, "auc": 0.5}

    return evaluate_fn


def config_func(rnd: int) -> Dict[str, str]:
    """Configuração enviada aos clientes a cada rodada"""
    return {"global_round": str(rnd)}


def create_server_fn(data_processor: DataProcessor, num_server_rounds: int, params: Dict):
    """Factory function para criar função do servidor"""

    def server_fn(context: Context):
        cfg = replace_keys(unflatten_dict(context.run_config))

        num_rounds = int(cfg.get("num_server_rounds", num_server_rounds))
        fraction_fit = float(cfg.get("fraction_fit", 1.0))
        fraction_evaluate = float(cfg.get("fraction_evaluate", 1.0))
        train_method = cfg.get("train_method", "cyclic")
        centralised_eval = cfg.get("centralised_eval", True)

        parameters = Parameters(tensor_type="", tensors=[])
        evaluate_fn = get_evaluate_fn(data_processor, params) if centralised_eval else None
        fraction_eval = 0.0 if centralised_eval and train_method == "bagging" else fraction_evaluate
        agg_fn = evaluate_metrics_aggregation if fraction_eval > 0 else None

        if train_method == "bagging":
            strategy = FedXgbBagging(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_eval,
                on_evaluate_config_fn=config_func,
                on_fit_config_fn=config_func,
                evaluate_metrics_aggregation_fn=agg_fn,
                initial_parameters=parameters,
            )
        else:  # cyclic
            strategy = FedXgbCyclic(
                fraction_fit=1.0,
                fraction_evaluate=fraction_eval,
                evaluate_metrics_aggregation_fn=agg_fn,
                on_evaluate_config_fn=config_func,
                on_fit_config_fn=config_func,
                initial_parameters=parameters,
            )

        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn


def safe_run_simulation(server_app, client_app, num_supernodes, backend_config=None, run_cfg=None):
    """Executa simulação com fallback para diferentes versões do Flower"""
    sig = inspect.signature(run_simulation)
    params = sig.parameters.keys()

    kwargs = {"server_app": server_app, "client_app": client_app, "num_supernodes": num_supernodes}

    if backend_config is not None and "backend_config" in params:
        kwargs["backend_config"] = backend_config

    if run_cfg is not None:
        if "run_config" in params:
            kwargs["run_config"] = run_cfg
        elif "config" in params:
            kwargs["config"] = run_cfg

    try:
        log(INFO, "Chamando run_simulation com args: %s", list(kwargs.keys()))
        return run_simulation(**kwargs)
    except TypeError:
        log(WARNING, "Fallback: tentando sem run_config")
        kwargs2 = {"server_app": server_app, "client_app": client_app, "num_supernodes": num_supernodes}
        if backend_config is not None and "backend_config" in params:
            kwargs2["backend_config"] = backend_config
        return run_simulation(**kwargs2)


def run_xgboost_experiment(data_processor: DataProcessor, num_clients: int,
                          num_server_rounds: int, num_local_boost_round: int,
                          train_method: str = "cyclic", seed: int = 42):
    """
    Executa experimento de Federated Learning com XGBoost

    Args:
        data_processor: Processador de dados já inicializado
        num_clients: Número de clientes
        num_server_rounds: Número de rodadas
        num_local_boost_round: Rodadas locais de boosting
        train_method: 'cyclic' ou 'bagging'
        seed: Seed para reprodutibilidade

    Returns:
        Histórico de resultados
    """
    USE_GPU = torch.cuda.is_available()
    tree_method = "gpu_hist" if USE_GPU else "hist"

    log(INFO, f"GPU disponível: {USE_GPU} | tree_method: {tree_method}")

    params = {
        "eta": 0.1,
        "max_depth": 6,
        "tree_method": tree_method,
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
    }

    # Criar aplicações cliente e servidor
    client_fn = create_client_fn(data_processor, num_local_boost_round, params)
    client_app = ClientApp(client_fn=client_fn)

    server_fn = create_server_fn(data_processor, num_server_rounds, params)
    server_app = ServerApp(server_fn=server_fn)

    # Configurar recursos
    backend_config = {
        "client_resources": {
            "num_cpus": 1.0,
            "num_gpus": 1.0 / num_clients if USE_GPU else 0.0
        }
    }

    run_cfg = {
        "num-server-rounds": str(num_server_rounds),
        "fraction-fit": "1.0",
        "fraction-evaluate": "1.0",
        "local-epochs": str(num_local_boost_round),
        "train-method": train_method,
        "partitioner-type": "uniform",
        "seed": str(seed),
        "test-fraction": "0.2",
        "centralised-eval": "True",
        "centralised-eval-client": "False",
        "scaled-lr": "False",
        "params": params,
    }

    print(f"\n{'*'*50}")
    print(f"EXECUTANDO XGBOOST - ESTRATÉGIA: {train_method.upper()}")
    print(f"{'*'*50}")

    result = safe_run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config=backend_config,
        run_cfg=run_cfg,
    )

    return result
