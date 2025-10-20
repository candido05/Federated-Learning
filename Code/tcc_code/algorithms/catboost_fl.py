"""
Módulo para Federated Learning com CatBoost
Baseado no código funcional de archive/catbbost.py
"""

import os
import inspect
from catboost import CatBoost, Pool
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
from flwr.server.strategy import Strategy
from flwr.simulation import run_simulation
from logging import INFO, WARNING

from common import (
    DataProcessor, replace_keys, calculate_comprehensive_metrics,
    print_metrics_summary, evaluate_metrics_aggregation
)


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

        if global_round <= 1 or not ins.parameters.tensors:
            # Primeira rodada: treinar do zero
            model = CatBoost(self.params)
            model.fit(self.train_pool, eval_set=self.valid_pool, verbose=False)
        else:
            # Carregar modelo global e continuar treinamento
            try:
                global_model_bytes = bytearray(ins.parameters.tensors[0])
                temp_model_path = f"/tmp/catboost_global_model_{self.cid}_{global_round}.cbm"
                with open(temp_model_path, 'wb') as f:
                    f.write(global_model_bytes)

                model = CatBoost(self.params)
                model.load_model(temp_model_path)
                model.fit(self.train_pool, eval_set=self.valid_pool, verbose=False, init_model=model)

                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)

            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Falha ao carregar modelo global: {e}; treinando do zero.")
                model = CatBoost(self.params)
                model.fit(self.train_pool, eval_set=self.valid_pool, verbose=False)

        # Calcular métricas avançadas
        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = model.predict(self.valid_pool, prediction_type='Probability')[:, 1]
                metrics = calculate_comprehensive_metrics(self.y_valid, y_pred_proba)
                print_metrics_summary(metrics, client_id=self.cid, round_num=global_round)
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Erro ao calcular métricas: {e}")

        # Salvar modelo local
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

        # Carregar modelo
        temp_model_path = f"/tmp/catboost_eval_model_{self.cid}.cbm"
        model_bytes = bytearray(ins.parameters.tensors[0])

        with open(temp_model_path, 'wb') as f:
            f.write(model_bytes)

        model = CatBoost(self.params)
        model.load_model(temp_model_path)

        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

        # Calcular métricas avançadas
        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = model.predict(self.valid_pool, prediction_type='Probability')[:, 1]
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


# Estratégias customizadas para CatBoost

class FedCatBoostBagging(Strategy):
    """Estratégia de Bagging para CatBoost"""

    def __init__(self, fraction_fit=1.0, fraction_evaluate=1.0, evaluate_fn=None,
                 evaluate_metrics_aggregation_fn=None, on_fit_config_fn=None,
                 on_evaluate_config_fn=None, initial_parameters=None):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.evaluate_fn = evaluate_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.initial_parameters = initial_parameters or Parameters(tensor_type="", tensors=[])

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        clients = client_manager.sample(
            num_clients=int(client_manager.num_available() * self.fraction_fit),
            min_num_clients=1,
        )

        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        aggregated_parameters = results[0][1].parameters
        log(INFO, f"[Bagging] Agregando {len(results)} modelos na rodada {server_round}")

        return aggregated_parameters, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        clients = client_manager.sample(
            num_clients=int(client_manager.num_available() * self.fraction_evaluate),
            min_num_clients=1,
        )

        return [(client, EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        if self.evaluate_metrics_aggregation_fn:
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                [(res.num_examples, res.metrics) for _, res in results]
            )
        else:
            metrics_aggregated = {}

        return None, metrics_aggregated

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters, {})


class FedCatBoostCyclic(Strategy):
    """Estratégia Cíclica para CatBoost"""

    def __init__(self, fraction_fit=1.0, fraction_evaluate=0.0, evaluate_fn=None,
                 evaluate_metrics_aggregation_fn=None, on_fit_config_fn=None,
                 on_evaluate_config_fn=None, initial_parameters=None):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.evaluate_fn = evaluate_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.initial_parameters = initial_parameters or Parameters(tensor_type="", tensors=[])
        self.current_client_idx = 0

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        all_clients = list(client_manager.all().values())

        if not all_clients:
            return []

        client = all_clients[self.current_client_idx % len(all_clients)]
        self.current_client_idx += 1

        log(INFO, f"[Cyclic] Rodada {server_round}: treinando com cliente {(self.current_client_idx - 1) % len(all_clients)}")

        return [(client, FitIns(parameters, config))]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        aggregated_parameters = results[0][1].parameters

        return aggregated_parameters, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        clients = client_manager.sample(
            num_clients=int(client_manager.num_available() * self.fraction_evaluate),
            min_num_clients=1,
        )

        return [(client, EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        if self.evaluate_metrics_aggregation_fn:
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                [(res.num_examples, res.metrics) for _, res in results]
            )
        else:
            metrics_aggregated = {}

        return None, metrics_aggregated

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters, {})


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

        train_pool = Pool(train_X, label=train_y)
        valid_pool = Pool(valid_X, label=valid_y)
        num_train = train_X.shape[0]
        num_val = valid_X.shape[0]

        return CatBoostClient(
            train_pool, valid_pool, num_train, num_val,
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
            temp_model_path = "/tmp/catboost_server_eval.cbm"
            if parameters.tensors:
                model_bytes = bytearray(parameters.tensors[-1])
                with open(temp_model_path, 'wb') as f:
                    f.write(model_bytes)

                model = CatBoost(params)
                model.load_model(temp_model_path)

                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)

                test_pool = Pool(data_processor.X_test_all, label=data_processor.y_test_all)
                y_pred_proba = model.predict(test_pool, prediction_type='Probability')[:, 1]
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
            strategy = FedCatBoostBagging(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_eval,
                evaluate_fn=evaluate_fn,
                evaluate_metrics_aggregation_fn=agg_fn,
                on_fit_config_fn=config_func,
                on_evaluate_config_fn=config_func,
                initial_parameters=parameters,
            )
        else:  # cyclic
            strategy = FedCatBoostCyclic(
                fraction_fit=1.0,
                fraction_evaluate=fraction_eval,
                evaluate_fn=evaluate_fn,
                evaluate_metrics_aggregation_fn=agg_fn,
                on_fit_config_fn=config_func,
                on_evaluate_config_fn=config_func,
                initial_parameters=parameters,
            )

        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn


def safe_run_simulation(server_app, client_app, num_supernodes, backend_config=None, run_cfg=None):
    """Executa simulação com fallback para diferentes versões do Flower"""
    sig = inspect.signature(run_simulation)
    params_keys = sig.parameters.keys()

    kwargs = {"server_app": server_app, "client_app": client_app, "num_supernodes": num_supernodes}

    if backend_config is not None and "backend_config" in params_keys:
        kwargs["backend_config"] = backend_config

    if run_cfg is not None:
        if "run_config" in params_keys:
            kwargs["run_config"] = run_cfg
        elif "config" in params_keys:
            kwargs["config"] = run_cfg

    try:
        log(INFO, "Chamando run_simulation com args: %s", list(kwargs.keys()))
        return run_simulation(**kwargs)
    except TypeError:
        log(WARNING, "Fallback: tentando sem run_config")
        kwargs2 = {"server_app": server_app, "client_app": client_app, "num_supernodes": num_supernodes}
        if backend_config is not None and "backend_config" in params_keys:
            kwargs2["backend_config"] = backend_config
        return run_simulation(**kwargs2)


def run_catboost_experiment(data_processor: DataProcessor, num_clients: int,
                            num_server_rounds: int, num_local_boost_round: int,
                            train_method: str = "cyclic", seed: int = 42):
    """
    Executa experimento de Federated Learning com CatBoost

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
    task_type = "GPU" if USE_GPU else "CPU"

    log(INFO, f"GPU disponível: {USE_GPU} | task_type: {task_type}")

    params = {
        "iterations": num_local_boost_round,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.1,
        "depth": 6,
        "task_type": task_type,
        "verbose": False,
        "random_seed": seed
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
    print(f"EXECUTANDO CATBOOST - ESTRATÉGIA: {train_method.upper()}")
    print(f"{'*'*50}")

    result = safe_run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config=backend_config,
        run_cfg=run_cfg,
    )

    return result
