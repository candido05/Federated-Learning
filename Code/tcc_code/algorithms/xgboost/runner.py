"""
Runner para executar experimentos XGBoost FL
"""

import inspect
import numpy as np
import xgboost as xgb
import torch
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation
from flwr.common.config import unflatten_dict
from flwr.common.context import Context
from flwr.common.logger import log
from logging import INFO, WARNING

from common import (
    DataProcessor, replace_keys, ExperimentLogger,
    get_stable_tree_params, ClientCyclingStrategy
)
from .client import XGBoostClient
from .server import create_server_fn


def create_client_fn(data_processor: DataProcessor, num_local_round: int, params: dict, advanced_config: dict = None):
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
            partition_id, X_valid=valid_X, y_valid=valid_y,
            train_y=train_y, advanced_config=advanced_config
        ).to_client()

    return client_fn


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
        result = run_simulation(**kwargs)
        log(INFO, f"run_simulation retornou: type={type(result)}, value={result}")
        return result
    except TypeError:
        log(WARNING, "Fallback: tentando sem run_config")
        kwargs2 = {"server_app": server_app, "client_app": client_app, "num_supernodes": num_supernodes}
        if backend_config is not None and "backend_config" in params:
            kwargs2["backend_config"] = backend_config
        result2 = run_simulation(**kwargs2)
        log(INFO, f"run_simulation (fallback) retornou: type={type(result2)}, value={result2}")
        return result2
    except Exception as e:
        log(WARNING, f"Exceção inesperada em run_simulation: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_xgboost_experiment(data_processor: DataProcessor, num_clients: int,
                          num_server_rounds: int, num_local_boost_round: int,
                          train_method: str = "cyclic", seed: int = 42,
                          advanced_config: dict = None):
    """Executa experimento de Federated Learning com XGBoost"""
    advanced_config = advanced_config or {}

    logger = ExperimentLogger(
        algorithm_name="xgboost",
        strategy_name=train_method,
        num_clients=num_clients,
        num_rounds=num_server_rounds,
        num_local_rounds=num_local_boost_round,
        samples_per_client=data_processor.samples_per_client
    )
    logger.start_experiment()

    USE_GPU = torch.cuda.is_available()
    tree_method = "gpu_hist" if USE_GPU else "hist"

    log(INFO, f"GPU disponível: {USE_GPU} | tree_method: {tree_method}")

    num_classes = len(np.unique(data_processor.y_test_all))
    log(INFO, f"Número de classes detectadas: {num_classes}")

    if num_classes == 2:
        objective = "binary:logistic"
        eval_metric = ["logloss", "error"]
    else:
        objective = "multi:softprob"
        eval_metric = ["mlogloss", "merror"]

    params = {
        "eta": 0.1,
        "max_depth": 6,
        "tree_method": tree_method,
        "verbosity": 0,
        "objective": objective,
        "eval_metric": eval_metric,
        "num_class": num_classes if num_classes > 2 else None,
    }

    if advanced_config.get('use_stable_params'):
        stable_params = get_stable_tree_params('xgboost')
        params.update(stable_params)
        log(INFO, f"[STABLE PARAMS] Aplicados: {stable_params}")

    if params["num_class"] is None:
        del params["num_class"]

    advanced_config['num_server_rounds'] = num_server_rounds

    client_fn = create_client_fn(data_processor, num_local_boost_round, params, advanced_config)
    client_app = ClientApp(client_fn=client_fn)

    server_fn = create_server_fn(data_processor, num_server_rounds, params, logger, advanced_config)
    server_app = ServerApp(server_fn=server_fn)

    max_concurrent_clients = min(num_clients, 3)

    backend_config = {
        "client_resources": {
            "num_cpus": 1.0,
            "num_gpus": 1.0 / num_clients if USE_GPU else 0.0
        },
        "init_args": {
            "num_cpus": max_concurrent_clients,
        }
    }

    run_cfg = {
        "num-server-rounds": str(num_server_rounds),
        "fraction-fit": str(1.0 / num_clients) if train_method == "cyclic" else "1.0",
        "fraction-evaluate": "0.0",
        "local-epochs": str(num_local_boost_round),
        "train-method": train_method,
        "partitioner-type": "uniform",
        "seed": str(seed),
        "test-fraction": "0.2",
        "centralised-eval": "False",
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

    metrics_history = logger.end_experiment(final_history=result)

    success = len(metrics_history) > 0

    return {
        "success": success,
        "history": result,
        "metrics": metrics_history,
        "num_rounds": len(metrics_history)
    }
