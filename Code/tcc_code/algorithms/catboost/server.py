"""
Servidor Federated Learning para CatBoost
"""

import os
from catboost import CatBoost, Pool
from typing import Dict
from flwr.common import Parameters, Scalar
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
from flwr.common import Code, EvaluateIns, FitIns
from flwr.common.config import unflatten_dict
from flwr.common.context import Context
from logging import INFO, WARNING

from common import (
    DataProcessor, replace_keys, calculate_comprehensive_metrics,
    print_metrics_summary, evaluate_metrics_aggregation, ExperimentLogger,
    FederatedAggregationWeights, DiversityMetrics, ClientCyclingStrategy
)


def get_evaluate_fn(data_processor: DataProcessor, params: Dict, logger: ExperimentLogger = None):
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
                y_pred_proba = model.predict(test_pool, prediction_type='Probability')
                if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                comprehensive_metrics = calculate_comprehensive_metrics(data_processor.y_test_all, y_pred_proba)

                if logger:
                    logger.log_round_metrics(server_round, comprehensive_metrics, source="server")
                else:
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


class FedCatBoostBagging(Strategy):
    """Estratégia de Bagging para CatBoost"""

    def __init__(self, fraction_fit=1.0, fraction_evaluate=1.0, evaluate_fn=None,
                 evaluate_metrics_aggregation_fn=None, on_fit_config_fn=None,
                 on_evaluate_config_fn=None, initial_parameters=None, advanced_config=None):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.evaluate_fn = evaluate_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.initial_parameters = initial_parameters or Parameters(tensor_type="", tensors=[])
        self.advanced_config = advanced_config or {}

        self.aggregation_weights = None
        if self.advanced_config.get('diversity_aggregation'):
            alpha = self.advanced_config.get('diversity_alpha', 0.5)
            self.aggregation_weights = FederatedAggregationWeights(alpha=alpha)

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
                 on_evaluate_config_fn=None, initial_parameters=None, advanced_config=None):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.evaluate_fn = evaluate_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.initial_parameters = initial_parameters or Parameters(tensor_type="", tensors=[])
        self.advanced_config = advanced_config or {}
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


def create_server_fn(data_processor: DataProcessor, num_server_rounds: int, params: Dict, logger: ExperimentLogger = None, advanced_config: Dict = None):
    """Factory function para criar função do servidor"""

    def server_fn(context: Context):
        cfg = replace_keys(unflatten_dict(context.run_config))

        num_rounds = int(cfg.get("num_server_rounds", num_server_rounds))
        fraction_fit = float(cfg.get("fraction_fit", 1.0))
        fraction_evaluate = float(cfg.get("fraction_evaluate", 1.0))
        train_method = cfg.get("train_method", "cyclic")
        centralised_eval = cfg.get("centralised_eval", True)

        parameters = Parameters(tensor_type="", tensors=[])
        evaluate_fn = get_evaluate_fn(data_processor, params, logger) if centralised_eval else None
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
                advanced_config=advanced_config,
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
                advanced_config=advanced_config,
            )

        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn
