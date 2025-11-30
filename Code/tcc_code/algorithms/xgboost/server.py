"""
Servidor Federated Learning para XGBoost
"""

import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Union
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common.config import unflatten_dict
from flwr.common.context import Context
from logging import INFO, WARNING

from common import (
    DataProcessor, replace_keys, calculate_comprehensive_metrics,
    print_metrics_summary, evaluate_metrics_aggregation, ExperimentLogger
)


def get_evaluate_fn(data_processor: DataProcessor, params: Dict, logger: ExperimentLogger = None):
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

                # Log usando o logger se disponível
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


class FedXgbBaggingCustom(FedAvg):
    """Estratégia Bagging customizada para XGBoost que não deserializa parâmetros"""

    def __init__(self, data_processor=None, params=None, logger=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_processor = data_processor
        self.params = params
        self.logger = logger

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Agrega modelos XGBoost sem tentar deserializar como numpy arrays"""
        if not results:
            return None, {}

        # Para bagging, pega o último modelo (todos treinaram do mesmo ponto inicial)
        # Em uma implementação real de bagging, você faria ensemble ou média
        # Aqui simplificamos pegando o último modelo
        _, fit_res = results[-1]

        # Avaliar manualmente no servidor
        if self.data_processor and self.params and fit_res.parameters.tensors:
            self._evaluate_on_server(server_round, fit_res.parameters)

        return fit_res.parameters, {}

    def _evaluate_on_server(self, server_round: int, parameters: Parameters):
        """Avalia modelo no servidor sem deserializar via Flower"""
        try:
            bst = xgb.Booster(params=self.params)
            para_b = bytearray(parameters.tensors[-1])
            bst.load_model(para_b)

            test_dmatrix = xgb.DMatrix(self.data_processor.X_test_all, label=self.data_processor.y_test_all)
            y_pred_proba = bst.predict(test_dmatrix)
            comprehensive_metrics = calculate_comprehensive_metrics(self.data_processor.y_test_all, y_pred_proba)

            if self.logger:
                self.logger.log_round_metrics(server_round, comprehensive_metrics, source="server")
            else:
                print_metrics_summary(comprehensive_metrics, round_num=server_round)
        except Exception as e:
            log(WARNING, f"Erro na avaliação manual do servidor: {e}")


class FedXgbCyclicCustom(FedAvg):
    """Estratégia Cyclic customizada para XGBoost que passa modelo sequencialmente"""

    def __init__(self, data_processor=None, params=None, logger=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_processor = data_processor
        self.params = params
        self.logger = logger

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Passa modelo de um cliente para o próximo (treinamento cíclico)"""
        if not results:
            return None, {}

        # Para cyclic, simplesmente pega o modelo do cliente que treinou
        _, fit_res = results[0]

        # Avaliar manualmente no servidor
        if self.data_processor and self.params and fit_res.parameters.tensors:
            self._evaluate_on_server(server_round, fit_res.parameters)

        return fit_res.parameters, {}

    def _evaluate_on_server(self, server_round: int, parameters: Parameters):
        """Avalia modelo no servidor sem deserializar via Flower"""
        try:
            bst = xgb.Booster(params=self.params)
            para_b = bytearray(parameters.tensors[-1])
            bst.load_model(para_b)

            test_dmatrix = xgb.DMatrix(self.data_processor.X_test_all, label=self.data_processor.y_test_all)
            y_pred_proba = bst.predict(test_dmatrix)
            comprehensive_metrics = calculate_comprehensive_metrics(self.data_processor.y_test_all, y_pred_proba)

            if self.logger:
                self.logger.log_round_metrics(server_round, comprehensive_metrics, source="server")
            else:
                print_metrics_summary(comprehensive_metrics, round_num=server_round)
        except Exception as e:
            log(WARNING, f"Erro na avaliação manual do servidor: {e}")


def create_server_fn(data_processor: DataProcessor, num_server_rounds: int, params: Dict, logger: ExperimentLogger = None):
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
            strategy = FedXgbBaggingCustom(
                data_processor=data_processor,
                params=params,
                logger=logger,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_eval,
                evaluate_fn=None,  # Desabilitar para evitar deserialização
                on_evaluate_config_fn=config_func,
                on_fit_config_fn=config_func,
                evaluate_metrics_aggregation_fn=agg_fn,
                initial_parameters=parameters,
            )
        else:  # cyclic
            strategy = FedXgbCyclicCustom(
                data_processor=data_processor,
                params=params,
                logger=logger,
                fraction_fit=1.0,
                fraction_evaluate=fraction_eval,
                evaluate_fn=None,  # Desabilitar para evitar deserialização
                evaluate_metrics_aggregation_fn=agg_fn,
                on_evaluate_config_fn=config_func,
                on_fit_config_fn=config_func,
                initial_parameters=parameters,
            )

        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn
