"""
Estratégia base abstrata para Federated Learning.

Define interface comum para todas as estratégias de agregação.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union
from logging import Logger, getLogger

from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    NDArrays,
)


class BaseStrategy(Strategy, ABC):
    """
    Classe base abstrata para estratégias de Federated Learning.

    Fornece interface comum e atributos compartilhados para todas as estratégias
    de agregação. Agnóstica ao framework de ML (XGBoost, CatBoost, LightGBM).

    Attributes:
        fraction_fit: Fração de clientes a selecionar para treinamento.
        fraction_evaluate: Fração de clientes a selecionar para avaliação.
        min_fit_clients: Número mínimo de clientes para treinamento.
        min_evaluate_clients: Número mínimo de clientes para avaliação.
        min_available_clients: Número mínimo de clientes disponíveis.
        evaluate_fn: Função de avaliação centralizada no servidor (opcional).
        evaluate_metrics_aggregation_fn: Função para agregar métricas de avaliação.
        on_fit_config_fn: Função para configurar treinamento (opcional).
        on_evaluate_config_fn: Função para configurar avaliação (opcional).
        initial_parameters: Parâmetros iniciais do modelo global (opcional).
        logger: Logger para rastreamento de eventos.
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]]
            ]
        ] = None,
        evaluate_metrics_aggregation_fn: Optional[
            Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        """
        Inicializa estratégia base.

        Args:
            fraction_fit: Fração de clientes para treinamento (0.0 a 1.0).
            fraction_evaluate: Fração de clientes para avaliação (0.0 a 1.0).
            min_fit_clients: Número mínimo de clientes para fit.
            min_evaluate_clients: Número mínimo de clientes para evaluate.
            min_available_clients: Número mínimo de clientes disponíveis.
            evaluate_fn: Função de avaliação centralizada (servidor).
            evaluate_metrics_aggregation_fn: Agregação customizada de métricas.
            on_fit_config_fn: Configuração para cada round de fit.
            on_evaluate_config_fn: Configuração para cada round de evaluate.
            initial_parameters: Parâmetros iniciais do modelo.
        """
        super().__init__()

        # Validação de parâmetros
        if not 0.0 <= fraction_fit <= 1.0:
            raise ValueError("fraction_fit deve estar entre 0.0 e 1.0")
        if not 0.0 <= fraction_evaluate <= 1.0:
            raise ValueError("fraction_evaluate deve estar entre 0.0 e 1.0")
        if min_fit_clients < 1:
            raise ValueError("min_fit_clients deve ser pelo menos 1")
        if min_evaluate_clients < 1:
            raise ValueError("min_evaluate_clients deve ser pelo menos 1")
        if min_available_clients < min_fit_clients:
            raise ValueError(
                "min_available_clients deve ser >= min_fit_clients"
            )

        # Atributos de seleção de clientes
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        # Funções de callback
        self.evaluate_fn = evaluate_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn

        # Parâmetros do modelo
        self.initial_parameters = initial_parameters

        # Logger
        self.logger: Logger = getLogger(self.__class__.__name__)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
        Inicializa parâmetros globais do modelo.

        Args:
            client_manager: Gerenciador de clientes Flower.

        Returns:
            Parâmetros iniciais ou None.
        """
        if self.initial_parameters is not None:
            self.logger.info("Usando parâmetros iniciais fornecidos")
            return self.initial_parameters

        self.logger.info("Nenhum parâmetro inicial fornecido")
        return None

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Avalia modelo global no servidor (avaliação centralizada).

        Args:
            server_round: Número do round atual.
            parameters: Parâmetros do modelo global.

        Returns:
            Tupla (loss, metrics) ou None se evaluate_fn não fornecido.
        """
        if self.evaluate_fn is None:
            return None

        # Converte Parameters para NDArrays
        parameters_ndarrays = parameters.tensors

        self.logger.info(f"Round {server_round}: Executando avaliação centralizada")

        # Executa avaliação
        eval_result = self.evaluate_fn(server_round, parameters_ndarrays, {})

        if eval_result is None:
            return None

        loss, metrics = eval_result
        self.logger.info(
            f"Round {server_round}: Loss centralizada = {loss:.4f}, "
            f"Métricas = {metrics}"
        )

        return loss, metrics

    @abstractmethod
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configura próximo round de treinamento.

        Args:
            server_round: Número do round atual.
            parameters: Parâmetros atuais do modelo global.
            client_manager: Gerenciador de clientes disponíveis.

        Returns:
            Lista de tuplas (cliente, instruções de fit).
        """
        pass

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Agrega resultados de treinamento dos clientes.

        Args:
            server_round: Número do round atual.
            results: Resultados bem-sucedidos dos clientes.
            failures: Falhas que ocorreram.

        Returns:
            Tupla (parâmetros agregados, métricas agregadas).
        """
        pass

    @abstractmethod
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configura próximo round de avaliação.

        Args:
            server_round: Número do round atual.
            parameters: Parâmetros atuais do modelo global.
            client_manager: Gerenciador de clientes disponíveis.

        Returns:
            Lista de tuplas (cliente, instruções de evaluate).
        """
        pass

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Agrega resultados de avaliação dos clientes.

        Args:
            server_round: Número do round atual.
            results: Resultados bem-sucedidos dos clientes.
            failures: Falhas que ocorreram.

        Returns:
            Tupla (loss agregada, métricas agregadas).
        """
        pass

    def _get_fit_config(self, server_round: int) -> Dict[str, Scalar]:
        """
        Obtém configuração para round de treinamento.

        Args:
            server_round: Número do round atual.

        Returns:
            Dicionário de configuração.
        """
        if self.on_fit_config_fn is not None:
            return self.on_fit_config_fn(server_round)
        return {}

    def _get_evaluate_config(self, server_round: int) -> Dict[str, Scalar]:
        """
        Obtém configuração para round de avaliação.

        Args:
            server_round: Número do round atual.

        Returns:
            Dicionário de configuração.
        """
        if self.on_evaluate_config_fn is not None:
            return self.on_evaluate_config_fn(server_round)
        return {}

    def _aggregate_metrics(
        self, results: List[Tuple[int, Dict[str, Scalar]]]
    ) -> Dict[str, Scalar]:
        """
        Agrega métricas de múltiplos clientes.

        Args:
            results: Lista de tuplas (num_examples, metrics).

        Returns:
            Métricas agregadas.
        """
        if self.evaluate_metrics_aggregation_fn is not None:
            return self.evaluate_metrics_aggregation_fn(results)

        # Agregação padrão: média ponderada por número de exemplos
        if not results:
            return {}

        # Calcula total de exemplos
        total_examples = sum(num_examples for num_examples, _ in results)

        # Identifica todas as métricas disponíveis
        all_metrics = set()
        for _, metrics in results:
            all_metrics.update(metrics.keys())

        # Calcula média ponderada para cada métrica
        aggregated = {}
        for metric_name in all_metrics:
            weighted_sum = 0.0
            for num_examples, metrics in results:
                if metric_name in metrics:
                    weighted_sum += num_examples * float(metrics[metric_name])

            aggregated[metric_name] = weighted_sum / total_examples

        return aggregated
