"""
Estratégia FedBagging para Federated Learning.

Implementa agregação estilo bagging onde todos (ou fração) dos clientes
treinam em paralelo e modelos são agregados.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
from logging import INFO

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

from .base_strategy import BaseStrategy


class FedBagging(BaseStrategy):
    """
    Estratégia de Federated Bagging.

    Seleciona fração de clientes para treinar em paralelo a cada round,
    agrega modelos resultantes. Agnóstica ao framework de ML.

    Attributes:
        Herda todos os atributos de BaseStrategy.
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
        Inicializa estratégia FedBagging.

        Args:
            Veja BaseStrategy para documentação dos parâmetros.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
        )

        self.logger.setLevel(INFO)
        self.logger.info("Estratégia FedBagging inicializada")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configura próximo round de treinamento selecionando clientes.

        Args:
            server_round: Número do round atual.
            parameters: Parâmetros atuais do modelo global.
            client_manager: Gerenciador de clientes disponíveis.

        Returns:
            Lista de tuplas (cliente, instruções de fit).
        """
        # Obtém configuração para este round
        config = self._get_fit_config(server_round)

        # Amostra clientes disponíveis
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        self.logger.info(
            f"Round {server_round}: Selecionando {sample_size} de "
            f"{client_manager.num_available()} clientes disponíveis para treinamento"
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Cria instruções de fit para cada cliente
        fit_ins = FitIns(parameters, config)

        self.logger.info(
            f"Round {server_round}: Configurado treinamento para "
            f"{len(clients)} clientes"
        )

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Agrega resultados de treinamento dos clientes.

        Implementação simplificada: usa primeiro modelo mas loga quantos foram agregados.
        Para agregação real de modelos tree-based, seria necessário lógica específica
        de cada framework (ensemble, averaging de árvores, etc.).

        Args:
            server_round: Número do round atual.
            results: Resultados bem-sucedidos dos clientes.
            failures: Falhas que ocorreram.

        Returns:
            Tupla (parâmetros agregados, métricas agregadas).
        """
        if not results:
            self.logger.warning(f"Round {server_round}: Nenhum resultado para agregar")
            return None, {}

        # Loga falhas se houver
        if failures:
            self.logger.warning(
                f"Round {server_round}: {len(failures)} clientes falharam no treinamento"
            )

        self.logger.info(
            f"Round {server_round}: Agregando {len(results)} modelos de clientes"
        )

        # Agregação simplificada: usa primeiro modelo
        # TODO: Implementar agregação real específica por framework
        # - XGBoost: Ensemble de boosters ou média de árvores
        # - CatBoost: Similar ao XGBoost
        # - LightGBM: Similar ao XGBoost
        aggregated_parameters = results[0][1].parameters

        self.logger.info(
            f"Round {server_round}: Modelo agregado criado "
            f"(simplificado: primeiro de {len(results)} modelos)"
        )

        # Agrega métricas de treinamento
        metrics_aggregated = {}
        if results:
            # Coleta métricas de todos os clientes
            metrics_list = []
            for client_proxy, fit_res in results:
                metrics_list.append((fit_res.num_examples, fit_res.metrics))

            # Usa função de agregação customizada ou padrão
            metrics_aggregated = self._aggregate_metrics(metrics_list)

            self.logger.info(
                f"Round {server_round}: Métricas agregadas = {metrics_aggregated}"
            )

        return aggregated_parameters, metrics_aggregated

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
        # Se não quisermos avaliação em clientes, retorna lista vazia
        if self.fraction_evaluate == 0.0:
            return []

        # Obtém configuração para este round
        config = self._get_evaluate_config(server_round)

        # Amostra clientes para avaliação
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        self.logger.info(
            f"Round {server_round}: Selecionando {sample_size} de "
            f"{client_manager.num_available()} clientes para avaliação"
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Cria instruções de evaluate para cada cliente
        evaluate_ins = EvaluateIns(parameters, config)

        self.logger.info(
            f"Round {server_round}: Configurada avaliação para {len(clients)} clientes"
        )

        return [(client, evaluate_ins) for client in clients]

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
        if not results:
            self.logger.warning(
                f"Round {server_round}: Nenhum resultado de avaliação para agregar"
            )
            return None, {}

        # Loga falhas se houver
        if failures:
            self.logger.warning(
                f"Round {server_round}: {len(failures)} clientes falharam na avaliação"
            )

        self.logger.info(
            f"Round {server_round}: Agregando resultados de avaliação de "
            f"{len(results)} clientes"
        )

        # Agrega loss (média ponderada)
        total_examples = sum(num_examples for _, res in results for num_examples in [res.num_examples])
        aggregated_loss = sum(
            res.loss * res.num_examples for _, res in results
        ) / total_examples

        # Agrega métricas
        metrics_list = []
        for _, evaluate_res in results:
            metrics_list.append((evaluate_res.num_examples, evaluate_res.metrics))

        metrics_aggregated = self._aggregate_metrics(metrics_list)

        self.logger.info(
            f"Round {server_round}: Loss agregada = {aggregated_loss:.4f}, "
            f"Métricas agregadas = {metrics_aggregated}"
        )

        return aggregated_loss, metrics_aggregated

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """
        Calcula número de clientes para treinamento.

        Args:
            num_available_clients: Número de clientes disponíveis.

        Returns:
            Tupla (sample_size, min_num_clients).
        """
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_fit_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """
        Calcula número de clientes para avaliação.

        Args:
            num_available_clients: Número de clientes disponíveis.

        Returns:
            Tupla (sample_size, min_num_clients).
        """
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_evaluate_clients
