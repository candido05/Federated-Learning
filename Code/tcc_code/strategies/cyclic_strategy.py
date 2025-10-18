"""
Estratégia FedCyclic para Federated Learning.

Implementa agregação cíclica onde apenas 1 cliente treina por vez,
alternando entre clientes de forma round-robin.
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


class FedCyclic(BaseStrategy):
    """
    Estratégia de Federated Cyclic (Round-Robin).

    Seleciona apenas 1 cliente por round para treinar, alternando ciclicamente
    entre todos os clientes. Útil para simular treinamento sequencial ou
    economizar recursos computacionais.

    Attributes:
        current_client_idx: Índice do próximo cliente a ser selecionado.
        Herda demais atributos de BaseStrategy.
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
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
        Inicializa estratégia FedCyclic.

        Args:
            Veja BaseStrategy para documentação dos parâmetros.
            Nota: min_fit_clients é sempre forçado para 1 (cyclic).
        """
        # Força min_fit_clients = 1 para estratégia cíclica
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=1,  # Sempre 1 cliente por round
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
        )

        # Índice do próximo cliente (round-robin)
        self.current_client_idx = 0

        # Cache de IDs de clientes para manter ordem consistente
        self._client_ids_cache: List[str] = []

        self.logger.setLevel(INFO)
        self.logger.info("Estratégia FedCyclic inicializada (1 cliente por round)")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configura próximo round de treinamento selecionando 1 cliente ciclicamente.

        Args:
            server_round: Número do round atual.
            parameters: Parâmetros atuais do modelo global.
            client_manager: Gerenciador de clientes disponíveis.

        Returns:
            Lista com 1 tupla (cliente, instruções de fit).
        """
        # Obtém todos os clientes disponíveis
        all_clients = client_manager.all()
        num_available = len(all_clients)

        if num_available == 0:
            self.logger.error(f"Round {server_round}: Nenhum cliente disponível")
            return []

        # Atualiza cache de IDs se mudou o número de clientes
        current_ids = sorted([client.cid for client in all_clients.values()])
        if current_ids != self._client_ids_cache:
            self.logger.info(
                f"Round {server_round}: Atualizando cache de clientes "
                f"({len(current_ids)} clientes)"
            )
            self._client_ids_cache = current_ids
            # Reseta índice se lista de clientes mudou
            self.current_client_idx = 0

        # Seleciona cliente atual usando módulo para circular
        selected_client_id = self._client_ids_cache[
            self.current_client_idx % len(self._client_ids_cache)
        ]

        # Busca proxy do cliente selecionado
        selected_client = None
        for cid, client_proxy in all_clients.items():
            if cid == selected_client_id:
                selected_client = client_proxy
                break

        if selected_client is None:
            self.logger.error(
                f"Round {server_round}: Cliente {selected_client_id} não encontrado"
            )
            return []

        self.logger.info(
            f"Round {server_round}: Cliente selecionado = {selected_client_id} "
            f"(índice {self.current_client_idx % len(self._client_ids_cache)} de "
            f"{len(self._client_ids_cache)} clientes)"
        )

        # Incrementa índice para próximo round
        self.current_client_idx += 1

        # Obtém configuração para este round
        config = self._get_fit_config(server_round)

        # Cria instruções de fit
        fit_ins = FitIns(parameters, config)

        return [(selected_client, fit_ins)]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Agrega resultados de treinamento (apenas 1 cliente).

        Como apenas 1 cliente treinou, simplesmente retorna o modelo desse cliente.

        Args:
            server_round: Número do round atual.
            results: Resultados bem-sucedidos dos clientes.
            failures: Falhas que ocorreram.

        Returns:
            Tupla (parâmetros do único cliente, métricas do único cliente).
        """
        if not results:
            self.logger.warning(f"Round {server_round}: Nenhum resultado para agregar")
            return None, {}

        # Loga falhas se houver
        if failures:
            self.logger.warning(
                f"Round {server_round}: {len(failures)} falha(s) no treinamento"
            )

        # Deve haver apenas 1 resultado
        if len(results) != 1:
            self.logger.warning(
                f"Round {server_round}: Esperado 1 resultado, recebido {len(results)}. "
                "Usando primeiro resultado."
            )

        client_proxy, fit_res = results[0]

        self.logger.info(
            f"Round {server_round}: Modelo do cliente {client_proxy.cid} aceito como "
            f"modelo global ({fit_res.num_examples} exemplos treinados)"
        )

        # Retorna parâmetros e métricas do único cliente
        return fit_res.parameters, fit_res.metrics

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
                f"Round {server_round}: {len(failures)} falha(s) na avaliação"
            )

        self.logger.info(
            f"Round {server_round}: Agregando resultados de avaliação de "
            f"{len(results)} cliente(s)"
        )

        # Agrega loss (média ponderada)
        total_examples = sum(res.num_examples for _, res in results)
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
