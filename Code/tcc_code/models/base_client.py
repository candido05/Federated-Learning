"""Classe base abstrata para clientes de Federated Learning.

Este módulo define a interface abstrata que todos os clientes FL devem implementar,
fornecendo funcionalidades comuns de treinamento e avaliação.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
from flwr.client import Client
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
    Code,
)

from utils import ExperimentLogger, MetricsCalculator

logger = logging.getLogger(__name__)


class BaseFLClient(Client, ABC):
    """Classe base abstrata para clientes de Federated Learning.

    Todos os clientes concretos (XGBoost, LightGBM, CatBoost) devem herdar
    desta classe e implementar os métodos abstratos específicos do modelo.

    Attributes:
        train_data: Tupla (X_train, y_train) com dados de treinamento do cliente.
        valid_data: Tupla (X_valid, y_valid) com dados de validação do cliente.
        num_train: Número de amostras de treinamento.
        num_val: Número de amostras de validação.
        num_local_round: Número de rodadas de treinamento local.
        params: Hiperparâmetros do modelo.
        train_method: Método de treinamento ("fit", "update", etc.).
        client_id: Identificador único do cliente.
        X_valid: Features de validação.
        y_valid: Labels de validação.
        logger: Logger de experimento para registrar métricas.
    """

    def __init__(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        valid_data: Tuple[np.ndarray, np.ndarray],
        num_local_round: int,
        params: dict,
        train_method: str = "fit",
        client_id: str = "0",
        experiment_logger: Optional[ExperimentLogger] = None,
    ) -> None:
        """Inicializa o cliente base de Federated Learning.

        Args:
            train_data: Tupla (X_train, y_train) com dados de treinamento.
            valid_data: Tupla (X_valid, y_valid) com dados de validação.
            num_local_round: Número de rodadas de treinamento local por round global.
            params: Hiperparâmetros específicos do modelo.
            train_method: Método de treinamento (padrão: "fit").
            client_id: Identificador único do cliente (padrão: "0").
            experiment_logger: Logger para registrar métricas (opcional).
        """
        super().__init__()

        # Dados de treinamento e validação
        self.train_data = train_data
        self.valid_data = valid_data
        X_train, y_train = train_data
        self.X_valid, self.y_valid = valid_data

        # Tamanhos dos datasets
        self.num_train = len(X_train)
        self.num_val = len(self.X_valid)

        # Configurações de treinamento
        self.num_local_round = num_local_round
        self.params = params
        self.train_method = train_method
        self.client_id = client_id

        # Logger de experimento
        self.experiment_logger = experiment_logger

        logger.info(
            f"Cliente {self.client_id} inicializado: "
            f"train={self.num_train}, val={self.num_val}, "
            f"local_rounds={self.num_local_round}"
        )

    @abstractmethod
    def train_local_model(self, global_model_bytes: Optional[bytes]) -> Any:
        """Treina o modelo local usando os dados do cliente.

        Args:
            global_model_bytes: Bytes do modelo global (None se primeira rodada).

        Returns:
            Modelo treinado localmente.

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar train_local_model()")

    @abstractmethod
    def save_model_bytes(self, model: Any) -> bytes:
        """Serializa o modelo treinado em bytes.

        Args:
            model: Modelo treinado a ser serializado.

        Returns:
            Bytes representando o modelo serializado.

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar save_model_bytes()")

    @abstractmethod
    def load_model_from_bytes(self, model_bytes: bytes) -> Any:
        """Carrega um modelo a partir de bytes serializados.

        Args:
            model_bytes: Bytes do modelo serializado.

        Returns:
            Modelo carregado e pronto para uso.

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar load_model_from_bytes()")

    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Faz predições usando o modelo.

        Args:
            model: Modelo treinado.
            X: Features para predição (n_samples, n_features).

        Returns:
            Array numpy com predições ou probabilidades.

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar predict()")

    def fit(self, ins: FitIns) -> FitRes:
        """Treina o modelo local e retorna parâmetros atualizados.

        Método concreto que orquestra o processo de treinamento:
        1. Extrai configuração (round number)
        2. Carrega modelo global se existir
        3. Treina modelo local
        4. Calcula métricas
        5. Serializa modelo
        6. Retorna resultado

        Args:
            ins: Instruções de treinamento do Flower (FitIns).

        Returns:
            FitRes contendo parâmetros atualizados, número de amostras e métricas.
        """
        try:
            logger.info(f"[Cliente {self.client_id}] Iniciando fit()")

            # Extrai configuração
            config = ins.config
            global_round = int(config.get("global_round", 0))
            logger.info(f"[Cliente {self.client_id}] Rodada global: {global_round}")

            # Extrai parâmetros do modelo global (se existirem)
            global_model_bytes = None
            if ins.parameters and ins.parameters.tensors:
                global_model_bytes = ins.parameters.tensors[0]
                logger.info(
                    f"[Cliente {self.client_id}] Modelo global recebido: "
                    f"{len(global_model_bytes)} bytes"
                )
            else:
                logger.info(f"[Cliente {self.client_id}] Primeira rodada - sem modelo global")

            # Treina modelo local
            logger.info(f"[Cliente {self.client_id}] Treinando modelo local...")
            trained_model = self.train_local_model(global_model_bytes)
            logger.info(f"[Cliente {self.client_id}] Treinamento local concluído")

            # Avalia modelo nos dados de validação
            logger.info(f"[Cliente {self.client_id}] Avaliando modelo...")
            y_pred_proba = self.predict(trained_model, self.X_valid)

            # Calcula métricas comprehensivas
            metrics = MetricsCalculator.calculate_comprehensive_metrics(
                self.y_valid, y_pred_proba
            )
            logger.info(
                f"[Cliente {self.client_id}] Métricas: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"f1={metrics['f1_score']:.4f}"
            )

            # Loga métricas no logger de experimento
            if self.experiment_logger:
                self.experiment_logger.log_client(
                    client_id=self.client_id,
                    metrics=metrics,
                    round_num=global_round,
                )

            # Serializa modelo treinado
            logger.info(f"[Cliente {self.client_id}] Serializando modelo...")
            model_bytes = self.save_model_bytes(trained_model)
            logger.info(f"[Cliente {self.client_id}] Modelo serializado: {len(model_bytes)} bytes")

            # Prepara parâmetros para retorno
            parameters = Parameters(tensors=[model_bytes], tensor_type="")

            # Prepara métricas para retorno (apenas valores JSON serializáveis)
            metrics_json = MetricsCalculator.metrics_to_json(metrics)

            logger.info(f"[Cliente {self.client_id}] fit() concluído com sucesso")

            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=parameters,
                num_examples=self.num_train,
                metrics=metrics_json,
            )

        except Exception as e:
            logger.error(
                f"[Cliente {self.client_id}] Erro durante fit(): {e}",
                exc_info=True,
            )
            # Retorna resposta de erro
            return FitRes(
                status=Status(code=Code.UNKNOWN_ERROR, message=str(e)),
                parameters=Parameters(tensors=[], tensor_type=""),
                num_examples=0,
                metrics={},
            )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Avalia o modelo global nos dados de validação do cliente.

        Método concreto que orquestra o processo de avaliação:
        1. Carrega modelo global
        2. Faz predições
        3. Calcula métricas comprehensivas
        4. Loga resultados
        5. Retorna métricas

        Args:
            ins: Instruções de avaliação do Flower (EvaluateIns).

        Returns:
            EvaluateRes contendo loss, número de amostras e métricas.
        """
        try:
            logger.info(f"[Cliente {self.client_id}] Iniciando evaluate()")

            # Extrai configuração
            config = ins.config
            global_round = int(config.get("global_round", 0))
            logger.info(f"[Cliente {self.client_id}] Avaliando rodada global: {global_round}")

            # Extrai e carrega modelo global
            if not ins.parameters or not ins.parameters.tensors:
                logger.error(f"[Cliente {self.client_id}] Nenhum parâmetro recebido para avaliação")
                return EvaluateRes(
                    status=Status(code=Code.UNKNOWN_ERROR, message="No parameters"),
                    loss=float("inf"),
                    num_examples=0,
                    metrics={},
                )

            global_model_bytes = ins.parameters.tensors[0]
            logger.info(
                f"[Cliente {self.client_id}] Carregando modelo global: "
                f"{len(global_model_bytes)} bytes"
            )

            model = self.load_model_from_bytes(global_model_bytes)
            logger.info(f"[Cliente {self.client_id}] Modelo global carregado")

            # Faz predições
            logger.info(f"[Cliente {self.client_id}] Fazendo predições...")
            y_pred_proba = self.predict(model, self.X_valid)

            # Calcula métricas comprehensivas
            logger.info(f"[Cliente {self.client_id}] Calculando métricas...")
            metrics = MetricsCalculator.calculate_comprehensive_metrics(
                self.y_valid, y_pred_proba
            )

            # Calcula loss (1 - accuracy como proxy)
            loss = 1.0 - metrics["accuracy"]

            logger.info(
                f"[Cliente {self.client_id}] Avaliação: "
                f"loss={loss:.4f}, accuracy={metrics['accuracy']:.4f}, "
                f"f1={metrics['f1_score']:.4f}"
            )

            # Loga métricas no logger de experimento
            if self.experiment_logger:
                self.experiment_logger.log_client(
                    client_id=self.client_id,
                    metrics=metrics,
                    round_num=global_round,
                )

            # Prepara métricas para retorno
            metrics_json = MetricsCalculator.metrics_to_json(metrics)

            logger.info(f"[Cliente {self.client_id}] evaluate() concluído com sucesso")

            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=loss,
                num_examples=self.num_val,
                metrics=metrics_json,
            )

        except Exception as e:
            logger.error(
                f"[Cliente {self.client_id}] Erro durante evaluate(): {e}",
                exc_info=True,
            )
            # Retorna resposta de erro
            return EvaluateRes(
                status=Status(code=Code.UNKNOWN_ERROR, message=str(e)),
                loss=float("inf"),
                num_examples=0,
                metrics={},
            )

    def get_properties(self, ins) -> dict:
        """Retorna propriedades do cliente.

        Args:
            ins: Instruções do Flower.

        Returns:
            Dicionário com propriedades do cliente.
        """
        return {
            "client_id": self.client_id,
            "num_train": self.num_train,
            "num_val": self.num_val,
            "num_local_round": self.num_local_round,
            "train_method": self.train_method,
        }

    def get_parameters(self, ins) -> Parameters:
        """Retorna os parâmetros do modelo atual.

        Args:
            ins: Instruções do Flower.

        Returns:
            Parameters do Flower (vazio neste caso, modelo é enviado em fit()).
        """
        # Não implementado - modelo é enviado apenas em fit()
        return Parameters(tensors=[], tensor_type="")
