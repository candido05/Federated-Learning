"""Classe base abstrata para datasets em Aprendizado Federado.

Este módulo define a interface abstrata que todos os datasets devem implementar,
garantindo consistência e modularidade no sistema.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


class BaseDataset(ABC):
    """Classe base abstrata para datasets de Aprendizado Federado.

    Todos os datasets concretos devem herdar desta classe e implementar
    todos os métodos abstratos definidos.

    Attributes:
        config: Objeto de configuração global do experimento.
        scaler: StandardScaler treinado nos dados de treino.
        _data_loaded: Flag indicando se os dados já foram carregados.
    """

    def __init__(self, config) -> None:
        """Inicializa o dataset base.

        Args:
            config: Objeto GlobalConfig com configurações do experimento.
        """
        self.config = config
        self.scaler: StandardScaler = StandardScaler()
        self._data_loaded: bool = False

    @abstractmethod
    def load_data(self) -> None:
        """Carrega os dados do dataset.

        Este método deve ser implementado para carregar dados da fonte
        (arquivo, API, HuggingFace, etc.) e armazená-los internamente.

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar load_data()")

    @abstractmethod
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna os dados de treinamento.

        Returns:
            Tupla (X_train, y_train) onde:
                - X_train: Array numpy de shape (n_samples, n_features) com features
                - y_train: Array numpy de shape (n_samples,) com labels

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar get_train_data()")

    @abstractmethod
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna os dados de teste.

        Returns:
            Tupla (X_test, y_test) onde:
                - X_test: Array numpy de shape (n_samples, n_features) com features
                - y_test: Array numpy de shape (n_samples,) com labels

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar get_test_data()")

    @abstractmethod
    def get_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna os dados de validação.

        Returns:
            Tupla (X_val, y_val) onde:
                - X_val: Array numpy de shape (n_samples, n_features) com features
                - y_val: Array numpy de shape (n_samples,) com labels

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar get_validation_data()")

    @abstractmethod
    def get_partitions(self, num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Particiona os dados de treino entre os clientes.

        Args:
            num_clients: Número de clientes para particionar os dados.

        Returns:
            Lista de tuplas (X_client, y_client) para cada cliente, onde:
                - X_client: Array numpy com features do cliente
                - y_client: Array numpy com labels do cliente

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar get_partitions()")

    @abstractmethod
    def get_scaler(self) -> StandardScaler:
        """Retorna o StandardScaler treinado nos dados de treino.

        Returns:
            StandardScaler já fitted nos dados de treinamento.

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar get_scaler()")

    @abstractmethod
    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """Aplica pré-processamento (scaling) aos dados.

        Args:
            X: Array numpy de shape (n_samples, n_features) com features brutas.

        Returns:
            Array numpy com features transformadas/normalizadas.

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar preprocess()")

    @abstractmethod
    def get_num_features(self) -> int:
        """Retorna o número de features do dataset.

        Returns:
            Número inteiro de features.

        Raises:
            NotImplementedError: Se não for implementado na subclasse.
        """
        raise NotImplementedError("Subclasses devem implementar get_num_features()")

    def get_dataset_info(self) -> dict:
        """Retorna informações gerais sobre o dataset.

        Returns:
            Dicionário com informações do dataset (nome, tamanhos, features, etc.).

        Note:
            Este método pode ser sobrescrito nas subclasses para fornecer
            informações específicas adicionais.
        """
        if not self._data_loaded:
            self.load_data()

        X_train, y_train = self.get_train_data()
        X_test, y_test = self.get_test_data()

        return {
            "dataset_name": self.__class__.__name__,
            "num_features": self.get_num_features(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
            "num_classes": len(np.unique(y_train)),
            "class_distribution_train": dict(zip(*np.unique(y_train, return_counts=True))),
            "class_distribution_test": dict(zip(*np.unique(y_test, return_counts=True))),
        }
