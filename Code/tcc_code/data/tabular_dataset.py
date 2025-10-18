"""Implementação de dataset tabular para Aprendizado Federado.

Este módulo fornece a classe TabularDataset que carrega e processa
datasets tabulares do HuggingFace para experimentos de FL.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class TabularDataset(BaseDataset):
    """Dataset tabular genérico para classificação em Aprendizado Federado.

    Classe genérica que carrega datasets tabulares do HuggingFace.
    Atualmente configurada para o dataset HIGGS, mas pode ser adaptada
    para outros datasets tabulares.

    Attributes:
        config: Configuração global do experimento.
        X_train: Features de treinamento.
        y_train: Labels de treinamento.
        X_test: Features de teste.
        y_test: Labels de teste.
        X_val: Features de validação (opcional).
        y_val: Labels de validação (opcional).
        scaler: StandardScaler treinado nos dados de treino.
        cache_dir: Diretório para cache dos dados.
        dataset_source: Fonte do dataset no HuggingFace (padrão: "jxie/higgs").
    """

    def __init__(
        self,
        config,
        cache_dir: Optional[str] = None,
        dataset_source: str = "jxie/higgs",
    ) -> None:
        """Inicializa o TabularDataset.

        Args:
            config: Objeto GlobalConfig com configurações.
            cache_dir: Diretório para cache (padrão: None = usa cache padrão do HuggingFace).
            dataset_source: Fonte do dataset no HuggingFace (padrão: "jxie/higgs").
        """
        super().__init__(config)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.dataset_source = dataset_source

        # Dados internos
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None

        # Informações do dataset
        self._num_features: Optional[int] = None

    def load_data(self) -> None:
        """Carrega o dataset tabular do HuggingFace.

        Carrega os dados, aplica amostragem baseada na configuração,
        divide em treino/teste, treina o scaler e armazena internamente.

        Raises:
            Exception: Se houver erro ao carregar o dataset.
        """
        if self._data_loaded:
            logger.info("Dados já carregados. Pulando carregamento.")
            return

        logger.info(f"Carregando dataset '{self.dataset_source}' do HuggingFace...")

        try:
            # Carrega o dataset completo do HuggingFace
            dataset = load_dataset(
                self.dataset_source,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )

            logger.info(f"Dataset carregado: {dataset}")

            # Converte HuggingFace dataset para numpy arrays
            # Usa 'train' split do HuggingFace
            hf_train = dataset["train"]

            # Calcula o tamanho total necessário
            total_samples = self.config.num_clients * self.config.sample_per_client

            # Aplica amostragem se necessário
            if len(hf_train) > total_samples:
                logger.info(
                    f"Amostrando {total_samples} de {len(hf_train)} amostras "
                    f"({self.config.num_clients} clientes × {self.config.sample_per_client} amostras/cliente)"
                )
                # Usa seed para reprodutibilidade
                hf_train = hf_train.shuffle(seed=self.config.seed).select(range(total_samples))
            else:
                logger.warning(
                    f"Dataset tem apenas {len(hf_train)} amostras. "
                    f"Solicitado: {total_samples}. Usando todas as amostras disponíveis."
                )

            # Converte para numpy
            X_full, y_full = self.hf_to_xy(hf_train)

            logger.info(f"Shape dos dados: X={X_full.shape}, y={y_full.shape}")

            # Armazena número de features
            self._num_features = X_full.shape[1]

            # Divide em treino e teste
            test_size = self.config.test_fraction
            X_train, X_test, y_train, y_test = train_test_split(
                X_full,
                y_full,
                test_size=test_size,
                random_state=self.config.seed,
                stratify=y_full,  # Mantém distribuição de classes
            )

            logger.info(
                f"Split treino/teste: treino={X_train.shape[0]}, teste={X_test.shape[0]} "
                f"(test_fraction={test_size})"
            )

            # Treina o scaler apenas nos dados de treino
            self.scaler.fit(X_train)
            logger.info("StandardScaler treinado nos dados de treino")

            # Aplica normalização
            self.X_train = self.scaler.transform(X_train)
            self.X_test = self.scaler.transform(X_test)
            self.y_train = y_train
            self.y_test = y_test

            # Validação será o conjunto de teste por padrão
            # (pode ser modificado com get_validation_data se precisar split separado)
            self.X_val = self.X_test
            self.y_val = self.y_test

            self._data_loaded = True
            logger.info(f"Dataset '{self.dataset_source}' carregado e pré-processado com sucesso!")

        except Exception as e:
            logger.error(f"Erro ao carregar dataset '{self.dataset_source}': {e}", exc_info=True)
            raise

    def hf_to_xy(self, hf_dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Converte dataset HuggingFace para arrays numpy (X, y).

        Args:
            hf_dataset: Dataset do HuggingFace.

        Returns:
            Tupla (X, y) onde:
                - X: Array numpy com features de shape (n_samples, n_features)
                - y: Array numpy com labels de shape (n_samples,)
        """
        # Remove a coluna 'label' para obter features
        # Assume que a primeira coluna é o label e as demais são features
        # ou que existe uma coluna chamada 'label'

        # Converte para pandas para facilitar
        df = hf_dataset.to_pandas()

        # Identifica coluna de label
        if "label" in df.columns:
            y = df["label"].values
            X = df.drop("label", axis=1).values
        elif "target" in df.columns:
            y = df["target"].values
            X = df.drop("target", axis=1).values
        else:
            # Assume que a primeira coluna é o label
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values

        return X.astype(np.float32), y.astype(np.int64)

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna os dados de treinamento normalizados.

        Returns:
            Tupla (X_train, y_train) com dados de treino.

        Raises:
            RuntimeError: Se os dados não foram carregados.
        """
        if not self._data_loaded:
            self.load_data()

        return self.X_train, self.y_train

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna os dados de teste normalizados.

        Returns:
            Tupla (X_test, y_test) com dados de teste.

        Raises:
            RuntimeError: Se os dados não foram carregados.
        """
        if not self._data_loaded:
            self.load_data()

        return self.X_test, self.y_test

    def get_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna os dados de validação.

        Por padrão, retorna os dados de teste. Pode ser modificado
        para criar um split separado se necessário.

        Returns:
            Tupla (X_val, y_val) com dados de validação.

        Raises:
            RuntimeError: Se os dados não foram carregados.
        """
        if not self._data_loaded:
            self.load_data()

        # Por padrão, validação = teste
        # Se precisar de split separado, pode implementar aqui
        return self.X_val, self.y_val

    def get_partitions(self, num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Particiona os dados de treino entre clientes de forma IID.

        Divide os dados de treino igualmente entre os clientes usando
        particionamento IID (Independent and Identically Distributed).

        Args:
            num_clients: Número de clientes para particionar.

        Returns:
            Lista de tuplas (X_client, y_client) para cada cliente.

        Raises:
            RuntimeError: Se os dados não foram carregados.
            ValueError: Se num_clients for inválido.
        """
        if not self._data_loaded:
            self.load_data()

        if num_clients <= 0:
            raise ValueError(f"num_clients deve ser positivo, recebido: {num_clients}")

        logger.info(
            f"Particionando dados entre {num_clients} clientes (IID)..."
        )

        # Embaralha os dados antes de particionar (mantendo X e y sincronizados)
        indices = np.arange(len(self.X_train))
        np.random.seed(self.config.seed)
        np.random.shuffle(indices)

        X_shuffled = self.X_train[indices]
        y_shuffled = self.y_train[indices]

        # Divide em partições IID usando array_split
        X_partitions = np.array_split(X_shuffled, num_clients)
        y_partitions = np.array_split(y_shuffled, num_clients)

        # Combina em lista de tuplas
        partitions = list(zip(X_partitions, y_partitions))

        # Log de informações das partições
        for i, (X_part, y_part) in enumerate(partitions):
            logger.debug(
                f"Cliente {i}: {len(X_part)} amostras, "
                f"distribuição de classes: {dict(zip(*np.unique(y_part, return_counts=True)))}"
            )

        logger.info(
            f"Particionamento concluído: {num_clients} partições criadas "
            f"(tamanho médio: {len(self.X_train) // num_clients} amostras/cliente)"
        )

        return partitions

    def get_scaler(self) -> StandardScaler:
        """Retorna o StandardScaler treinado nos dados de treino.

        Returns:
            StandardScaler já fitted.

        Raises:
            RuntimeError: Se os dados não foram carregados.
        """
        if not self._data_loaded:
            self.load_data()

        return self.scaler

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """Aplica normalização StandardScaler aos dados.

        Args:
            X: Array numpy de shape (n_samples, n_features) com features brutas.

        Returns:
            Array numpy com features normalizadas.

        Raises:
            RuntimeError: Se os dados não foram carregados (scaler não treinado).
        """
        if not self._data_loaded:
            self.load_data()

        return self.scaler.transform(X)

    def get_num_features(self) -> int:
        """Retorna o número de features do dataset.

        Returns:
            Número de features do dataset carregado.

        Raises:
            RuntimeError: Se os dados não foram carregados.
        """
        if not self._data_loaded:
            self.load_data()

        return self._num_features
