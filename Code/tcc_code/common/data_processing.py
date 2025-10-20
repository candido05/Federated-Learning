"""
Módulo comum para processamento de dataset
Compartilhado por todos os algoritmos (XGBoost, LightGBM, CatBoost)
"""

import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flwr.common.logger import log
from logging import INFO


class DataProcessor:
    """Gerencia carregamento e preparação do dataset HIGGS para Federated Learning"""

    def __init__(self, num_clients: int, sample_per_client: int, seed: int = 42):
        """
        Args:
            num_clients: Número de clientes para particionar os dados
            sample_per_client: Quantidade de amostras por cliente
            seed: Seed para reprodutibilidade
        """
        self.num_clients = num_clients
        self.sample_per_client = sample_per_client
        self.seed = seed
        self.scaler = StandardScaler()

        # Dados particionados
        self.partitions_X = None
        self.partitions_y = None
        self.X_test_all = None
        self.y_test_all = None

    def load_and_prepare_data(self):
        """Carrega dataset HIGGS e prepara para FL"""
        log(INFO, "Carregando dataset HIGGS...")
        ds = load_dataset("jxie/higgs")

        max_train = self.sample_per_client * self.num_clients
        max_test = self.sample_per_client

        train_all = ds["train"].select(range(min(len(ds["train"]), max_train)))
        test_all = ds["test"].select(range(min(len(ds["test"]), max_test)))

        X_train_all, y_train_all = self._hf_to_xy(train_all)
        X_test_all, y_test_all = self._hf_to_xy(test_all)

        # Normalização
        self.scaler.fit(X_train_all)
        X_train_all = self.scaler.transform(X_train_all)
        X_test_all = self.scaler.transform(X_test_all)

        # Particionamento IID
        self.partitions_X = np.array_split(X_train_all, self.num_clients)
        self.partitions_y = np.array_split(y_train_all, self.num_clients)
        self.X_test_all = X_test_all
        self.y_test_all = y_test_all

        log(INFO, f"Dataset particionado em {self.num_clients} clientes (~{len(self.partitions_X[0])} amostras por cliente).")

        return self.partitions_X, self.partitions_y, self.X_test_all, self.y_test_all

    @staticmethod
    def _hf_to_xy(hf_dataset):
        """Converte HuggingFace Dataset para arrays X, y"""
        if "inputs" in hf_dataset.column_names:
            X = np.array(hf_dataset["inputs"], dtype=np.float32)
        else:
            feature_cols = [c for c in hf_dataset.column_names if c != "label"]
            X = np.vstack([[example[c] for c in feature_cols] for example in hf_dataset]).astype(np.float32)
        y = np.array(hf_dataset["label"]).astype(int)
        return X, y

    def get_client_data(self, partition_id: int, test_fraction: float = 0.2,
                       centralised_eval_client: bool = False):
        """
        Retorna dados de treino/validação para um cliente específico

        Args:
            partition_id: ID da partição (cliente)
            test_fraction: Fração dos dados para validação
            centralised_eval_client: Se True, usa dataset de teste global para validação

        Returns:
            train_X, train_y, valid_X, valid_y
        """
        X_part = self.partitions_X[partition_id]
        y_part = self.partitions_y[partition_id]

        if centralised_eval_client:
            train_X = X_part
            train_y = y_part
            valid_X = self.X_test_all
            valid_y = self.y_test_all
        else:
            train_X, valid_X, train_y, valid_y = train_test_split(
                X_part, y_part, test_size=test_fraction, random_state=self.seed
            )

        return train_X, train_y, valid_X, valid_y


def replace_keys(input_dict, match="-", target="_"):
    """
    Substitui caracteres em chaves de dicionário recursivamente
    Usado para converter configurações do Flower
    """
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
