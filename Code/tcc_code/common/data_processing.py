"""
Módulo comum para processamento de dataset de veículos
Compartilhado por todos os algoritmos (XGBoost, LightGBM, CatBoost)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flwr.common.logger import log
from logging import INFO, WARNING

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBALANCE_AVAILABLE = True
except ImportError:
    IMBALANCE_AVAILABLE = False
    log(WARNING, "imblearn não disponível. Balanceamento de classes desabilitado.")


class DataProcessor:
    """Gerencia carregamento e preparação do dataset de veículos para Federated Learning"""

    def __init__(self, num_clients: int, seed: int = 42,
                 train_csv_path: str = None, validation_csv_path: str = None,
                 use_all_data: bool = True, balance_strategy: str = None):
        """
        Args:
            num_clients: Número de clientes para particionar os dados
            seed: Seed para reprodutibilidade
            train_csv_path: Caminho para CSV de treino (OBRIGATÓRIO)
            validation_csv_path: Caminho para CSV de validação (OBRIGATÓRIO)
            use_all_data: Se True, usa TODOS os dados disponíveis distribuídos entre clientes
            balance_strategy: Estratégia de balanceamento de classes:
                - None: Sem balanceamento (padrão)
                - 'oversample': Oversampling aleatório da classe minoritária
                - 'smote': SMOTE (Synthetic Minority Over-sampling Technique)
                - 'undersample': Undersampling da classe majoritária
                - 'weights': Retorna pesos de classe (sem modificar dados)
        """
        if train_csv_path is None or validation_csv_path is None:
            raise ValueError("train_csv_path e validation_csv_path são obrigatórios!")

        self.num_clients = num_clients
        self.seed = seed
        self.use_all_data = use_all_data
        self.balance_strategy = balance_strategy
        self.scaler = StandardScaler()

        # Caminhos dos CSVs
        self.train_csv_path = train_csv_path
        self.validation_csv_path = validation_csv_path

        self.partitions_X = None
        self.partitions_y = None
        self.X_test_all = None
        self.y_test_all = None
        self.samples_per_client = None
        self.class_weights = None

    def load_and_prepare_data(self):
        """Carrega dataset de veículos e prepara para FL com particionamento por veículo"""
        log(INFO, f"Carregando dataset de CSV: {self.train_csv_path}")
        X_train_all, y_train_all, vehicle_ids_train = self._load_csv(self.train_csv_path)

        log(INFO, f"Carregando validação de CSV: {self.validation_csv_path}")
        X_test_all, y_test_all, vehicle_ids_test = self._load_csv(self.validation_csv_path)

        total_samples = X_train_all.shape[0]
        log(INFO, f"Dataset de treino: {total_samples} amostras, {X_train_all.shape[1]} features")
        log(INFO, f"Dataset de validação: {X_test_all.shape[0]} amostras")
        log(INFO, f"Classes no treino: {np.unique(y_train_all)}")
        log(INFO, f"Classes na validação: {np.unique(y_test_all)}")

        if vehicle_ids_train is None:
            log(WARNING, "Coluna 'vehicle_id' não encontrada! Usando particionamento IID tradicional.")
            return self._partition_iid(X_train_all, y_train_all, X_test_all, y_test_all)

        unique_vehicles = np.unique(vehicle_ids_train)
        num_vehicles = len(unique_vehicles)
        log(INFO, f"Número de veículos únicos detectados: {num_vehicles}")

        log(INFO, "Normalizando dados com StandardScaler...")
        self.scaler.fit(X_train_all)
        X_train_all = self.scaler.transform(X_train_all)
        X_test_all = self.scaler.transform(X_test_all)

        np.random.seed(self.seed)
        shuffled_vehicles = np.random.permutation(unique_vehicles)

        vehicles_per_client = num_vehicles // self.num_clients
        log(INFO, f"Particionando por veículo: {vehicles_per_client} veículos por cliente")

        self.partitions_X = []
        self.partitions_y = []
        client_vehicles = []

        for client_id in range(self.num_clients):
            start_idx = client_id * vehicles_per_client
            end_idx = start_idx + vehicles_per_client if client_id < self.num_clients - 1 else num_vehicles

            client_vehicle_ids = shuffled_vehicles[start_idx:end_idx]
            client_vehicles.append(client_vehicle_ids)

            mask = np.isin(vehicle_ids_train, client_vehicle_ids)
            client_X = X_train_all[mask]
            client_y = y_train_all[mask]

            self.partitions_X.append(client_X)
            self.partitions_y.append(client_y)

        self.X_test_all = X_test_all
        self.y_test_all = y_test_all
        self.samples_per_client = int(np.mean([len(p) for p in self.partitions_X]))

        log(INFO, f"[OK] Dataset particionado por veículo com sucesso!")
        log(INFO, f"  - Média de amostras por cliente: {self.samples_per_client}")
        log(INFO, f"  - Total distribuído: {sum(len(p) for p in self.partitions_X)}")
        log(INFO, f"  - Validação centralizada: {len(self.X_test_all)} amostras")

        log(INFO, "\nDistribuição por cliente:")
        for i, (y_part, veh_ids) in enumerate(zip(self.partitions_y, client_vehicles)):
            unique, counts = np.unique(y_part, return_counts=True)
            dist = dict(zip(unique, counts))
            percentages = {k: f"{v/len(y_part)*100:.1f}%" for k, v in dist.items()}
            log(INFO, f"  Cliente {i}: {len(y_part)} amostras | {len(veh_ids)} veículos | Classes: {percentages}")
            log(INFO, f"    Veículos: {sorted(veh_ids.tolist())[:10]}{'...' if len(veh_ids) > 10 else ''}")

        return self.partitions_X, self.partitions_y, self.X_test_all, self.y_test_all

    def _partition_iid(self, X_train_all, y_train_all, X_test_all, y_test_all):
        """Particionamento IID tradicional (fallback)"""
        from sklearn.utils import shuffle
        X_train_all, y_train_all = shuffle(X_train_all, y_train_all, random_state=self.seed)

        if self.balance_strategy:
            X_train_all, y_train_all = self._balance_classes(X_train_all, y_train_all)

        total_samples = X_train_all.shape[0]
        log(INFO, f"Particionando {total_samples} amostras entre {self.num_clients} clientes (IID)...")
        self.partitions_X = np.array_split(X_train_all, self.num_clients)
        self.partitions_y = np.array_split(y_train_all, self.num_clients)
        self.X_test_all = X_test_all
        self.y_test_all = y_test_all
        self.samples_per_client = len(self.partitions_X[0])

        log(INFO, f"[OK] Dataset particionado (IID) com sucesso!")
        log(INFO, f"  - Amostras por cliente: {self.samples_per_client}")
        log(INFO, f"  - Total distribuído: {sum(len(p) for p in self.partitions_X)}")

        return self.partitions_X, self.partitions_y, self.X_test_all, self.y_test_all

    @staticmethod
    def _load_csv(csv_path: str):
        """Carrega dataset de arquivo CSV retornando também vehicle_id"""
        df = pd.read_csv(csv_path)

        cols_to_drop = []
        vehicle_ids = None

        if "vehicle_id" in df.columns:
            vehicle_ids = df["vehicle_id"].values
            cols_to_drop.append("vehicle_id")

        if "label" in df.columns:
            y = df["label"].values.astype(int)
            cols_to_drop.append("label")
        else:
            log(WARNING, "Coluna 'label' não encontrada. Usando última coluna como label.")
            y = df.iloc[:, -1].values.astype(int)

        if cols_to_drop:
            X = df.drop(columns=cols_to_drop).values.astype(np.float32)
        else:
            X = df.iloc[:, :-1].values.astype(np.float32)

        return X, y, vehicle_ids

    def _balance_classes(self, X: np.ndarray, y: np.ndarray):
        """Aplica balanceamento de classes conforme estratégia configurada"""
        unique, counts = np.unique(y, return_counts=True)
        log(INFO, f"\n[BALANCEAMENTO] Distribuição original de classes:")
        for cls, count in zip(unique, counts):
            log(INFO, f"  Classe {cls}: {count} ({count/len(y)*100:.1f}%)")

        if self.balance_strategy == 'weights':
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            self.class_weights = dict(zip(classes, class_weights))

            log(INFO, f"\n[BALANCEAMENTO] Estratégia: Class Weights")
            log(INFO, f"  Pesos calculados: {self.class_weights}")
            log(INFO, f"  Dados NÃO modificados (usar scale_pos_weight no modelo)")

            return X, y

        if not IMBALANCE_AVAILABLE:
            log(WARNING, "[BALANCEAMENTO] imblearn não disponível! Instale com: pip install imbalanced-learn")
            log(WARNING, "  Usando dados sem balanceamento.")
            return X, y

        try:
            if self.balance_strategy == 'oversample':
                sampler = RandomOverSampler(random_state=self.seed)
                log(INFO, f"\n[BALANCEAMENTO] Estratégia: Random Oversampling")

            elif self.balance_strategy == 'smote':
                min_samples = min(counts)
                k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
                sampler = SMOTE(random_state=self.seed, k_neighbors=k_neighbors)
                log(INFO, f"\n[BALANCEAMENTO] Estratégia: SMOTE (k_neighbors={k_neighbors})")

            elif self.balance_strategy == 'undersample':
                sampler = RandomUnderSampler(random_state=self.seed)
                log(INFO, f"\n[BALANCEAMENTO] Estratégia: Random Undersampling")

            else:
                log(WARNING, f"[BALANCEAMENTO] Estratégia '{self.balance_strategy}' desconhecida. Opções: oversample, smote, undersample, weights")
                return X, y

            X_balanced, y_balanced = sampler.fit_resample(X, y)

            unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
            log(INFO, f"  Distribuição balanceada:")
            for cls, count in zip(unique_bal, counts_bal):
                log(INFO, f"    Classe {cls}: {count} ({count/len(y_balanced)*100:.1f}%)")

            log(INFO, f"  Total: {len(y)} -> {len(y_balanced)} amostras")

            return X_balanced, y_balanced

        except Exception as e:
            log(WARNING, f"[BALANCEAMENTO] Erro ao aplicar {self.balance_strategy}: {e}")
            log(WARNING, "  Usando dados sem balanceamento.")
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
