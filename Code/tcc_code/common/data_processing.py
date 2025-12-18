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
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBALANCE_AVAILABLE = True
except ImportError:
    IMBALANCE_AVAILABLE = False
    log(WARNING, "imblearn não disponível. Balanceamento de classes desabilitado.")


class DataProcessor:
    """Gerencia carregamento e preparação do dataset de veículos para Federated Learning"""

    def __init__(self, num_clients: int, seed: int = 42,
                 train_csv_path: str = None, validation_csv_path: str = None,
                 use_all_data: bool = True, balance_strategy: str = None,
                 vehicles_per_client: int = None, stratified: bool = True):
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
            vehicles_per_client: Número fixo de veículos por cliente (sobrescreve use_all_data)
            stratified: Se True, garante representação proporcional de classes em cada cliente
        """
        if train_csv_path is None or validation_csv_path is None:
            raise ValueError("train_csv_path e validation_csv_path são obrigatórios!")

        self.num_clients = num_clients
        self.seed = seed
        self.use_all_data = use_all_data
        self.balance_strategy = balance_strategy
        self.vehicles_per_client = vehicles_per_client
        self.stratified = stratified
        self.scaler = StandardScaler()

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

        if self.vehicles_per_client:
            vehicles_per_client = self.vehicles_per_client
            total_vehicles_needed = self.vehicles_per_client * self.num_clients
            if total_vehicles_needed > num_vehicles:
                log(WARNING, f"vehicles_per_client={self.vehicles_per_client} × {self.num_clients} clientes = {total_vehicles_needed} > {num_vehicles} disponíveis")
                log(WARNING, f"Usando todos os {num_vehicles} veículos disponíveis")
                vehicles_per_client = num_vehicles // self.num_clients
        else:
            vehicles_per_client = num_vehicles // self.num_clients

        log(INFO, f"Particionando por veículo: {vehicles_per_client} veículos por cliente")

        if self.stratified and self.vehicles_per_client:
            selected_vehicles = self._stratified_vehicle_sampling(
                X_train_all, y_train_all, vehicle_ids_train, unique_vehicles,
                vehicles_per_client * self.num_clients
            )
            np.random.shuffle(selected_vehicles)
        else:
            if self.vehicles_per_client:
                selected_vehicles = np.random.choice(
                    unique_vehicles,
                    size=min(vehicles_per_client * self.num_clients, num_vehicles),
                    replace=False
                )
            else:
                selected_vehicles = unique_vehicles
            np.random.shuffle(selected_vehicles)

        self.partitions_X = []
        self.partitions_y = []
        client_vehicles = []

        log(INFO, "\n[PARTICIONAMENTO] Distribuindo dados por cliente...")

        for client_id in range(self.num_clients):
            start_idx = client_id * vehicles_per_client
            end_idx = min(start_idx + vehicles_per_client, len(selected_vehicles))

            client_vehicle_ids = selected_vehicles[start_idx:end_idx]
            client_vehicles.append(client_vehicle_ids)

            mask = np.isin(vehicle_ids_train, client_vehicle_ids)
            client_X = X_train_all[mask]
            client_y = y_train_all[mask]

            unique_before, counts_before = np.unique(client_y, return_counts=True)
            log(INFO, f"\n  Cliente {client_id} - ANTES do balanceamento:")
            log(INFO, f"    Veículos: {len(client_vehicle_ids)} | Amostras: {len(client_y)}")
            for cls, count in zip(unique_before, counts_before):
                log(INFO, f"    Classe {cls}: {count} ({count/len(client_y)*100:.1f}%)")

            if self.balance_strategy:
                client_X, client_y = self._balance_classes(client_X, client_y)
                log(INFO, f"  Cliente {client_id} - APÓS balanceamento: {len(client_y)} amostras")
            else:
                log(INFO, f"  Cliente {client_id} - Balanceamento NÃO aplicado")

            self.partitions_X.append(client_X)
            self.partitions_y.append(client_y)

        self.X_test_all = X_test_all
        self.y_test_all = y_test_all
        self.samples_per_client = int(np.mean([len(p) for p in self.partitions_X]))

        log(INFO, f"[OK] Dataset particionado por veículo com sucesso!")
        log(INFO, f"  - Média de amostras por cliente: {self.samples_per_client}")
        log(INFO, f"  - Total distribuído: {sum(len(p) for p in self.partitions_X)}")
        log(INFO, f"  - Validação centralizada: {len(self.X_test_all)} amostras")

        log(INFO, "\nDistribuição FINAL por cliente (APÓS balanceamento):")
        for i, (y_part, veh_ids) in enumerate(zip(self.partitions_y, client_vehicles)):
            unique, counts = np.unique(y_part, return_counts=True)
            dist = dict(zip(unique, counts))
            percentages = {k: f"{v/len(y_part)*100:.1f}%" for k, v in dist.items()}
            log(INFO, f"  Cliente {i}: {len(y_part)} amostras | {len(veh_ids)} veículos | Classes: {percentages}")
            log(INFO, f"    Veículos: {sorted(veh_ids.tolist())[:10]}{'...' if len(veh_ids) > 10 else ''}")

            if len(counts) == 3:
                max_diff = (max(counts) - min(counts)) / len(y_part) * 100
                if max_diff < 5:
                    log(INFO, f"    [OK] Balanceado! Diferença máxima: {max_diff:.1f}%")
                else:
                    log(WARNING, f"    [AVISO] Desbalanceado! Diferença máxima: {max_diff:.1f}%")

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

    def _stratified_vehicle_sampling(self, X, y, vehicle_ids, unique_vehicles, n_vehicles_needed):
        """
        Seleciona veículos garantindo representação proporcional das classes

        Args:
            X: Features
            y: Labels
            vehicle_ids: IDs dos veículos
            unique_vehicles: Lista de veículos únicos
            n_vehicles_needed: Número total de veículos a selecionar

        Returns:
            Array com IDs dos veículos selecionados (stratified)
        """
        vehicle_class_dist = {}
        for veh_id in unique_vehicles:
            mask = vehicle_ids == veh_id
            veh_classes = y[mask]
            unique_cls, counts = np.unique(veh_classes, return_counts=True)
            majority_class = unique_cls[np.argmax(counts)]
            vehicle_class_dist[veh_id] = majority_class

        vehicles_by_class = {0: [], 1: [], 2: []}
        for veh_id, cls in vehicle_class_dist.items():
            vehicles_by_class[cls].append(veh_id)

        log(INFO, f"[STRATIFIED] Veículos por classe majoritária:")
        for cls in [0, 1, 2]:
            log(INFO, f"  Classe {cls}: {len(vehicles_by_class[cls])} veículos")

        total_vehicles = len(unique_vehicles)
        vehicles_per_class = {}
        for cls in [0, 1, 2]:
            proportion = len(vehicles_by_class[cls]) / total_vehicles
            n_select = int(n_vehicles_needed * proportion)
            vehicles_per_class[cls] = min(n_select, len(vehicles_by_class[cls]))

        total_selected = sum(vehicles_per_class.values())
        if total_selected < n_vehicles_needed:
            cls_max = max(vehicles_by_class.keys(), key=lambda c: len(vehicles_by_class[c]))
            vehicles_per_class[cls_max] += (n_vehicles_needed - total_selected)

        log(INFO, f"[STRATIFIED] Selecionando veículos (stratified):")
        selected_vehicles = []
        for cls in [0, 1, 2]:
            n_select = vehicles_per_class[cls]
            selected = np.random.choice(vehicles_by_class[cls], size=n_select, replace=False)
            selected_vehicles.extend(selected)
            log(INFO, f"  Classe {cls}: {n_select} veículos selecionados")

        return np.array(selected_vehicles)

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

            elif self.balance_strategy == 'combined':
                min_samples = min(counts)
                k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
                sampler = SMOTETomek(
                    smote=SMOTE(random_state=self.seed, k_neighbors=k_neighbors),
                    random_state=self.seed
                )
                log(INFO, f"\n[BALANCEAMENTO] Estratégia: SMOTE + Tomek Links (k_neighbors={k_neighbors})")
                log(INFO, f"  Combina oversampling com limpeza de fronteira de decisão")

            elif self.balance_strategy == 'smoteenn':
                min_samples = min(counts)
                k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
                sampler = SMOTEENN(
                    smote=SMOTE(random_state=self.seed, k_neighbors=k_neighbors),
                    random_state=self.seed
                )
                log(INFO, f"\n[BALANCEAMENTO] Estratégia: SMOTE + ENN (k_neighbors={k_neighbors})")
                log(INFO, f"  Combina oversampling com limpeza agressiva de ruído")

            else:
                log(WARNING, f"[BALANCEAMENTO] Estratégia '{self.balance_strategy}' desconhecida.")
                log(WARNING, f"  Opções: oversample, smote, undersample, combined, smoteenn, weights")
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
