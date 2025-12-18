"""
Módulo de técnicas avançadas para Federated Learning com classes desbalanceadas

Implementa:
1. Class/Sample weights para treino local
2. Cálculo de diversidade de classes (entropia)
3. Agregação ponderada por diversidade
4. Cycling ordenado por entropia
5. Curriculum learning federado
"""

import numpy as np
from typing import Dict, List
from scipy.stats import entropy
from sklearn.utils.class_weight import compute_class_weight


class ClassBalancingHelper:
    """Helper para balanceamento de classes no nível local e federado"""

    def __init__(self, num_classes: int = 3, max_weight: float = 10.0):
        """
        Args:
            num_classes: Número de classes no problema
            max_weight: Peso máximo permitido (evita instabilidade)
        """
        self.num_classes = num_classes
        self.max_weight = max_weight

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calcula class weights balanceados para um cliente

        Args:
            y: Labels do cliente

        Returns:
            Dict com {classe: peso}
        """
        classes_present = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes_present, y=y)

        weights = np.clip(weights, 0.1, self.max_weight)

        return dict(zip(classes_present, weights))

    def compute_sample_weights(self, y: np.ndarray, strategy: str = 'balanced') -> np.ndarray:
        """
        Calcula sample weights para cada amostra

        Args:
            y: Labels
            strategy: 'balanced' ou 'custom'

        Returns:
            Array de pesos por amostra
        """
        if strategy == 'balanced':
            class_weights_dict = self.compute_class_weights(y)

            full_class_weights = {}
            for cls in range(self.num_classes):
                if cls in class_weights_dict:
                    full_class_weights[cls] = class_weights_dict[cls]
                else:
                    full_class_weights[cls] = np.mean(list(class_weights_dict.values()))

            sample_weights = np.array([full_class_weights[int(label)] for label in y])
            return sample_weights
        else:
            return np.ones(len(y))

    def get_scale_pos_weight(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calcula scale_pos_weight para XGBoost/LightGBM

        Para multi-classe, retorna proporção de cada classe vs majoritária

        Args:
            y: Labels

        Returns:
            Dict com scale_pos_weight por classe
        """
        unique, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)

        scale_weights = {}
        for cls, count in zip(unique, counts):
            scale_weights[int(cls)] = max_count / count if count > 0 else 1.0

        scale_weights = {k: min(v, self.max_weight) for k, v in scale_weights.items()}

        return scale_weights


class DiversityMetrics:
    """Métricas de diversidade de classes para ordenação de clientes"""

    @staticmethod
    def calculate_entropy(y: np.ndarray) -> float:
        """
        Calcula entropia de Shannon da distribuição de classes

        Entropia alta = distribuição uniforme (boa diversidade)
        Entropia baixa = concentrada em poucas classes (mono-classe)

        Args:
            y: Labels do cliente

        Returns:
            Entropia (0 = mono-classe, log(n_classes) = uniforme)
        """
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return entropy(probabilities, base=2)

    @staticmethod
    def calculate_class_balance_score(y: np.ndarray, num_classes: int = 3) -> float:
        """
        Calcula score de balanceamento (0 = desbalanceado, 1 = perfeito)

        Args:
            y: Labels
            num_classes: Número total de classes

        Returns:
            Score entre 0 e 1
        """
        unique, counts = np.unique(y, return_counts=True)

        ideal_proportion = 1.0 / num_classes

        actual_proportions = np.zeros(num_classes)
        for cls, count in zip(unique, counts):
            actual_proportions[int(cls)] = count / len(y)

        mae = np.mean(np.abs(actual_proportions - ideal_proportion))
        score = 1.0 - (mae * num_classes)

        return max(0.0, score)

    @staticmethod
    def calculate_gini_impurity(y: np.ndarray) -> float:
        """
        Calcula impureza de Gini (usado em árvores de decisão)

        Gini = 0: mono-classe (pura)
        Gini alto: distribuição uniforme

        Args:
            y: Labels

        Returns:
            Impureza de Gini
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini


class FederatedAggregationWeights:
    """Calcula pesos de agregação considerando diversidade de classes"""

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Balanceamento entre tamanho e diversidade
                   0.0 = apenas tamanho
                   1.0 = apenas diversidade
                   0.5 = meio a meio (padrão)

        Raises:
            ValueError: Se alpha não estiver entre 0.0 e 1.0
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha deve estar entre 0.0 e 1.0, recebido: {alpha}")
        self.alpha = alpha

    def compute_aggregation_weights(
        self,
        client_data_sizes: List[int],
        client_labels: List[np.ndarray]
    ) -> np.ndarray:
        """
        Calcula pesos de agregação considerando tamanho E diversidade

        Args:
            client_data_sizes: Lista com número de amostras por cliente
            client_labels: Lista com labels de cada cliente

        Returns:
            Array de pesos normalizados (soma = 1.0)
        """
        size_weights = np.array(client_data_sizes, dtype=float)
        size_weights = size_weights / np.sum(size_weights)

        diversity_scores = []
        normalized_entropies = []

        for labels in client_labels:
            entropy_score = DiversityMetrics.calculate_entropy(labels)
            diversity_scores.append(entropy_score)

            num_classes_present = len(np.unique(labels))
            max_entropy_client = np.log2(num_classes_present) if num_classes_present > 1 else 1.0

            normalized_entropy = entropy_score / max_entropy_client if max_entropy_client > 0 else 0.5
            normalized_entropies.append(normalized_entropy)

        diversity_weights = np.array(normalized_entropies, dtype=float)

        diversity_weights = np.maximum(diversity_weights, 0.1)
        diversity_weights = diversity_weights / np.sum(diversity_weights)

        final_weights = (1 - self.alpha) * size_weights + self.alpha * diversity_weights
        final_weights = final_weights / np.sum(final_weights)

        return final_weights

    def penalize_mono_class_clients(
        self,
        base_weights: np.ndarray,
        client_labels: List[np.ndarray],
        penalty_factor: float = 0.5
    ) -> np.ndarray:
        """
        Penaliza clientes mono-classe ou altamente enviesados

        Args:
            base_weights: Pesos base de agregação
            client_labels: Labels de cada cliente
            penalty_factor: Fator de penalização (0-1)

        Returns:
            Pesos ajustados e normalizados
        """
        adjusted_weights = base_weights.copy()

        for i, labels in enumerate(client_labels):
            balance_score = DiversityMetrics.calculate_class_balance_score(labels)

            if balance_score < 0.5:
                penalty = 1.0 - (penalty_factor * (1.0 - balance_score))
                adjusted_weights[i] *= penalty

        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

        return adjusted_weights


class CurriculumLearning:
    """Curriculum learning federado para classes minoritárias"""

    def __init__(self, num_rounds: int, warmup_rounds: int = 5):
        """
        Args:
            num_rounds: Total de rodadas federadas
            warmup_rounds: Rodadas iniciais com pesos mais suaves

        Raises:
            ValueError: Se parâmetros forem inválidos
        """
        if num_rounds <= 0:
            raise ValueError(f"num_rounds deve ser > 0, recebido: {num_rounds}")
        if warmup_rounds < 0:
            raise ValueError(f"warmup_rounds deve ser >= 0, recebido: {warmup_rounds}")
        if warmup_rounds >= num_rounds:
            raise ValueError(
                f"warmup_rounds ({warmup_rounds}) deve ser < num_rounds ({num_rounds})"
            )

        self.num_rounds = num_rounds
        self.warmup_rounds = warmup_rounds

    def get_round_weights_multiplier(self, current_round: int) -> float:
        """
        Retorna multiplicador de pesos para a rodada atual

        Curriculum: Começa com pesos suaves, aumenta progressivamente

        Args:
            current_round: Rodada atual (1-indexed)

        Returns:
            Multiplicador (0.5 a 1.5)
        """
        if self.warmup_rounds == 0:
            progress = current_round / self.num_rounds
            return 1.0 + 0.5 * progress

        if current_round <= self.warmup_rounds:
            progress = current_round / self.warmup_rounds
            return 0.5 + 0.5 * progress
        else:
            remaining_rounds = self.num_rounds - self.warmup_rounds
            progress = (current_round - self.warmup_rounds) / remaining_rounds
            return 1.0 + 0.5 * progress

    def adjust_class_weights_by_round(
        self,
        base_class_weights: Dict[int, float],
        current_round: int
    ) -> Dict[int, float]:
        """
        Ajusta class weights de acordo com o curriculum

        Args:
            base_class_weights: Pesos base balanceados
            current_round: Rodada atual

        Returns:
            Pesos ajustados pelo curriculum
        """
        multiplier = self.get_round_weights_multiplier(current_round)

        adjusted_weights = {}
        for cls, weight in base_class_weights.items():
            if weight > 1.0:
                adjusted_weights[cls] = 1.0 + (weight - 1.0) * multiplier
            else:
                adjusted_weights[cls] = weight

        return adjusted_weights


class ClientCyclingStrategy:
    """Estratégia de cycling de clientes ordenado por diversidade"""

    @staticmethod
    def order_clients_by_entropy(client_labels: List[np.ndarray]) -> np.ndarray:
        """
        Ordena clientes por entropia (diversidade de classes)

        Ordem: maior entropia (mais diverso) primeiro

        Args:
            client_labels: Lista de labels por cliente

        Returns:
            Array com índices ordenados
        """
        entropies = []
        for labels in client_labels:
            entropy_val = DiversityMetrics.calculate_entropy(labels)
            entropies.append(entropy_val)

        sorted_indices = np.argsort(entropies)[::-1]

        return sorted_indices

    @staticmethod
    def create_cyclic_schedule(
        num_clients: int,
        num_rounds: int,
        client_labels: List[np.ndarray],
        strategy: str = 'entropy'
    ) -> List[int]:
        """
        Cria schedule de cycling de clientes

        Args:
            num_clients: Número total de clientes
            num_rounds: Número de rodadas
            client_labels: Labels por cliente
            strategy: 'entropy' (por diversidade) ou 'sequential' (sequencial)

        Returns:
            Lista com ID do cliente para cada rodada
        """
        if strategy == 'entropy':
            order = ClientCyclingStrategy.order_clients_by_entropy(client_labels)
        else:
            order = np.arange(num_clients)

        schedule = []
        for round_idx in range(num_rounds):
            client_id = order[round_idx % num_clients]
            schedule.append(int(client_id))

        return schedule


def get_stable_tree_params(algorithm: str) -> Dict:
    """
    Retorna hiperparâmetros para árvores mais estáveis

    Características:
    - Árvores rasas (max_depth menor)
    - Learning rate menor
    - Maior número de iterações
    - Regularização mais forte

    Args:
        algorithm: 'xgboost', 'lightgbm', ou 'catboost'

    Returns:
        Dict com hiperparâmetros otimizados
    """
    if algorithm == 'xgboost':
        return {
            'max_depth': 4,  # Árvores rasas (antes: 6)
            'eta': 0.05,  # Learning rate menor (antes: 0.3)
            'min_child_weight': 3,  # Regularização
            'subsample': 0.8,  # Bagging
            'colsample_bytree': 0.8,  # Feature bagging
            'gamma': 0.1,  # Poda de árvores
            'lambda': 1.5,  # L2 regularization
            'alpha': 0.5,  # L1 regularization
        }

    elif algorithm == 'lightgbm':
        return {
            'max_depth': 4,
            'learning_rate': 0.05,
            'min_child_samples': 20,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 1.5,
            'min_split_gain': 0.01,
        }

    elif algorithm == 'catboost':
        return {
            'depth': 4,
            'learning_rate': 0.05,
            'min_data_in_leaf': 20,
            'subsample': 0.8,
            'rsm': 0.8,  # colsample_bytree
            'l2_leaf_reg': 3.0,
            'random_strength': 0.5,
        }

    else:
        raise ValueError(
            f"Algoritmo '{algorithm}' não suportado. "
            f"Opções válidas: 'xgboost', 'lightgbm', 'catboost'"
        )
