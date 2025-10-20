"""
Módulo comum para cálculo de métricas e logging
Compartilhado por todos os algoritmos (XGBoost, LightGBM, CatBoost)
"""

import json
import time
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from flwr.common.logger import log
from logging import INFO


def calculate_comprehensive_metrics(y_true, y_pred_proba, threshold=0.5) -> Dict:
    """
    Calcula métricas abrangentes para classificação binária

    Args:
        y_true: labels verdadeiros
        y_pred_proba: probabilidades preditas
        threshold: limiar para conversão de probabilidade em classe

    Returns:
        Dict com todas as métricas
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calcular métricas básicas
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = 0.5

    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    # Matriz de confusão
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Acurácia
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Especificidade (Taxa de Verdadeiros Negativos)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'specificity': float(specificity),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }

    return metrics


def print_metrics_summary(metrics: Dict, client_id: Optional[int] = None,
                         round_num: Optional[int] = None):
    """
    Imprime um resumo organizado das métricas

    Args:
        metrics: Dicionário com métricas calculadas
        client_id: ID do cliente (opcional)
        round_num: Número do round (opcional)
    """
    prefix = f"[Client {client_id}]" if client_id is not None else "[Server]"
    if round_num is not None:
        prefix += f" Round {round_num}"

    print(f"\n{prefix} Métricas de Performance:")
    print(f"  Acurácia:    {metrics['accuracy']:.4f}")
    print(f"  Precisão:    {metrics['precision']:.4f}")
    print(f"  Revocação:   {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  AUC:         {metrics['auc']:.4f}")

    if 'specificity' in metrics:
        print(f"  Especific.:  {metrics['specificity']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"  Matriz de Confusão:")
    print(f"    TN: {cm['tn']:4d} | FP: {cm['fp']:4d}")
    print(f"    FN: {cm['fn']:4d} | TP: {cm['tp']:4d}")


def save_metrics_to_file(metrics_history, filename="federated_learning_metrics.json"):
    """
    Salva histórico de métricas em arquivo JSON

    Args:
        metrics_history: Dicionário com histórico de métricas
        filename: Nome do arquivo para salvar
    """
    try:
        with open(filename, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f"\nMétricas salvas em: {filename}")
    except Exception as e:
        print(f"Erro ao salvar métricas: {e}")


def print_final_analysis(strategy_name: str, metrics_history=None):
    """
    Imprime análise final dos resultados

    Args:
        strategy_name: Nome da estratégia executada
        metrics_history: Histórico de métricas (opcional)
    """
    print(f"\n{'='*60}")
    print(f"ANÁLISE FINAL - {strategy_name.upper()}")
    print(f"{'='*60}")

    if metrics_history:
        print("Resumo das métricas por rodada:")
        for round_num, metrics in metrics_history.items():
            print(f"\nRodada {round_num}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")

    print(f"{'='*60}\n")


def evaluate_metrics_aggregation(eval_metrics):
    """
    Agrega métricas de avaliação de forma robusta

    Args:
        eval_metrics: Lista de tuplas (num_examples, metrics)

    Returns:
        Dicionário com métricas agregadas
    """
    if not eval_metrics:
        return {
            "auc": 0.5,
            "accuracy": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    total_num = sum([num for num, _ in eval_metrics])
    if total_num == 0:
        return {
            "auc": 0.5,
            "accuracy": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    # Agregação robusta com valores padrão
    metric_sums = {
        "auc": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
    }

    for num, metrics in eval_metrics:
        for metric_name in metric_sums.keys():
            metric_val = metrics.get(metric_name, 0.5 if metric_name in ['auc', 'accuracy'] else 0.0)
            metric_sums[metric_name] += metric_val * num

    # Calcular médias ponderadas
    metrics_aggregated = {}
    for metric_name, total_sum in metric_sums.items():
        metrics_aggregated[metric_name] = total_sum / total_num

    log(INFO, f"Métricas agregadas:")
    for metric_name, value in metrics_aggregated.items():
        log(INFO, f"  {metric_name}: {value:.4f}")

    return metrics_aggregated


class ExperimentLogger:
    """Gerencia logging de experimentos de Federated Learning"""

    def __init__(self, algorithm_name: str, num_clients: int, num_rounds: int):
        """
        Args:
            algorithm_name: Nome do algoritmo (xgboost, lightgbm, catboost)
            num_clients: Número de clientes
            num_rounds: Número de rodadas
        """
        self.algorithm_name = algorithm_name
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.start_time = None
        self.metrics_history = {}

    def start_experiment(self):
        """Inicia experimento e marca tempo de início"""
        self.start_time = time.time()
        print(f"\n{'='*80}")
        print(f"INICIANDO EXPERIMENTO: {self.algorithm_name.upper()}")
        print(f"{'='*80}")
        print(f"Configuração:")
        print(f"  - Algoritmo: {self.algorithm_name}")
        print(f"  - Número de clientes: {self.num_clients}")
        print(f"  - Rodadas: {self.num_rounds}")
        print(f"{'='*80}\n")

    def log_round_metrics(self, round_num: int, metrics: Dict):
        """Loga métricas de uma rodada"""
        self.metrics_history[round_num] = metrics

    def end_experiment(self, save_file: bool = True):
        """Finaliza experimento e salva métricas"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        print(f"\n{'='*80}")
        print(f"EXPERIMENTO CONCLUÍDO: {self.algorithm_name.upper()}")
        print(f"Tempo total: {elapsed_time:.2f} segundos")
        print(f"{'='*80}\n")

        if save_file:
            filename = f"metrics_{self.algorithm_name}_{int(time.time())}.json"
            save_metrics_to_file(self.metrics_history, filename)

        return self.metrics_history
