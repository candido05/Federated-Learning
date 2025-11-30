"""
Cálculo de métricas para classificação binária e multi-classe
Compartilhado por todos os algoritmos (XGBoost, LightGBM, CatBoost)
"""

import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from flwr.common.logger import log
from logging import WARNING


def calculate_comprehensive_metrics(y_true, y_pred_proba, threshold=0.5) -> Dict:
    """
    Calcula métricas abrangentes para classificação binária ou multi-classe

    Args:
        y_true: labels verdadeiros
        y_pred_proba: probabilidades preditas (pode ser 1D para binário ou 2D para multi-classe)
        threshold: limiar para conversão de probabilidade em classe (apenas para binário)

    Returns:
        Dict com todas as métricas
    """
    # Detectar número de classes
    num_classes = len(np.unique(y_true))
    is_binary = num_classes == 2

    # Predição de classes
    if is_binary:
        # Classificação binária
        if len(y_pred_proba.shape) == 1:
            y_pred = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        # Classificação multi-classe
        if len(y_pred_proba.shape) == 1:
            # Probabilidades fornecidas como 1D, usar threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
        else:
            # Probabilidades fornecidas como 2D (N, C)
            y_pred = np.argmax(y_pred_proba, axis=1)

    # Calcular métricas básicas
    accuracy = accuracy_score(y_true, y_pred)

    # AUC (multi-classe ou binária)
    try:
        if is_binary:
            if len(y_pred_proba.shape) == 1:
                auc = roc_auc_score(y_true, y_pred_proba)
            else:
                auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            # Multi-classe: AUC one-vs-rest macro average
            auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    except Exception as e:
        log(WARNING, f"Erro ao calcular AUC: {e}")
        auc = 0.5

    # Precision, Recall, F1 (weighted average para multi-classe)
    avg_type = 'binary' if is_binary else 'weighted'
    precision = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg_type, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg_type, zero_division=0)

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'num_classes': int(num_classes),
        'confusion_matrix': cm.tolist()  # Lista para JSON serialization
    }

    # Para binário, adicionar especificidade
    if is_binary and cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['specificity'] = float(specificity)

    return metrics


def print_metrics_summary(metrics: Dict, client_id: int = None, round_num: int = None):
    """
    Imprime um resumo organizado das métricas (binário ou multi-classe)

    Args:
        metrics: Dicionário com métricas calculadas
        client_id: ID do cliente (opcional)
        round_num: Número do round (opcional)
    """
    prefix = f"[Client {client_id}]" if client_id is not None else "[Server]"
    if round_num is not None:
        prefix += f" Round {round_num}"

    num_classes = metrics.get('num_classes', 2)
    is_binary = num_classes == 2

    print(f"\n{prefix} Métricas de Performance:")
    print(f"  Acurácia:    {metrics['accuracy']:.4f}")
    print(f"  Precisão:    {metrics['precision']:.4f} ({'binary' if is_binary else 'weighted avg'})")
    print(f"  Revocação:   {metrics['recall']:.4f} ({'binary' if is_binary else 'weighted avg'})")
    print(f"  F1-Score:    {metrics['f1_score']:.4f} ({'binary' if is_binary else 'weighted avg'})")
    print(f"  AUC:         {metrics['auc']:.4f} ({'binary' if is_binary else 'macro avg'})")

    if 'specificity' in metrics:
        print(f"  Especific.:  {metrics['specificity']:.4f}")

    # Matriz de confusão
    cm = metrics['confusion_matrix']
    if isinstance(cm, dict):
        # Formato antigo (binário com dicionário)
        print(f"  Matriz de Confusão:")
        print(f"    TN: {cm['tn']:4d} | FP: {cm['fp']:4d}")
        print(f"    FN: {cm['fn']:4d} | TP: {cm['tp']:4d}")
    elif isinstance(cm, list):
        # Formato novo (lista/matriz)
        cm_array = np.array(cm)
        print(f"  Matriz de Confusão ({num_classes} classes):")
        for i, row in enumerate(cm_array):
            print(f"    Classe {i}: {row}")


def evaluate_metrics_aggregation(eval_metrics):
    """
    Agrega métricas de avaliação de forma robusta

    Args:
        eval_metrics: Lista de tuplas (num_examples, metrics)

    Returns:
        Dicionário com métricas agregadas
    """
    from flwr.common.logger import log
    from logging import INFO

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
