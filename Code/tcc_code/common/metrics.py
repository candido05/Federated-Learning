"""
Cálculo de métricas para classificação multi-classe (3 classes fixas)
Dataset: Veículos com 3 classes de comportamento
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
    balanced_accuracy_score,
    matthews_corrcoef,
)
from flwr.common.logger import log
from logging import WARNING

NUM_CLASSES = 3


def calculate_comprehensive_metrics(y_true, y_pred_proba, threshold=0.5) -> Dict:
    """
    Calcula métricas para classificação multi-classe (3 classes)

    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas (shape: [n_samples, 3])
        threshold: Não usado em multi-classe (mantido por compatibilidade)

    Returns:
        Dict com métricas: accuracy, precision, recall, F1, AUC, confusion matrix
    """
    if len(y_pred_proba.shape) == 1:
        y_pred = y_pred_proba.astype(int)
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    except Exception as e:
        log(WARNING, f"Erro ao calcular AUC: {e}")
        auc = 0.5

    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'mcc': float(mcc),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_score_weighted': float(f1_weighted),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_score_macro': float(f1_macro),
        'auc': float(auc),
        'num_classes': NUM_CLASSES,
        'confusion_matrix': cm.tolist(),
        'recall_class_0': float(recall_per_class[0]),
        'recall_class_1': float(recall_per_class[1]),
        'recall_class_2': float(recall_per_class[2]),
        'precision_class_0': float(precision_per_class[0]),
        'precision_class_1': float(precision_per_class[1]),
        'precision_class_2': float(precision_per_class[2]),
        'f1_class_0': float(f1_per_class[0]),
        'f1_class_1': float(f1_per_class[1]),
        'f1_class_2': float(f1_per_class[2]),
    }

    return metrics


def print_metrics_summary(metrics: Dict, client_id: int = None, round_num: int = None):
    """Imprime resumo organizado das métricas (3 classes)"""
    prefix = f"[Client {client_id}]" if client_id is not None else "[Server]"
    if round_num is not None:
        prefix += f" Round {round_num}"

    print(f"\n{prefix} Métricas de Performance:")
    print(f"  Acurácia:           {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {metrics.get('balanced_accuracy', 0):.4f}  [PRIORIDADE]")
    print(f"  MCC:                {metrics.get('mcc', 0):.4f}")
    print(f"  AUC:                {metrics['auc']:.4f} (macro avg)")

    print(f"\n  Métricas Macro (média não-ponderada - PRIORIZAR!):")
    print(f"    Precisão:  {metrics['precision_macro']:.4f}")
    print(f"    Revocação: {metrics['recall_macro']:.4f}  [MÉTRICA PRINCIPAL]")
    print(f"    F1-Score:  {metrics['f1_score_macro']:.4f}  [MÉTRICA PRINCIPAL]")

    print(f"\n  Recall por Classe (detectar problemas em classes minoritárias):")
    print(f"    Classe 0 (minoritária): {metrics.get('recall_class_0', 0):.4f}")
    print(f"    Classe 1 (majoritária): {metrics.get('recall_class_1', 0):.4f}")
    print(f"    Classe 2 (minoritária): {metrics.get('recall_class_2', 0):.4f}")

    cm = metrics['confusion_matrix']
    if isinstance(cm, list):
        cm_array = np.array(cm)
        print(f"\n  Matriz de Confusão (3 classes):")
        for i, row in enumerate(cm_array):
            print(f"    Classe {i}: {list(row)}")


def evaluate_metrics_aggregation(eval_metrics):
    """Agrega métricas de múltiplos clientes (3 classes)"""
    from flwr.common.logger import log
    from logging import INFO

    if not eval_metrics:
        return {
            "auc": 0.5,
            "accuracy": 0.5,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_score_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_score_macro": 0.0
        }

    total_num = sum([num for num, _ in eval_metrics])
    if total_num == 0:
        return {
            "auc": 0.5,
            "accuracy": 0.5,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_score_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_score_macro": 0.0
        }

    metric_sums = {
        "auc": 0.0,
        "accuracy": 0.0,
        "precision_weighted": 0.0,
        "recall_weighted": 0.0,
        "f1_score_weighted": 0.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "f1_score_macro": 0.0,
    }

    for num, metrics in eval_metrics:
        for metric_name in metric_sums.keys():
            # Compatibilidade: aceitar nomes antigos (precision, recall, f1_score)
            if metric_name == "precision_weighted":
                metric_val = metrics.get(metric_name, metrics.get("precision", 0.0))
            elif metric_name == "recall_weighted":
                metric_val = metrics.get(metric_name, metrics.get("recall", 0.0))
            elif metric_name == "f1_score_weighted":
                metric_val = metrics.get(metric_name, metrics.get("f1_score", 0.0))
            else:
                metric_val = metrics.get(metric_name, 0.5 if metric_name in ['auc', 'accuracy'] else 0.0)
            metric_sums[metric_name] += metric_val * num

    metrics_aggregated = {}
    for metric_name, total_sum in metric_sums.items():
        metrics_aggregated[metric_name] = total_sum / total_num

    log(INFO, f"Métricas agregadas:")
    for metric_name, value in metrics_aggregated.items():
        log(INFO, f"  {metric_name}: {value:.4f}")

    return metrics_aggregated
