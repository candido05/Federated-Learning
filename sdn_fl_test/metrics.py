import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score
)

def calculate_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Calcula um conjunto abrangente de métricas para avaliação."""
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    
    if y_proba is not None:
        try:
            # Para o XGBoost binário, y_proba geralmente tem 2 colunas
            m["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except:
            m["auc"] = 0.0
    return m