"""
Funcoes de metricas, thresholds e avaliacao de fairness por grupo.
"""

import numpy as np
from typing import Dict, Tuple

from sklearn.metrics import roc_auc_score, roc_curve


# =============================================================================
# Threshold Calculations
# =============================================================================

def calc_tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray,
                    fpr_target: float = 0.05) -> float:
    """Calcula TPR quando FPR <= fpr_target."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid_indices = np.where(fpr <= fpr_target)[0]

    if len(valid_indices) == 0:
        return 0.0

    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
    return tpr[best_idx]


def get_threshold_at_fpr(y_true: np.ndarray, y_prob: np.ndarray,
                         fpr_target: float = 0.05) -> float:
    """Retorna threshold correspondente ao FPR alvo."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_indices = np.where(fpr <= fpr_target)[0]

    if len(valid_indices) == 0:
        return 1.0

    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
    return thresholds[best_idx]


def get_group_threshold_at_fpr(y_true: np.ndarray, y_prob: np.ndarray,
                                fpr_target: float = 0.05) -> Tuple[float, float]:
    """
    Calcula threshold que resulta em FPR = fpr_target para um grupo especifico.
    Retorna (threshold, tpr_at_threshold).
    """
    if len(y_true) == 0 or y_true.sum() == 0 or (y_true == 0).sum() == 0:
        return 0.5, 0.0

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_indices = np.where(fpr <= fpr_target)[0]

    if len(valid_indices) == 0:
        return thresholds[0] if len(thresholds) > 0 else 1.0, 0.0

    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
    return thresholds[best_idx], tpr[best_idx]


# =============================================================================
# Fairness: Group-Based Thresholds
# =============================================================================

def _calc_group_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula TPR e FPR para um grupo."""
    if len(y_true) == 0:
        return {'tpr': 0.0, 'fpr': 0.0}

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {'tpr': tpr, 'fpr': fpr}


def predict_with_group_thresholds(y_prob: np.ndarray, y_true: np.ndarray,
                                   age_flag: np.ndarray,
                                   fpr_target: float = 0.05) -> Tuple[np.ndarray, Dict]:
    """
    Predicao com limiares especificos por grupo (correcao de fairness).
    """
    mask_young = age_flag == 0
    mask_old = age_flag == 1

    thresh_young, _ = get_group_threshold_at_fpr(
        y_true[mask_young], y_prob[mask_young], fpr_target
    )
    thresh_old, _ = get_group_threshold_at_fpr(
        y_true[mask_old], y_prob[mask_old], fpr_target
    )

    y_pred = np.zeros(len(y_prob), dtype=int)
    y_pred[mask_young] = (y_prob[mask_young] >= thresh_young).astype(int)
    y_pred[mask_old] = (y_prob[mask_old] >= thresh_old).astype(int)

    rates_young = _calc_group_rates(y_true[mask_young], y_pred[mask_young])
    rates_old = _calc_group_rates(y_true[mask_old], y_pred[mask_old])

    tp_global = ((y_pred == 1) & (y_true == 1)).sum()
    fp_global = ((y_pred == 1) & (y_true == 0)).sum()
    fn_global = ((y_pred == 0) & (y_true == 1)).sum()
    tn_global = ((y_pred == 0) & (y_true == 0)).sum()

    tpr_global = tp_global / (tp_global + fn_global) if (tp_global + fn_global) > 0 else 0.0
    fpr_global = fp_global / (fp_global + tn_global) if (fp_global + tn_global) > 0 else 0.0

    max_fpr = max(rates_old['fpr'], rates_young['fpr'])
    fairness_ratio = min(rates_old['fpr'], rates_young['fpr']) / max_fpr if max_fpr > 0 else 1.0

    return y_pred, {
        'thresh_young': thresh_young,
        'thresh_old': thresh_old,
        'tpr_young': rates_young['tpr'],
        'tpr_old': rates_old['tpr'],
        'fpr_young': rates_young['fpr'],
        'fpr_old': rates_old['fpr'],
        'tpr_global': tpr_global,
        'fpr_global': fpr_global,
        'fairness_ratio': fairness_ratio
    }


# =============================================================================
# Model Evaluation (Standard + Fair)
# =============================================================================

def evaluate_model_standard(y_true: np.ndarray, y_prob: np.ndarray,
                            age_flag: np.ndarray) -> Dict[str, float]:
    """Avaliacao com threshold unico."""
    tpr_at_5fpr = calc_tpr_at_fpr(y_true, y_prob, fpr_target=0.05)
    roc_auc = roc_auc_score(y_true, y_prob)
    threshold = get_threshold_at_fpr(y_true, y_prob, fpr_target=0.05)
    y_pred = (y_prob >= threshold).astype(int)

    mask_old = age_flag == 1
    mask_young = age_flag == 0

    def calc_fpr_group(y_true_g, y_pred_g):
        neg_mask = y_true_g == 0
        if neg_mask.sum() == 0:
            return 0.0
        fp = ((y_pred_g == 1) & (y_true_g == 0)).sum()
        return fp / neg_mask.sum()

    fpr_old = calc_fpr_group(y_true[mask_old], y_pred[mask_old])
    fpr_young = calc_fpr_group(y_true[mask_young], y_pred[mask_young])
    max_fpr = max(fpr_old, fpr_young)
    fairness_ratio = min(fpr_old, fpr_young) / max_fpr if max_fpr > 0 else 1.0

    return {
        'tpr_at_5fpr': tpr_at_5fpr,
        'roc_auc': roc_auc,
        'fairness_ratio': fairness_ratio,
        'fpr_old': fpr_old,
        'fpr_young': fpr_young,
        'threshold': threshold,
    }


def evaluate_model_fair(y_true: np.ndarray, y_prob: np.ndarray,
                        age_flag: np.ndarray) -> Dict[str, float]:
    """Avaliacao com fairness corrigido (limiares por grupo)."""
    roc_auc = roc_auc_score(y_true, y_prob)
    _, fair_metrics = predict_with_group_thresholds(y_prob, y_true, age_flag)

    return {
        'tpr_at_5fpr': fair_metrics['tpr_global'],
        'roc_auc': roc_auc,
        'fairness_ratio': fair_metrics['fairness_ratio'],
        'fpr_old': fair_metrics['fpr_old'],
        'fpr_young': fair_metrics['fpr_young'],
        'tpr_old': fair_metrics['tpr_old'],
        'tpr_young': fair_metrics['tpr_young'],
        'thresh_young': fair_metrics['thresh_young'],
        'thresh_old': fair_metrics['thresh_old'],
    }
