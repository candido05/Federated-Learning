from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

ALGORITHMS = ["xgboost", "lightgbm", "catboost"]
STRATEGIES = ["cyclic", "bagging"]

METRICS_TO_PLOT = [
    "accuracy",
    "f1_score_weighted",
    "auc",
    "precision_weighted",
    "recall_weighted",
    "balanced_accuracy"
]

METRIC_LABELS = {
    "accuracy": "Acurácia",
    "f1_score_weighted": "F1-Score (Weighted)",
    "auc": "AUC (Macro OvR)",
    "precision_weighted": "Precisão (Weighted)",
    "recall_weighted": "Recall (Weighted)",
    "balanced_accuracy": "Acurácia Balanceada"
}

PLOT_STYLE = {
    'figure_size': (12, 8),
    'font_size': 10,
    'dpi': 300,
    'grid_alpha': 0.3,
    'line_width': 2,
    'marker_size': 4
}
