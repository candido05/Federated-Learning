import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import METRICS_TO_PLOT, METRIC_LABELS, PLOT_STYLE, PLOTS_DIR

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = PLOT_STYLE['figure_size']
plt.rcParams['font.size'] = PLOT_STYLE['font_size']


def plot_metrics_evolution(algorithm: str, strategy: str,
                           rounds: List[int], metrics: Dict[str, List[float]],
                           exp_info: Dict, exp_timestamp: str = None) -> Path:
    _, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle(f'{algorithm.upper()} - Estrategia: {strategy.upper()}\n'
                 f'Clientes: {exp_info["num_clients"]} | '
                 f'Rodadas: {exp_info["num_rounds"]} | '
                 f'Tempo: {exp_info["total_time_seconds"]:.1f}s',
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for idx, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[idx]
        values = metrics[metric]

        ax.plot(rounds, values, marker='o',
               linewidth=PLOT_STYLE['line_width'],
               markersize=PLOT_STYLE['marker_size'])
        ax.set_xlabel('Rodada', fontweight='bold')
        ax.set_ylabel(METRIC_LABELS[metric], fontweight='bold')
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=PLOT_STYLE['grid_alpha'])

        final_value = values[-1]
        max_value = max(values)
        max_round = rounds[values.index(max_value)]

        ax.axhline(y=max_value, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(0.02, 0.98, f'Final: {final_value:.4f}\nMax: {max_value:.4f} (R{max_round})',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if exp_timestamp is None:
        exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = PLOTS_DIR / algorithm / strategy / exp_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "metrics_evolution.png"
    plt.savefig(output_file, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close('all')

    return output_file


def plot_confusion_matrix(algorithm: str, strategy: str, data: Dict, exp_timestamp: str = None):
    metrics_by_round = data.get("metrics_by_round", {})
    if not metrics_by_round:
        return None

    last_round = max(int(r) for r in metrics_by_round.keys())
    last_round_data = metrics_by_round[str(last_round)]

    cm = np.array(last_round_data.get("confusion_matrix", []))
    if cm.size == 0:
        return None

    _, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Numero de Amostras'})

    ax.set_xlabel('Classe Predita', fontweight='bold', fontsize=12)
    ax.set_ylabel('Classe Real', fontweight='bold', fontsize=12)
    ax.set_title(f'{algorithm.upper()} - {strategy.upper()}\n'
                 f'Matriz de Confusao (Rodada {last_round})',
                 fontsize=14, fontweight='bold', pad=20)

    num_classes = cm.shape[0]
    ax.set_xticklabels([f'Classe {i}' for i in range(num_classes)])
    ax.set_yticklabels([f'Classe {i}' for i in range(num_classes)])

    plt.tight_layout()

    if exp_timestamp is None:
        exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = PLOTS_DIR / algorithm / strategy / exp_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "confusion_matrix.png"
    plt.savefig(output_file, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close('all')

    return output_file


def plot_per_class_metrics(algorithm: str, strategy: str, rounds: List[int], data: Dict, exp_timestamp: str = None) -> Path:
    metrics_by_round = data.get("metrics_by_round", {})
    num_classes = data["experiment_info"].get("num_classes", 3)

    metrics_per_class = {
        'recall': {i: [] for i in range(num_classes)},
        'precision': {i: [] for i in range(num_classes)},
        'f1': {i: [] for i in range(num_classes)}
    }

    for round_num in rounds:
        round_data = metrics_by_round[str(round_num)]
        for cls in range(num_classes):
            metrics_per_class['recall'][cls].append(round_data.get(f'recall_class_{cls}', 0.0))
            metrics_per_class['precision'][cls].append(round_data.get(f'precision_class_{cls}', 0.0))
            metrics_per_class['f1'][cls].append(round_data.get(f'f1_class_{cls}', 0.0))

    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle(f'{algorithm.upper()} - {strategy.upper()}\nMetricas por Classe',
                 fontsize=14, fontweight='bold')

    metric_names = ['Recall', 'Precision', 'F1-Score']
    metric_keys = ['recall', 'precision', 'f1']

    for ax, metric_name, metric_key in zip(axes, metric_names, metric_keys):
        for cls in range(num_classes):
            ax.plot(rounds, metrics_per_class[metric_key][cls],
                   marker='o',
                   linewidth=PLOT_STYLE['line_width'],
                   markersize=PLOT_STYLE['marker_size'],
                   label=f'Classe {cls}')

        ax.set_xlabel('Rodada', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=PLOT_STYLE['grid_alpha'])

    plt.tight_layout()

    if exp_timestamp is None:
        exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = PLOTS_DIR / algorithm / strategy / exp_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "per_class_metrics.png"
    plt.savefig(output_file, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close('all')

    return output_file
