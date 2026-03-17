import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import (ALGORITHMS, STRATEGIES, METRICS_TO_PLOT,
                    METRIC_LABELS, PLOT_STYLE, PLOTS_DIR)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = PLOT_STYLE['figure_size']
plt.rcParams['font.size'] = PLOT_STYLE['font_size']


def plot_algorithm_comparison(all_data: Dict[str, Dict], comparison_timestamp: str = None) -> List[Path]:
    output_files = []

    if comparison_timestamp is None:
        comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for metric in METRICS_TO_PLOT:
        _, axes = plt.subplots(1, 2, figsize=(16, 6))
        plt.suptitle(f'Comparacao de Algoritmos - {METRIC_LABELS[metric]}',
                     fontsize=16, fontweight='bold')

        for strategy_idx, strategy in enumerate(STRATEGIES):
            ax = axes[strategy_idx]
            ax.set_title(f'Estrategia: {strategy.upper()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Rodada', fontweight='bold')
            ax.set_ylabel(METRIC_LABELS[metric], fontweight='bold')
            ax.grid(True, alpha=PLOT_STYLE['grid_alpha'])

            for algorithm in ALGORITHMS:
                key = f"{algorithm}_{strategy}"
                if key in all_data and all_data[key]:
                    rounds, metrics = all_data[key]['rounds'], all_data[key]['metrics']
                    values = metrics[metric]
                    ax.plot(rounds, values, marker='o',
                           linewidth=PLOT_STYLE['line_width'],
                           markersize=PLOT_STYLE['marker_size'],
                           label=algorithm.upper())

            ax.legend(loc='best')

        plt.tight_layout()

        output_dir = PLOTS_DIR / "comparisons" / comparison_timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        metric_name = metric.replace('_', '-')
        output_file = output_dir / f"comparison_{metric_name}.png"
        plt.savefig(output_file, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
        plt.close('all')

        output_files.append(output_file)

    return output_files


def plot_final_metrics_comparison(all_data: Dict[str, Dict], comparison_timestamp: str = None) -> Path:
    final_metrics = {algo: {strat: {} for strat in STRATEGIES} for algo in ALGORITHMS}

    for algorithm in ALGORITHMS:
        for strategy in STRATEGIES:
            key = f"{algorithm}_{strategy}"
            if key in all_data and all_data[key]:
                metrics = all_data[key]['metrics']
                for metric in METRICS_TO_PLOT:
                    final_metrics[algorithm][strategy][metric] = metrics[metric][-1]

    _, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle('Comparacao de Metricas Finais (Ultima Rodada)',
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    x = np.arange(len(ALGORITHMS))
    width = 0.35

    for idx, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[idx]

        cyclic_values = [final_metrics[algo]['cyclic'].get(metric, 0) for algo in ALGORITHMS]
        bagging_values = [final_metrics[algo]['bagging'].get(metric, 0) for algo in ALGORITHMS]

        ax.bar(x - width/2, cyclic_values, width, label='Cyclic', alpha=0.8)
        ax.bar(x + width/2, bagging_values, width, label='Bagging', alpha=0.8)

        ax.set_xlabel('Algoritmo', fontweight='bold')
        ax.set_ylabel(METRIC_LABELS[metric], fontweight='bold')
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in ALGORITHMS])
        ax.legend(loc='best')
        ax.grid(True, alpha=PLOT_STYLE['grid_alpha'], axis='y')

        for i, (c_val, b_val) in enumerate(zip(cyclic_values, bagging_values)):
            if c_val > 0:
                ax.text(i - width/2, c_val, f'{c_val:.3f}', ha='center', va='bottom', fontsize=8)
            if b_val > 0:
                ax.text(i + width/2, b_val, f'{b_val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if comparison_timestamp is None:
        comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = PLOTS_DIR / "comparisons" / comparison_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "final_metrics_bars.png"
    plt.savefig(output_file, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close('all')

    return output_file


def plot_training_time_comparison(all_data: Dict[str, Dict], comparison_timestamp: str = None) -> Path:
    times = {algo: {strat: 0 for strat in STRATEGIES} for algo in ALGORITHMS}

    for algorithm in ALGORITHMS:
        for strategy in STRATEGIES:
            key = f"{algorithm}_{strategy}"
            if key in all_data and all_data[key]:
                times[algorithm][strategy] = all_data[key]['exp_info']['total_time_seconds']

    _, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ALGORITHMS))
    width = 0.35

    cyclic_times = [times[algo]['cyclic'] for algo in ALGORITHMS]
    bagging_times = [times[algo]['bagging'] for algo in ALGORITHMS]

    ax.bar(x - width/2, cyclic_times, width, label='Cyclic', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, bagging_times, width, label='Bagging', alpha=0.8, color='coral')

    ax.set_xlabel('Algoritmo', fontweight='bold', fontsize=12)
    ax.set_ylabel('Tempo Total (segundos)', fontweight='bold', fontsize=12)
    ax.set_title('Comparacao de Tempo de Treinamento', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in ALGORITHMS])
    ax.legend(loc='best')
    ax.grid(True, alpha=PLOT_STYLE['grid_alpha'], axis='y')

    for time_val, x_pos in zip(cyclic_times + bagging_times,
                                list(x - width/2) + list(x + width/2)):
        if time_val > 0:
            ax.text(x_pos, time_val, f'{time_val:.1f}s',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if comparison_timestamp is None:
        comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = PLOTS_DIR / "comparisons" / comparison_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "training_time.png"
    plt.savefig(output_file, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close('all')

    return output_file
