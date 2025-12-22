"""
Plot Generation Module - Federated Learning Metrics Visualization

Estrutura modular:
- config.py: Configurações e constantes
- data_loader.py: Carregamento de dados dos logs
- single_plots.py: Plots para algoritmos individuais
- comparison_plots.py: Plots de comparação entre algoritmos
- plot_metrics.py: Script principal
"""

from .plot_metrics import main, generate_single_algorithm_plots, generate_comparison_plots
from .data_loader import load_all_experiments, find_most_recent_experiment
from .config import ALGORITHMS, STRATEGIES, METRICS_TO_PLOT, PLOTS_DIR

__all__ = [
    'main',
    'generate_single_algorithm_plots',
    'generate_comparison_plots',
    'load_all_experiments',
    'find_most_recent_experiment',
    'ALGORITHMS',
    'STRATEGIES',
    'METRICS_TO_PLOT',
    'PLOTS_DIR'
]
