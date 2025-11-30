"""
Common utilities para Federated Learning
Estrutura modular para processamento de dados, métricas e logging
"""

# Data processing
from .data_processing import DataProcessor

# Métricas
from .metrics import (
    calculate_comprehensive_metrics,
    print_metrics_summary,
    evaluate_metrics_aggregation
)

# Logging
from .logger import ExperimentLogger

# Utilities
from .utils import replace_keys

__all__ = [
    # Data
    'DataProcessor',

    # Metrics
    'calculate_comprehensive_metrics',
    'print_metrics_summary',
    'evaluate_metrics_aggregation',

    # Logging
    'ExperimentLogger',

    # Utils
    'replace_keys'
]
