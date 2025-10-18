"""Módulo de carregamento e particionamento de dados para Aprendizado Federado.

Este módulo gerencia o carregamento de datasets, pré-processamento e particionamento
entre clientes de aprendizado federado.
"""

from .base_dataset import BaseDataset
from .dataset_factory import (
    create_dataset,
    get_dataset_info,
    list_available_datasets,
)
from .tabular_dataset import TabularDataset

__all__ = [
    # Classes base
    "BaseDataset",
    # Implementações
    "TabularDataset",
    # Factory
    "create_dataset",
    "list_available_datasets",
    "get_dataset_info",
]
