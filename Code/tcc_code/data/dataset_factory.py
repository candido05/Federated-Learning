"""Factory para criação de datasets de Aprendizado Federado.

Este módulo fornece uma função factory que cria instâncias de datasets
baseado no nome, facilitando a extensão para novos datasets.
"""

import logging
from typing import Optional

from .base_dataset import BaseDataset
from .tabular_dataset import TabularDataset

logger = logging.getLogger(__name__)


def create_dataset(
    dataset_name: str,
    config,
    cache_dir: Optional[str] = None,
) -> BaseDataset:
    """Cria e retorna uma instância do dataset especificado.

    Factory function que instancia o dataset apropriado baseado no nome.
    Facilita a adição de novos datasets no futuro.

    Args:
        dataset_name: Nome do dataset a ser carregado (ex: "higgs").
        config: Objeto GlobalConfig com configurações do experimento.
        cache_dir: Diretório para cache dos dados (opcional).

    Returns:
        Instância de BaseDataset (ou subclasse) configurada.

    Raises:
        ValueError: Se o dataset_name não for reconhecido.

    Example:
        >>> from config import GlobalConfig
        >>> config = GlobalConfig(dataset_name="higgs")
        >>> dataset = create_dataset("higgs", config)
        >>> X_train, y_train = dataset.get_train_data()
    """
    dataset_name_lower = dataset_name.lower().strip()

    logger.info(f"Criando dataset: {dataset_name_lower}")

    # Mapeamento de nomes para classes de datasets
    dataset_registry = {
        "higgs": TabularDataset,  # Dataset tabular do HuggingFace (padrão: jxie/higgs)
        # Adicione novos datasets aqui:
        # "mnist": MNISTDataset,
        # "cifar10": CIFAR10Dataset,
        # "custom": CustomDataset,
    }

    # Verifica se o dataset é suportado
    if dataset_name_lower not in dataset_registry:
        available = ", ".join(dataset_registry.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' não reconhecido. "
            f"Datasets disponíveis: {available}"
        )

    # Instancia o dataset apropriado
    dataset_class = dataset_registry[dataset_name_lower]
    dataset = dataset_class(config=config, cache_dir=cache_dir)

    logger.info(f"Dataset {dataset_class.__name__} criado com sucesso")

    return dataset


def list_available_datasets() -> list[str]:
    """Lista todos os datasets disponíveis no sistema.

    Returns:
        Lista de nomes de datasets suportados.

    Example:
        >>> datasets = list_available_datasets()
        >>> print(f"Datasets disponíveis: {datasets}")
        Datasets disponíveis: ['higgs']
    """
    return ["higgs"]  # Expandir conforme novos datasets forem adicionados


def get_dataset_info(dataset_name: str) -> dict:
    """Retorna informações sobre um dataset específico.

    Args:
        dataset_name: Nome do dataset.

    Returns:
        Dicionário com informações do dataset (descrição, tipo, features, etc.).

    Raises:
        ValueError: Se o dataset não for reconhecido.

    Example:
        >>> info = get_dataset_info("higgs")
        >>> print(info["description"])
    """
    dataset_info_registry = {
        "higgs": {
            "name": "HIGGS",
            "full_name": "HIGGS Boson Detection Dataset",
            "description": (
                "Dataset de física de partículas para distinguir processos de sinal "
                "(Higgs boson) de processos de fundo usando features cinemáticas."
            ),
            "source": "HuggingFace (jxie/higgs)",
            "task": "binary_classification",
            "num_features": 28,
            "num_samples": "~11 milhões",
            "classes": ["background", "signal"],
            "reference": "https://huggingface.co/datasets/jxie/higgs",
        },
        # Adicione informações de novos datasets aqui
    }

    dataset_name_lower = dataset_name.lower().strip()

    if dataset_name_lower not in dataset_info_registry:
        available = ", ".join(dataset_info_registry.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' não reconhecido. "
            f"Datasets disponíveis: {available}"
        )

    return dataset_info_registry[dataset_name_lower]
