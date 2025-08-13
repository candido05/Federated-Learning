"""
Carregamento de dados MNIST para Federated Learning
"""

import numpy as np
from flwr_datasets import FederatedDataset
from datasets.utils.logging import disable_progress_bar
from typing import Tuple

disable_progress_bar()


def load_mnist_datasets(partition_id: int, num_partitions: int) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                                        Tuple[np.ndarray, np.ndarray], 
                                                                        Tuple[np.ndarray, np.ndarray]]:
    """
    Carrega os datasets MNIST para um cliente específico
    
    Args:
        partition_id: ID da partição do cliente
        num_partitions: Número total de partições
        
    Returns:
        Tupla contendo (train_data, val_data, test_data)
        Cada elemento é uma tupla (X, y) onde X são as imagens e y os rótulos
    """
    
    fds = FederatedDataset(dataset="mnist", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    train_data = partition_train_test["train"]
    X_train = np.array([np.array(img) for img in train_data["image"]])
    y_train = np.array(train_data["label"])
    
    val_data = partition_train_test["test"]
    X_val = np.array([np.array(img) for img in val_data["image"]])
    y_val = np.array(val_data["label"])
    
    testset = fds.load_split("test")
    X_test = np.array([np.array(img) for img in testset["image"]])
    y_test = np.array(testset["label"])
    
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    print(f"Cliente {partition_id}: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_centralized_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega dados de teste centralizados para avaliação do servidor
    
    Returns:
        Tupla (X_test, y_test)
    """

    fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    testset = fds.load_split("test")
    
    X_test = np.array([np.array(img) for img in testset["image"]])
    y_test = np.array(testset["label"])
    
    # Normaliza os dados
    X_test = X_test.astype(np.float32) / 255.0
    
    return X_test, y_test


def get_data_info():
    """Retorna informações sobre o dataset MNIST"""
    return {
        "name": "MNIST",
        "num_classes": 10,
        "input_shape": (28, 28),
        "input_size": 784,  # 28 * 28 para modelos que precisam de entrada flat
        "classes": list(range(10))
    }