"""
Cliente para Federated Learning com modelos de ML
"""

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from data_loader import load_mnist_datasets
from models import create_model
from typing import List
import numpy as np


class MLFlowerClient(NumPyClient):
    """Cliente Flower para modelos de Machine Learning"""
    
    def __init__(self, partition_id: int, model_name: str, num_partitions: int):
        self.partition_id = partition_id
        self.model_name = model_name
        self.model = create_model(model_name)
        
        # Carrega os dados do cliente
        (self.X_train, self.y_train), (self.X_val, self.y_val), _ = load_mnist_datasets(
            partition_id, num_partitions
        )
        
        print(f"[Cliente {partition_id}] Inicializado com modelo {model_name}")
        print(f"[Cliente {partition_id}] Dados: Train={len(self.X_train)}, Val={len(self.X_val)}")

    def get_parameters(self, config) -> List[np.ndarray]:
        """Retorna os parâmetros atuais do modelo"""
        print(f"[Cliente {self.partition_id}] get_parameters")
        return self.model.get_parameters()

    def fit(self, parameters: List[np.ndarray], config):
        """Treina o modelo com os parâmetros recebidos"""
        print(f"[Cliente {self.partition_id}] fit, config: {config}")
        
        self.model.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        
        # Retorna os parâmetros atualizados
        return self.model.get_parameters(), len(self.X_train), {}

    def evaluate(self, parameters: List[np.ndarray], config):
        """Avalia o modelo com os parâmetros recebidos"""
        print(f"[Cliente {self.partition_id}] evaluate, config: {config}")
        
        self.model.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val)
        
        print(f"[Cliente {self.partition_id}] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return float(loss), len(self.X_val), {"accuracy": float(accuracy)}


def create_client_fn(model_name: str, num_partitions: int):
    """Cria função de cliente para um modelo específico"""
    def client_fn(context: Context) -> Client:
        partition_id = context.node_config["partition-id"]
        return MLFlowerClient(partition_id, model_name, num_partitions).to_client()
    
    return client_fn


def create_client_app(model_name: str, num_partitions: int = 10) -> ClientApp:
    """Cria aplicação cliente para um modelo específico"""
    client_fn = create_client_fn(model_name, num_partitions)
    return ClientApp(client_fn=client_fn)