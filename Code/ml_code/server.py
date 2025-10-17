"""
Servidor para Federated Learning com modelos de ML
"""

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedOpt, FedAvgM, FedProx, FedAdagrad, FedAdam, FedMedian
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
from data_loader import load_centralized_test_data
from models import create_model
from typing import Dict, List, Optional, Tuple


class ServerEvaluator:
    """Classe para avaliação no servidor"""
    
    def __init__(self, model_name: str, results_dict: Dict[str, List[float]]):
        self.model_name = model_name
        self.results_dict = results_dict
        self.model = create_model(model_name)
        self.X_test, self.y_test = load_centralized_test_data()
        
    def evaluate(self, server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Avalia o modelo global no servidor"""
        print(f"[Servidor] Avaliando {self.model_name} - Rodada {server_round}")
        
        self.model.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        
        print(f"[Servidor] {self.model_name} - Rodada {server_round}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        # Armazena os resultados
        self.results_dict["rounds"].append(server_round)
        self.results_dict["loss"].append(loss)
        self.results_dict["accuracy"].append(accuracy)
        
        return loss, {"accuracy": accuracy}


def create_server_fn(strategy_name: str, model_name: str, results_dict: Dict[str, List[float]], 
                    num_partitions: int = 10, num_rounds: int = 10):
    """Cria função do servidor para uma estratégia específica"""
    
    def server_fn(context: Context) -> ServerAppComponents:
        # Cria modelo inicial para obter parâmetros
        initial_model = create_model(model_name)
        initial_params = ndarrays_to_parameters(initial_model.get_parameters())
        
        # Cria avaliador
        evaluator = ServerEvaluator(model_name, results_dict)
        
        # Parâmetros base para todas as estratégias
        base_params = {
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "min_fit_clients": min(5, num_partitions),
            "min_evaluate_clients": min(5, num_partitions),
            "min_available_clients": num_partitions,
            "initial_parameters": initial_params,
            "evaluate_fn": evaluator.evaluate,
        }
        
        # Cria estratégia baseada no nome
        strategies = {
            "FedAvg": lambda: FedAvg(**base_params),
            "FedOpt": lambda: FedOpt(**base_params, eta=0.05, beta_1=0.9, beta_2=0.999),
            "FedAvgM": lambda: FedAvgM(**base_params, server_learning_rate=1.0, server_momentum=0.9),
            "FedProx": lambda: FedProx(**base_params, proximal_mu=0.1),
            "FedAdagrad": lambda: FedAdagrad(**base_params, eta=0.01),
            "FedAdam": lambda: FedAdam(**base_params, eta=0.001, beta_1=0.9, beta_2=0.999),
            "FedMedian": lambda: FedMedian(**base_params),
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Estratégia {strategy_name} não suportada")
        
        strategy = strategies[strategy_name]()
        config = ServerConfig(num_rounds=num_rounds)
        
        return ServerAppComponents(strategy=strategy, config=config)
    
    return server_fn


def create_server_app(strategy_name: str, model_name: str, results_dict: Dict[str, List[float]], 
                     num_partitions: int = 10, num_rounds: int = 10) -> ServerApp:
    """Cria aplicação servidor"""
    server_fn = create_server_fn(strategy_name, model_name, results_dict, num_partitions, num_rounds)
    return ServerApp(server_fn=server_fn)


# Lista de estratégias disponíveis
AVAILABLE_STRATEGIES = [
    "FedAvg",
    "FedOpt", 
    "FedAvgM",
    "FedProx",
    "FedAdagrad",
    "FedAdam",
    "FedMedian"
]