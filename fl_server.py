"""Federated learning server implementation."""
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
from flwr.common import Context, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from models import NetResNet, NetEfficientNetV2, NetMobileNetV3
from data_utils import load_datasets
from utils import get_parameters, set_parameters, test, get_device
from fl_strategies import (
    VRAMFedAvg, VRAMFedOpt, VRAMFedAvgM, VRAMFedProx, 
    VRAMFedAdagrad, VRAMFedAdam, VRAMFedMedian
)

# Constants
NUM_CLIENTS = 10
NUM_ROUNDS = 3

# Get initial parameters for different models
def get_initial_parameters():
    """Get initial parameters for different models."""
    return {
        "resnet": get_parameters(NetResNet()),
        "efficientnet": get_parameters(NetEfficientNetV2()),
        "mobilenet": get_parameters(NetMobileNetV3())
    }


# Define evaluation function for ResNet model
def evaluate_resnet(
    server_round: int,
    parameters: List[np.ndarray],
    config: Dict[str, float],
    results_dict: Dict[str, List[float]],
) -> Optional[Tuple[float, Dict[str, float]]]:
    """Evaluate ResNet model on the server side."""
    device = get_device()
    net = NetResNet().to(device)
    _, _, testloader = load_datasets(0, NUM_CLIENTS)
    set_parameters(net, parameters)

    loss, accuracy = test(net, testloader, device)
    print(f"Server-side evaluation (Round {server_round}): Loss={loss:.4f}, Accuracy={accuracy:.4f}")

    # Store results
    results_dict["rounds"].append(server_round)
    results_dict["loss"].append(loss)
    results_dict["accuracy"].append(accuracy)

    return loss, {"accuracy": accuracy}


# Create server functions for all algorithms with VRAM-based client selection
def create_server_fn(strategy_name, strategy_class, params, results_dict):
    """Create a server function with the specified strategy and VRAM-based client selection."""
    def server_fn(context: Context) -> ServerAppComponents:
        """Configure server with VRAM-based client selection strategy."""
        strategy = strategy_class(
            fraction_fit=0.5,  # Select half of the available clients
            fraction_evaluate=0.5,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=NUM_CLIENTS,
            initial_parameters=ndarrays_to_parameters(params),
            evaluate_fn=lambda r, p, c: evaluate_resnet(r, p, c, results_dict),
        )
        config = ServerConfig(num_rounds=NUM_ROUNDS)
        return ServerAppComponents(strategy=strategy, config=config)
    return server_fn


# Create server apps for all strategies
def create_server_apps(initial_params, results):
    """Create server apps for all FL strategies."""
    return {
        "VRAMFedAvg": ServerApp(
            server_fn=create_server_fn("VRAMFedAvg", VRAMFedAvg, initial_params["resnet"], results["VRAMFedAvg"])
        ),
        "VRAMFedOpt": ServerApp(
            server_fn=create_server_fn("VRAMFedOpt", VRAMFedOpt, initial_params["resnet"], results["VRAMFedOpt"])
        ),
        "VRAMFedAvgM": ServerApp(
            server_fn=create_server_fn("VRAMFedAvgM", VRAMFedAvgM, initial_params["resnet"], results["VRAMFedAvgM"])
        ),
        "VRAMFedProx": ServerApp(
            server_fn=create_server_fn("VRAMFedProx", VRAMFedProx, initial_params["resnet"], results["VRAMFedProx"])
        ),
        "VRAMFedAdagrad": ServerApp(
            server_fn=create_server_fn("VRAMFedAdagrad", VRAMFedAdagrad, initial_params["resnet"], results["VRAMFedAdagrad"])
        ),
        "VRAMFedAdam": ServerApp(
            server_fn=create_server_fn("VRAMFedAdam", VRAMFedAdam, initial_params["resnet"], results["VRAMFedAdam"])
        ),
        "VRAMFedMedian": ServerApp(
            server_fn=create_server_fn("VRAMFedMedian", VRAMFedMedian, initial_params["resnet"], results["VRAMFedMedian"])
        )
    }