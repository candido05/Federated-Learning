"""Federated learning client implementation with VRAM reporting."""
import torch
from typing import Dict

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

from models import NetResNet
from data_utils import load_datasets
from utils import get_parameters, set_parameters, train, test, get_device, get_total_vram

# Constants
NUM_CLIENTS = 10


class EnhancedFlowerClient(NumPyClient):
    """Flower client that reports VRAM information."""
    def __init__(self, partition_id, net, trainloader, valloader, device_id=-1):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device_id = device_id
        self.device = get_device()  # Get the appropriate device (CPU or GPU)

        # Simulate VRAM values
        if torch.cuda.is_available() and device_id >= 0:
            try:
                self.total_vram = torch.cuda.get_device_properties(device_id).total_memory / (1024 * 1024 * 1024)  # GB
                # Simulate different free VRAM for different clients (between 20% and 90% of total)
                self.free_vram = self.total_vram * (0.2 + (0.7 * (partition_id / NUM_CLIENTS)))
            except RuntimeError as e:
                print(f"Error accessing GPU properties: {e}. Using simulated values.")
                self.total_vram = 8.0  # Simulated GB
                self.free_vram = 4.0  # Simulated GB
        else:
            self.total_vram = 8.0  # Simulated GB for CPU
            self.free_vram = 4.0  # Simulated GB for CPU

    def get_properties(self, config):
        """Return VRAM information for client selection."""
        print(f"[Client {self.partition_id}] get_properties called")
        return {"VRAM": self.free_vram}

    def get_parameters(self, config):
        """Get model parameters."""
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """Train model on local data."""
        print(f"[Client {self.partition_id}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=3, device=self.device)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """Evaluate model on local data."""
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn_resnet(context: Context) -> Client:
    """Factory for ResNet client with VRAM reporting."""
    # Get the device in the current process context
    device = get_device()

    net = NetResNet().to(device)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader, _ = load_datasets(partition_id, num_partitions)

    # Define device_id as 0 for GPU or -1 for CPU
    device_id = 0 if torch.cuda.is_available() else -1

    return EnhancedFlowerClient(partition_id, net, trainloader, valloader, device_id).to_client()


# Create the client application
def create_client_app():
    """Create a Flower client application."""
    return ClientApp(client_fn=client_fn_resnet)