"""Utility functions for federated learning."""
from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn
import numpy as np


def get_device():
    """Return the available device with safe initialization."""
    try:
        if torch.cuda.is_available():
            torch.cuda.init()  # Force GPU initialization
            device = torch.device("cuda:0")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("No GPU available, using CPU")
    except RuntimeError as e:
        print(f"Error initializing CUDA: {e}. Using CPU as fallback.")
        device = torch.device("cpu")
    return device


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """Extract model parameters and convert them to numpy arrays."""
    return [val.cpu().detach().numpy() for val in net.state_dict().values()]


def set_parameters(net: nn.Module, parameters: List[np.ndarray]):
    """Set model parameters from a list of numpy arrays."""
    # Validate parameters are not empty
    if not parameters:
        raise ValueError("Parameter list is empty. Check received data.")

    # Build state dictionary
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    # Check for dimension inconsistencies before loading
    for name, param in state_dict.items():
        expected_shape = net.state_dict()[name].shape
        if param.shape != expected_shape:
            raise ValueError(
                f"Parameter inconsistency: {name} expected {expected_shape}, but received {param.shape}"
            )

    # Load parameters into model
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int, device):
    """Train the model."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")


def test(net, testloader, device):
    """Test the model."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def get_free_vram(device_id=0):
    """Get free VRAM in GB for a specific GPU device."""
    if not torch.cuda.is_available():
        return 0.0

    # Total memory of the GPU
    total_memory = torch.cuda.get_device_properties(device_id).total_memory
    # Currently allocated memory
    allocated_memory = torch.cuda.memory_allocated(device_id)
    # Currently reserved (cached) memory
    reserved_memory = torch.cuda.memory_reserved(device_id)

    # Free memory is the difference between total and (allocated + reserved)
    free_memory = total_memory - (allocated_memory + reserved_memory)
    return free_memory / (1024**3)  # Convert to GB


def get_total_vram(device_id=0):
    """Get total VRAM in GB for a specific GPU device."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # Convert to GB


def display_gpu_info():
    """Display GPU information if available."""
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 2), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 2), "GB")
        print("Total:    ", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
    else:
        print("No GPU available")