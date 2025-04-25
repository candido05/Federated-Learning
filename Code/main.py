"""Main script for federated learning with RAM-based client selection."""
import sys
import argparse
import matplotlib.pyplot as plt
import torch
import flwr

from flwr.simulation import run_simulation

from utils import get_device, display_gpu_info
from models import NetResNet, NetEfficientNetV2, NetMobileNetV3
from data_utils import load_datasets, visualize_dataset_sample
from fl_client import create_client_app
from fl_server import get_initial_parameters, create_server_apps

# Constants
NUM_CLIENTS = 10
BATCH_SIZE = 32
NUM_ROUNDS = 3


def init_environment():
    """Initialize the environment, check GPU, etc."""
    # Check for GPU and set device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Free previously allocated GPU memory
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    device = get_device()
    print(f"Training on {device}")
    print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
    
    # Display more GPU information if available
    display_gpu_info()
    
    return device


def run_single_simulation(strategy_name, server_app, client_app, backend_config):
    """Run a simulation for a single strategy."""
    print(f"\n\nRunning simulation with {strategy_name}:")
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )


def run_all_simulations(server_apps, client_app, backend_config):
    """Run simulations for all algorithms with RAM-based client selection."""
    for strategy_name, server_app in server_apps.items():
        run_single_simulation(strategy_name, server_app, client_app, backend_config)


def visualize_results(results):
    """Visualize the results of the different FL strategies."""
    plt.figure(figsize=(12, 10))
    
    # Plot accuracies
    plt.subplot(2, 1, 1)
    for strategy_name, strategy_results in results.items():
        if strategy_results["rounds"]:  # Only plot if we have data
            plt.plot(strategy_results["rounds"], strategy_results["accuracy"], marker='o', label=strategy_name)
    
    plt.title('Accuracy vs. Rounds for Different FL Strategies')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot losses
    plt.subplot(2, 1, 2)
    for strategy_name, strategy_results in results.items():
        if strategy_results["rounds"]:  # Only plot if we have data
            plt.plot(strategy_results["rounds"], strategy_results["loss"], marker='x', label=strategy_name)
    
    plt.title('Loss vs. Rounds for Different FL Strategies')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fl_results_comparison.png')
    plt.show()


def explore_dataset():
    """Explore the dataset by loading a sample and visualizing it."""
    trainloader, _, _ = load_datasets(partition_id=0, num_partitions=NUM_CLIENTS)
    visualize_dataset_sample(trainloader)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Federated Learning with VRAM-based Client Selection')
    parser.add_argument('--strategy', choices=['all', 'fedavg', 'fedopt', 'fedavgm', 'fedprox', 'fedadagrad', 'fedadam', 'fedmedian'], 
                        default='all', help='Which FL strategy to run')
    parser.add_argument('--explore-data', action='store_true', help='Explore the dataset')
    parser.add_argument('--num-clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--num-rounds', type=int, default=3, help='Number of rounds')
    
    args = parser.parse_args()
    
    # Update global constants if provided
    global NUM_CLIENTS, NUM_ROUNDS
    NUM_CLIENTS = args.num_clients
    NUM_ROUNDS = args.num_rounds
    
    # Initialize environment
    device = init_environment()
    
    # Explore dataset if requested
    if args.explore_data:
        explore_dataset()
        return

    # Initialize results dictionary
    results = {
        "VRAMFedAvg": {"rounds": [], "loss": [], "accuracy": []},
        "VRAMFedOpt": {"rounds": [], "loss": [], "accuracy": []},
        "VRAMFedAvgM": {"rounds": [], "loss": [], "accuracy": []},
        "VRAMFedProx": {"rounds": [], "loss": [], "accuracy": []},
        "VRAMFedAdagrad": {"rounds": [], "loss": [], "accuracy": []},
        "VRAMFedAdam": {"rounds": [], "loss": [], "accuracy": []},
        "VRAMFedMedian": {"rounds": [], "loss": [], "accuracy": []},
    }
    
    # Get initial parameters
    initial_params = get_initial_parameters()
    
    # Create client app
    client_app = create_client_app()
    
    # Create server apps for all strategies
    server_apps = create_server_apps(initial_params, results)
    
    # Backend configuration
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.2 if torch.cuda.is_available() else 0}}
    
    # Run simulations based on the selected strategy
    if args.strategy == 'all':
        run_all_simulations(server_apps, client_app, backend_config)
    else:
        strategy_key = f"VRAM{args.strategy.capitalize()}"
        if strategy_key in server_apps:
            run_single_simulation(strategy_key, server_apps[strategy_key], client_app, backend_config)
        else:
            print(f"Strategy {args.strategy} not found!")
            return
    
    # Visualize results
    visualize_results(results)


if __name__ == "__main__":
    main()