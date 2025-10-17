"""Federated learning strategies with VRAM-based client selection."""
from typing import List, Tuple, Dict
import numpy as np

from flwr.common import (
    Parameters, FitIns, NDArrays, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays, GetPropertiesIns
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg, FedOpt, FedAvgM, FedProx, FedAdagrad, FedAdam, FedMedian


class VRAMBasedClientSelection:
    """Mixin class for VRAM-based client selection."""

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with VRAM-based client selection."""
        weights = parameters_to_ndarrays(parameters)
        if hasattr(self, 'pre_weights'):
            self.pre_weights = weights
        parameters = ndarrays_to_parameters(weights)

        config = {}
        if hasattr(self, 'on_fit_config_fn') and self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        all_clients = client_manager.all()  # Dict[str, ClientProxy], Return all available clients
        selected_clients = []  # The clients list we will select

        # Set minimum VRAM threshold (in GB)
        min_vram_threshold = 1.0  # Select clients with at least 1GB of free VRAM

        for client in all_clients.values():  # Look at all clients
            config_properties = GetPropertiesIns({"VRAM": 0})
            # Config parameters requested by the server
            # Tell the client which properties are needed along with Scalar attributes

            print(f"[Server] Calling get_properties for client {client.node_id}")
            client_properties = client.get_properties(config_properties, timeout=30.0, group_id=0)
            print(f"[Server] Received properties: {client_properties}")
            # Get each client's properties by passing GetPropertiesIns and timeout parameter

            client_property = client_properties.properties
            if client_property["VRAM"] > min_vram_threshold:  # Choose clients with free VRAM > threshold
                selected_clients.append(client)

        if len(selected_clients) == 0:
            print("Warning: No client has been selected! Using all available clients.")
            selected_clients = list(all_clients.values())[:sample_size]
        elif len(selected_clients) > sample_size:
            # If more clients than needed meet the VRAM requirement, select the ones with the most VRAM
            selected_clients.sort(
                key=lambda client: client.get_properties(GetPropertiesIns({"VRAM": 0}), timeout=2.0).properties["VRAM"],
                reverse=True
            )
            selected_clients = selected_clients[:sample_size]

        print(f"Selected {len(selected_clients)} clients for training in round {server_round}")

        # Return client/config pairs
        return [(client, fit_ins) for client in selected_clients]


# Create RAM-based versions of all strategies
class VRAMFedAvg(VRAMBasedClientSelection, FedAvg):
    """FedAvg with RAM-based client selection."""

    def __repr__(self) -> str:
        return "RAMFedAvg"


class VRAMFedOpt(VRAMBasedClientSelection, FedOpt):
    """FedOpt with RAM-based client selection."""

    def __repr__(self) -> str:
        return "RAMFedOpt"


class VRAMFedAvgM(VRAMBasedClientSelection, FedAvgM):
    """FedAvgM with RAM-based client selection."""

    def __repr__(self) -> str:
        return "RAMFedAvgM"

class VRAMFedAvgM(VRAMBasedClientSelection, FedAvgM):
    """FedAvgM with RAM-based client selection."""

    def __repr__(self) -> str:
        return "RAMFedAvgM"


class VRAMFedProx(VRAMBasedClientSelection, FedProx):
    """FedProx with RAM-based client selection."""

    def __repr__(self) -> str:
        return "RAMFedProx"


class VRAMFedAdagrad(VRAMBasedClientSelection, FedAdagrad):
    """FedAdagrad with RAM-based client selection."""

    def __repr__(self) -> str:
        return "RAMFedAdagrad"


class VRAMFedAdam(VRAMBasedClientSelection, FedAdam):
    """FedAdam with RAM-based client selection."""

    def __repr__(self) -> str:
        return "RAMFedAdam"


class VRAMFedMedian(VRAMBasedClientSelection, FedMedian):
    """FedMedian with RAM-based client selection."""

    def __repr__(self) -> str:
        return "RAMFedMedian"