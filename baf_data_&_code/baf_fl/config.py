"""
Configuracoes globais, dataclasses de registro e estado compartilhado do treinamento federado.
"""

import time
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


@dataclass
class FederatedConfig:
    """Configuracoes do treinamento federado."""
    num_clients: int = 3
    num_rounds: int = 15
    local_epochs: int = 15
    optuna_trials: int = 15
    fpr_target: float = 0.05
    random_state: int = 42


class GlobalState:
    """
    Singleton que gerencia o estado global compartilhado entre modulos.

    Necessario porque fl.simulation exige que client_fn acesse dados
    via estado global (nao aceita argumentos extras).
    """

    _instance: Optional['GlobalState'] = None

    def __new__(cls) -> 'GlobalState':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.client_data: Dict[int, Dict] = {}
        self.val_data: Dict[str, np.ndarray] = {}
        self.test_data: Dict[str, np.ndarray] = {}
        self.model_params: Dict[str, Dict] = {}
        self.scale_pos_weight: float = 1.0
        self.model_type: str = "xgboost"

    def set_client_data(self, client_data_list: list):
        """Popula dados dos clientes a partir de uma lista."""
        self.client_data = {i: data for i, data in enumerate(client_data_list)}

    def set_validation_data(self, X: np.ndarray, y: np.ndarray, age: np.ndarray):
        self.val_data = {'X': X, 'y': y, 'age': age}

    def set_test_data(self, X: np.ndarray, y: np.ndarray, age: np.ndarray):
        self.test_data = {'X': X, 'y': y, 'age': age}

    @classmethod
    def reset(cls):
        """Reseta a instancia singleton (util para testes)."""
        cls._instance = None


CONFIG = FederatedConfig()
global_state = GlobalState()


# --- Dataclasses de registro de experimentos ---

@dataclass
class ClientRoundRecord:
    """Metricas de um cliente em uma rodada."""
    round_num: int
    client_id: int
    train_tpr: float
    val_tpr: float
    training_time_sec: float
    model_size_bytes: int
    num_train_samples: int


@dataclass
class ServerRoundRecord:
    """Metricas do servidor em uma rodada."""
    round_num: int
    tpr_standard: float
    fairness_standard: float
    tpr_fair: float
    fairness_fair: float
    round_time_sec: float
    trained_client: Optional[int] = None
    num_client_models: Optional[int] = None
    total_bytes_this_round: int = 0


@dataclass
class SimulationRecord:
    """Registro completo de uma simulacao (modelo + estrategia)."""
    model_type: str
    strategy_type: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_sec: float = 0.0
    client_rounds: List[ClientRoundRecord] = field(default_factory=list)
    server_rounds: List[ServerRoundRecord] = field(default_factory=list)
    total_bytes_transferred: int = 0
    final_metrics_standard: Optional[Dict[str, float]] = None
    final_metrics_fair: Optional[Dict[str, float]] = None


class SimulationRecorder:
    """
    Coleta dados durante uma simulacao.

    A strategy chama record_client_round() em aggregate_fit()
    e record_server_round() em evaluate().
    runner.py chama start()/stop() ao redor da simulacao.
    """

    def __init__(self, model_type: str, strategy_type: str):
        self.record = SimulationRecord(
            model_type=model_type,
            strategy_type=strategy_type,
        )
        self._round_start_times: Dict[int, float] = {}

    def start(self):
        self.record.start_time = time.time()

    def stop(self):
        self.record.end_time = time.time()
        self.record.duration_sec = self.record.end_time - self.record.start_time

    def mark_round_start(self, server_round: int):
        """Marca inicio de uma rodada (chamado em configure_fit)."""
        self._round_start_times[server_round] = time.time()

    def record_client_round(
        self,
        round_num: int,
        client_id: int,
        train_tpr: float,
        val_tpr: float,
        training_time_sec: float,
        model_size_bytes: int,
        num_train_samples: int,
    ):
        self.record.client_rounds.append(ClientRoundRecord(
            round_num=round_num,
            client_id=client_id,
            train_tpr=train_tpr,
            val_tpr=val_tpr,
            training_time_sec=training_time_sec,
            model_size_bytes=model_size_bytes,
            num_train_samples=num_train_samples,
        ))
        self.record.total_bytes_transferred += model_size_bytes

    def record_server_round(
        self,
        round_num: int,
        tpr_standard: float,
        fairness_standard: float,
        tpr_fair: float,
        fairness_fair: float,
        trained_client: Optional[int] = None,
        num_client_models: Optional[int] = None,
        total_bytes_this_round: int = 0,
    ):
        round_start = self._round_start_times.get(round_num, 0)
        round_time = time.time() - round_start if round_start else 0.0

        self.record.server_rounds.append(ServerRoundRecord(
            round_num=round_num,
            tpr_standard=tpr_standard,
            fairness_standard=fairness_standard,
            tpr_fair=tpr_fair,
            fairness_fair=fairness_fair,
            round_time_sec=round_time,
            trained_client=trained_client,
            num_client_models=num_client_models,
            total_bytes_this_round=total_bytes_this_round,
        ))
