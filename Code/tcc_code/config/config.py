"""Configurações globais e de logging para experimentos de Federated Learning.

Este módulo define dataclasses de configuração para definições de experimentos
e comportamento de logging.
"""

from dataclasses import dataclass


@dataclass
class GlobalConfig:
    """Configuração global para experimentos de Federated Learning.

    Attributes:
        num_clients: Número de clientes no aprendizado federado.
        sample_per_client: Número de amostras alocadas para cada cliente.
        num_server_rounds: Número de rodadas de comunicação entre servidor e clientes.
        num_local_boost_round: Número de rodadas de boosting local por cliente.
        seed: Semente aleatória para reprodutibilidade.
        test_fraction: Fração dos dados para usar em testes.
        dataset_name: Nome do dataset a ser usado (ex: "higgs").
    """

    num_clients: int = 6
    sample_per_client: int = 8000
    num_server_rounds: int = 6
    num_local_boost_round: int = 20
    seed: int = 42
    test_fraction: float = 0.2
    dataset_name: str = "higgs"

    def __post_init__(self) -> None:
        """Valida os parâmetros de configuração após a inicialização."""
        if self.num_clients <= 0:
            raise ValueError("num_clients deve ser positivo")
        if self.sample_per_client <= 0:
            raise ValueError("sample_per_client deve ser positivo")
        if self.num_server_rounds <= 0:
            raise ValueError("num_server_rounds deve ser positivo")
        if self.num_local_boost_round <= 0:
            raise ValueError("num_local_boost_round deve ser positivo")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction deve estar entre 0 e 1")


@dataclass
class LoggingConfig:
    """Configuração para o comportamento de logging durante experimentos.

    Attributes:
        log_dir: Diretório para armazenar arquivos de log.
        save_client_logs: Se deve salvar logs individuais dos clientes.
        save_round_logs: Se deve salvar logs de cada rodada.
        verbose: Se deve imprimir logs detalhados no console.
    """

    log_dir: str = "logs"
    save_client_logs: bool = True
    save_round_logs: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        """Valida a configuração de logging após a inicialização."""
        if not self.log_dir:
            raise ValueError("log_dir não pode ser vazio")
