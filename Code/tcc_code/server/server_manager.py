"""
Gerenciador do servidor Federated Learning.

Coordena experimentos FL, incluindo setup, criação de estratégias e execução.
"""

import logging
from typing import Dict, Optional, Any, Callable

from flwr.server import ServerConfig

from config import GlobalConfig, LoggingConfig
from config.model_params import XGBOOST_PARAMS, CATBOOST_PARAMS, LIGHTGBM_PARAMS
from strategies import FedBagging, FedCyclic, BaseStrategy
from models import XGBoostClient, CatBoostClient, LightGBMClient
from data.dataset_factory import create_dataset
from utils.logging_utils import ExperimentLogger
from .evaluation import get_evaluate_fn


class FederatedServer:
    """
    Gerenciador do servidor Federated Learning.

    Coordena setup, criação de estratégias, clientes e execução de experimentos.

    Attributes:
        config: Configuração global do experimento.
        logging_config: Configuração de logging.
        logger: Logger para eventos do servidor.
        experiment_logger: Logger de experimentos.
        dataset: Dataset carregado.
        current_model_type: Tipo de modelo atual.
        current_strategy_type: Tipo de estratégia atual.
    """

    def __init__(
        self,
        config: GlobalConfig,
        logging_config: LoggingConfig,
    ):
        """
        Inicializa servidor federado.

        Args:
            config: Configuração global.
            logging_config: Configuração de logging.
        """
        self.config = config
        self.logging_config = logging_config

        # Logger básico
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Atributos de experimento
        self.experiment_logger: Optional[ExperimentLogger] = None
        self.dataset = None
        self.current_model_type: Optional[str] = None
        self.current_strategy_type: Optional[str] = None

        self.logger.info("Servidor Federado inicializado")
        self.logger.info(f"Configuração: {config.num_clients} clientes, {config.num_rounds} rounds")

    def setup_experiment(
        self,
        model_type: str,
        strategy_type: str,
        dataset_source: str = "jxie/higgs",
    ) -> None:
        """
        Configura experimento FL (logger, dataset, estratégia).

        Args:
            model_type: Tipo do modelo ('xgboost', 'catboost', 'lightgbm').
            strategy_type: Tipo da estratégia ('bagging', 'cyclic').
            dataset_source: Fonte do dataset HuggingFace.

        Raises:
            ValueError: Se model_type ou strategy_type inválidos.
        """
        # Valida inputs
        valid_models = ['xgboost', 'catboost', 'lightgbm']
        valid_strategies = ['bagging', 'cyclic']

        if model_type.lower() not in valid_models:
            raise ValueError(
                f"Modelo '{model_type}' inválido. Use: {valid_models}"
            )

        if strategy_type.lower() not in valid_strategies:
            raise ValueError(
                f"Estratégia '{strategy_type}' inválida. Use: {valid_strategies}"
            )

        self.current_model_type = model_type.lower()
        self.current_strategy_type = strategy_type.lower()

        self.logger.info(f"Configurando experimento: {model_type} + {strategy_type}")

        # Configura logger de experimento
        self.experiment_logger = ExperimentLogger(
            experiment_name=f"{model_type}_{strategy_type}",
            model_name=model_type,
            strategy_name=strategy_type,
            num_clients=self.config.num_clients,
            base_dir=self.logging_config.base_dir,
        )

        self.experiment_logger.logger.info(
            f"Experimento configurado: {model_type} + {strategy_type}"
        )

        # Carrega dataset
        self.logger.info(f"Carregando dataset: {dataset_source}")
        self.dataset = create_dataset(
            dataset_type="tabular",
            config=self.config,
            dataset_source=dataset_source,
        )

        self.dataset.prepare()
        self.experiment_logger.logger.info(
            f"Dataset carregado: {len(self.dataset.X_train)} amostras de treino, "
            f"{len(self.dataset.X_test)} amostras de teste"
        )

    def create_strategy(
        self,
        strategy_type: str,
        evaluate_fn: Optional[Callable] = None,
        params: Optional[Dict] = None,
    ) -> BaseStrategy:
        """
        Cria estratégia de agregação FL.

        Args:
            strategy_type: Tipo da estratégia ('bagging', 'cyclic').
            evaluate_fn: Função de avaliação centralizada.
            params: Parâmetros adicionais para estratégia.

        Returns:
            Estratégia configurada (FedBagging ou FedCyclic).

        Raises:
            ValueError: Se strategy_type não suportado.
        """
        if params is None:
            params = {}

        self.logger.info(f"Criando estratégia: {strategy_type}")

        if strategy_type.lower() == 'bagging':
            strategy = FedBagging(
                fraction_fit=params.get('fraction_fit', 1.0),
                fraction_evaluate=params.get('fraction_evaluate', 1.0),
                min_fit_clients=params.get('min_fit_clients', self.config.num_clients),
                min_evaluate_clients=params.get('min_evaluate_clients', 2),
                min_available_clients=params.get('min_available_clients', self.config.num_clients),
                evaluate_fn=evaluate_fn,
            )

        elif strategy_type.lower() == 'cyclic':
            strategy = FedCyclic(
                fraction_evaluate=params.get('fraction_evaluate', 1.0),
                min_evaluate_clients=params.get('min_evaluate_clients', 1),
                min_available_clients=params.get('min_available_clients', 1),
                evaluate_fn=evaluate_fn,
            )

        else:
            raise ValueError(
                f"Estratégia '{strategy_type}' não suportada. Use: bagging, cyclic"
            )

        self.logger.info(f"Estratégia {strategy_type} criada com sucesso")
        return strategy

    def create_client_fn(
        self,
        model_type: str,
        dataset,
        params: Optional[Dict] = None,
    ) -> Callable[[str], Any]:
        """
        Cria função client_fn compatível com Flower.

        Args:
            model_type: Tipo do modelo ('xgboost', 'catboost', 'lightgbm').
            dataset: Dataset com partições de clientes.
            params: Parâmetros do modelo.

        Returns:
            Função client_fn(cid: str) que cria clientes FL.

        Raises:
            ValueError: Se model_type não suportado.
        """
        if params is None:
            # Usa parâmetros padrão baseado no tipo
            if model_type.lower() == 'xgboost':
                params = XGBOOST_PARAMS
            elif model_type.lower() == 'catboost':
                params = CATBOOST_PARAMS
            elif model_type.lower() == 'lightgbm':
                params = LIGHTGBM_PARAMS
            else:
                raise ValueError(f"Modelo '{model_type}' não suportado")

        # Número de rounds locais
        num_local_round = self.config.num_local_rounds

        def client_fn(cid: str):
            """
            Cria cliente FL para ID específico.

            Args:
                cid: ID do cliente (string numérica).

            Returns:
                Cliente FL configurado.
            """
            # Converte cid para índice
            client_idx = int(cid)

            # Obtém dados do cliente
            train_data, valid_data = dataset.get_client_data(client_idx)

            # Cria cliente baseado no tipo
            if model_type.lower() == 'xgboost':
                return XGBoostClient(
                    train_data=train_data,
                    valid_data=valid_data,
                    num_local_round=num_local_round,
                    params=params,
                    client_id=cid,
                    experiment_logger=self.experiment_logger,
                )

            elif model_type.lower() == 'catboost':
                return CatBoostClient(
                    train_data=train_data,
                    valid_data=valid_data,
                    num_local_round=num_local_round,
                    params=params,
                    client_id=cid,
                    experiment_logger=self.experiment_logger,
                )

            elif model_type.lower() == 'lightgbm':
                return LightGBMClient(
                    train_data=train_data,
                    valid_data=valid_data,
                    num_local_round=num_local_round,
                    params=params,
                    client_id=cid,
                    experiment_logger=self.experiment_logger,
                )

            else:
                raise ValueError(f"Modelo '{model_type}' não suportado")

        return client_fn

    def run_experiment(
        self,
        model_type: str,
        strategy_type: str,
        dataset_source: str = "jxie/higgs",
    ) -> Dict[str, Any]:
        """
        Executa experimento FL completo.

        Args:
            model_type: Tipo do modelo ('xgboost', 'catboost', 'lightgbm').
            strategy_type: Tipo da estratégia ('bagging', 'cyclic').
            dataset_source: Fonte do dataset.

        Returns:
            Dicionário com histórico de métricas e resultados.

        Raises:
            Exception: Se experimento falhar.
        """
        try:
            # Setup do experimento
            self.setup_experiment(model_type, strategy_type, dataset_source)

            # Cria função de avaliação centralizada
            test_data = (self.dataset.X_test, self.dataset.y_test)
            evaluate_fn = get_evaluate_fn(
                test_data=test_data,
                model_type=model_type,
            )

            # Cria estratégia
            strategy = self.create_strategy(
                strategy_type=strategy_type,
                evaluate_fn=evaluate_fn,
            )

            # Cria função de criação de clientes
            client_fn = self.create_client_fn(
                model_type=model_type,
                dataset=self.dataset,
            )

            # Configura servidor
            server_config = ServerConfig(num_rounds=self.config.num_rounds)

            # Detecta GPU e configura backend
            backend_config = self._detect_gpu_and_configure()

            self.logger.info(
                f"Iniciando simulação FL: {self.config.num_rounds} rounds, "
                f"{self.config.num_clients} clientes"
            )

            # Executa simulação com fallback para diferentes versões do Flower
            history = self._safe_run_simulation(
                client_fn=client_fn,
                num_clients=self.config.num_clients,
                config=server_config,
                strategy=strategy,
                backend_config=backend_config,
            )

            self.logger.info("Simulação FL concluída com sucesso")

            # Salva sumário do experimento
            if self.experiment_logger:
                self.experiment_logger.save_summary()

            # Prepara resultados
            results = {
                "history": history,
                "model_type": model_type,
                "strategy_type": strategy_type,
                "num_rounds": self.config.num_rounds,
                "num_clients": self.config.num_clients,
            }

            return results

        except Exception as e:
            error_msg = f"Erro ao executar experimento {model_type}+{strategy_type}: {e}"
            self.logger.error(error_msg)
            if self.experiment_logger:
                self.experiment_logger.logger.error(error_msg)
            raise

    def _safe_run_simulation(
        self,
        client_fn: Callable,
        num_clients: int,
        config: ServerConfig,
        strategy: BaseStrategy,
        backend_config: Optional[Dict] = None,
    ) -> Any:
        """
        Executa simulação FL com fallback para diferentes versões do Flower.

        Args:
            client_fn: Função de criação de clientes.
            num_clients: Número de clientes.
            config: Configuração do servidor.
            strategy: Estratégia de agregação.
            backend_config: Configuração de backend (GPU/CPU).

        Returns:
            History object do Flower.
        """
        try:
            # Tenta importar versão mais recente (com run_simulation)
            from flwr.simulation import start_simulation

            self.logger.info("Usando flwr.simulation.start_simulation")

            if backend_config:
                history = start_simulation(
                    client_fn=client_fn,
                    num_clients=num_clients,
                    config=config,
                    strategy=strategy,
                    client_resources=backend_config.get("client_resources"),
                )
            else:
                history = start_simulation(
                    client_fn=client_fn,
                    num_clients=num_clients,
                    config=config,
                    strategy=strategy,
                )

            return history

        except ImportError:
            # Fallback para versão antiga
            self.logger.warning(
                "start_simulation não disponível, tentando fallback"
            )

            try:
                from flwr.simulation import start_simulation as legacy_start

                self.logger.info("Usando método legado de simulação")

                history = legacy_start(
                    client_fn=client_fn,
                    num_clients=num_clients,
                    num_rounds=config.num_rounds,
                    strategy=strategy,
                )

                return history

            except Exception as e:
                self.logger.error(f"Fallback falhou: {e}")
                raise

    def _detect_gpu_and_configure(self) -> Optional[Dict]:
        """
        Detecta GPU disponível e configura backend apropriado.

        Returns:
            Dicionário de configuração de backend ou None.
        """
        try:
            import subprocess

            # Tenta detectar GPU NVIDIA
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split('\n')[0]
                self.logger.info(f"GPU detectada: {gpu_info}")

                # Configura recursos do cliente para usar GPU
                backend_config = {
                    "client_resources": {
                        "num_cpus": 1,
                        "num_gpus": 0.2,  # Fração de GPU por cliente
                    }
                }

                return backend_config

        except Exception as e:
            self.logger.info(f"GPU não detectada ou nvidia-smi não disponível: {e}")

        # Fallback para CPU
        self.logger.info("Usando CPU para simulação")
        backend_config = {
            "client_resources": {
                "num_cpus": 2,
                "num_gpus": 0.0,
            }
        }

        return backend_config

    def _get_model_params(self, model_type: str) -> Dict:
        """
        Obtém parâmetros padrão para tipo de modelo.

        Args:
            model_type: Tipo do modelo.

        Returns:
            Dicionário de parâmetros.
        """
        if model_type.lower() == 'xgboost':
            return XGBOOST_PARAMS
        elif model_type.lower() == 'catboost':
            return CATBOOST_PARAMS
        elif model_type.lower() == 'lightgbm':
            return LIGHTGBM_PARAMS
        else:
            return {}
