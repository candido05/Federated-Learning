"""Ponto de entrada principal para experimentos de Federated Learning.

Este script orquestra o processo de aprendizado federado, incluindo
carregamento de dataset, inicialização de clientes, configuração do servidor
e execução do treinamento.

Uso:
    python main.py
    python main.py --model xgboost --strategy FedAvg
    python main.py --num-clients 10 --num-rounds 10
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from config import GlobalConfig, LoggingConfig


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Configura o logging para a aplicação.

    Args:
        config: Objeto de configuração de logging.

    Returns:
        Instância do logger configurado.
    """
    # Cria o diretório de logs se não existir
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configura o formato de logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = logging.DEBUG if config.verbose else logging.INFO

    # Configura os handlers
    handlers = [logging.StreamHandler(sys.stdout)]

    if config.save_round_logs:
        file_handler = logging.FileHandler(log_dir / "experiment.log")
        handlers.append(file_handler)

    # Configura o logger raiz
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
    )

    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Faz o parsing dos argumentos de linha de comando.

    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Aprendizado Federado com Modelos Baseados em Árvores"
    )

    # Seleção do modelo
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "lightgbm", "catboost"],
        default="xgboost",
        help="Modelo a ser usado no aprendizado federado",
    )

    # Estratégia FL
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["FedAvg", "FedProx", "FedAdam", "FedAdagrad", "FedYogi", "FedMedian"],
        default="FedAvg",
        help="Estratégia de agregação do aprendizado federado",
    )

    # Configuração do experimento
    parser.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help="Número de clientes no aprendizado federado",
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Número de rodadas de comunicação",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Nome do dataset a ser usado",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semente aleatória para reprodutibilidade",
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Diretório para armazenar arquivos de log",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilita logging verboso",
    )

    return parser.parse_args()


def create_config(args: argparse.Namespace) -> tuple[GlobalConfig, LoggingConfig]:
    """Cria objetos de configuração a partir dos argumentos de linha de comando.

    Args:
        args: Argumentos de linha de comando parseados.

    Returns:
        Tupla com (GlobalConfig, LoggingConfig).
    """
    # Cria configuração global com sobrescritas da linha de comando
    global_config = GlobalConfig()

    if args.num_clients is not None:
        global_config.num_clients = args.num_clients
    if args.num_rounds is not None:
        global_config.num_server_rounds = args.num_rounds
    if args.dataset is not None:
        global_config.dataset_name = args.dataset
    if args.seed is not None:
        global_config.seed = args.seed

    # Cria configuração de logging
    logging_config = LoggingConfig()

    if args.log_dir is not None:
        logging_config.log_dir = args.log_dir
    if args.verbose:
        logging_config.verbose = True

    return global_config, logging_config


def run_experiment(
    model_name: str,
    strategy_name: str,
    global_config: GlobalConfig,
    logging_config: LoggingConfig,
) -> None:
    """Executa um único experimento de aprendizado federado.

    Args:
        model_name: Nome do modelo a ser usado.
        strategy_name: Nome da estratégia FL a ser usada.
        global_config: Configuração global do experimento.
        logging_config: Configuração de logging.
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Iniciando Experimento de Aprendizado Federado")
    logger.info("=" * 80)
    logger.info(f"Modelo: {model_name}")
    logger.info(f"Estratégia: {strategy_name}")
    logger.info(f"Dataset: {global_config.dataset_name}")
    logger.info(f"Número de clientes: {global_config.num_clients}")
    logger.info(f"Rodadas de comunicação: {global_config.num_server_rounds}")
    logger.info(f"Rodadas de boosting local: {global_config.num_local_boost_round}")
    logger.info(f"Semente aleatória: {global_config.seed}")
    logger.info("=" * 80)

    # TODO: Implementar lógica do experimento
    # 1. Carregar e particionar dataset
    # 2. Inicializar clientes
    # 3. Inicializar servidor com estratégia
    # 4. Executar aprendizado federado
    # 5. Coletar e salvar resultados

    logger.warning("Lógica do experimento ainda não implementada!")
    logger.info("Próximos passos:")
    logger.info("  1. Implementar carregamento de dados em data/")
    logger.info("  2. Implementar wrappers de modelos em models/")
    logger.info("  3. Implementar estratégias FL em strategies/")
    logger.info("  4. Implementar lógica do servidor em server/")
    logger.info("  5. Completar implementação de run_experiment()")


def main() -> None:
    """Ponto de entrada principal para a aplicação."""
    # Faz parsing dos argumentos de linha de comando
    args = parse_arguments()

    # Cria configurações
    global_config, logging_config = create_config(args)

    # Configura logging
    logger = setup_logging(logging_config)

    try:
        # Executa experimento
        run_experiment(
            model_name=args.model,
            strategy_name=args.strategy,
            global_config=global_config,
            logging_config=logging_config,
        )

        logger.info("Experimento concluído com sucesso!")

    except Exception as e:
        logger.error(f"Experimento falhou com erro: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
