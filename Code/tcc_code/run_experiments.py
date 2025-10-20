"""
Script principal para executar experimentos de Federated Learning
Suporta: XGBoost, LightGBM, CatBoost
Estratégias: Cyclic, Bagging
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import sys
from pathlib import Path

# Adicionar diretório ao path
sys.path.append(str(Path(__file__).parent))

from common import DataProcessor, ExperimentLogger, save_metrics_to_file

# CONFIGURAÇÕES PADRÃO
DEFAULT_CONFIG = {
    "num_clients": 6,
    "sample_per_client": 8000,
    "num_server_rounds": 6,
    "num_local_boost_round": 20,
    "seed": 42
}


def run_xgboost(config: dict, train_method: str = "cyclic"):
    """Executa experimento com XGBoost"""
    from algorithms.xgboost_fl import run_xgboost_experiment

    print(f"\n{'='*80}")
    print(f"EXPERIMENTO: XGBOOST - {train_method.upper()}")
    print(f"{'='*80}\n")

    # Preparar dados
    data_processor = DataProcessor(
        num_clients=config['num_clients'],
        sample_per_client=config['sample_per_client'],
        seed=config['seed']
    )
    data_processor.load_and_prepare_data()

    # Executar experimento
    result = run_xgboost_experiment(
        data_processor=data_processor,
        num_clients=config['num_clients'],
        num_server_rounds=config['num_server_rounds'],
        num_local_boost_round=config['num_local_boost_round'],
        train_method=train_method,
        seed=config['seed']
    )

    return result


def run_lightgbm(config: dict, train_method: str = "cyclic"):
    """Executa experimento com LightGBM"""
    from algorithms.lightgbm_fl import run_lightgbm_experiment

    print(f"\n{'='*80}")
    print(f"EXPERIMENTO: LIGHTGBM - {train_method.upper()}")
    print(f"{'='*80}\n")

    # Preparar dados
    data_processor = DataProcessor(
        num_clients=config['num_clients'],
        sample_per_client=config['sample_per_client'],
        seed=config['seed']
    )
    data_processor.load_and_prepare_data()

    # Executar experimento
    result = run_lightgbm_experiment(
        data_processor=data_processor,
        num_clients=config['num_clients'],
        num_server_rounds=config['num_server_rounds'],
        num_local_boost_round=config['num_local_boost_round'],
        train_method=train_method,
        seed=config['seed']
    )

    return result


def run_catboost(config: dict, train_method: str = "cyclic"):
    """Executa experimento com CatBoost"""
    from algorithms.catboost_fl import run_catboost_experiment

    print(f"\n{'='*80}")
    print(f"EXPERIMENTO: CATBOOST - {train_method.upper()}")
    print(f"{'='*80}\n")

    # Preparar dados
    data_processor = DataProcessor(
        num_clients=config['num_clients'],
        sample_per_client=config['sample_per_client'],
        seed=config['seed']
    )
    data_processor.load_and_prepare_data()

    # Executar experimento
    result = run_catboost_experiment(
        data_processor=data_processor,
        num_clients=config['num_clients'],
        num_server_rounds=config['num_server_rounds'],
        num_local_boost_round=config['num_local_boost_round'],
        train_method=train_method,
        seed=config['seed']
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Federated Learning - Tree-based Models")
    parser.add_argument("--algorithm", type=str, choices=["xgboost", "lightgbm", "catboost", "all"],
                       default="xgboost", help="Algoritmo a executar")
    parser.add_argument("--strategy", type=str, choices=["cyclic", "bagging", "both"],
                       default="cyclic", help="Estratégia de agregação")
    parser.add_argument("--num-clients", type=int, default=6, help="Número de clientes")
    parser.add_argument("--num-rounds", type=int, default=6, help="Número de rodadas")
    parser.add_argument("--local-rounds", type=int, default=20, help="Rodadas locais de boosting")
    parser.add_argument("--samples", type=int, default=8000, help="Amostras por cliente")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Configuração
    config = {
        "num_clients": args.num_clients,
        "sample_per_client": args.samples,
        "num_server_rounds": args.num_rounds,
        "num_local_boost_round": args.local_rounds,
        "seed": args.seed
    }

    print(f"\n{'='*80}")
    print("FEDERATED LEARNING - TREE-BASED MODELS")
    print(f"{'='*80}")
    print(f"Configuração:")
    print(f"  Algoritmo: {args.algorithm}")
    print(f"  Estratégia: {args.strategy}")
    print(f"  Clientes: {config['num_clients']}")
    print(f"  Rodadas: {config['num_server_rounds']}")
    print(f"  Boosting local: {config['num_local_boost_round']}")
    print(f"  Amostras/cliente: {config['sample_per_client']}")
    print(f"{'='*80}\n")

    # Determinar quais algoritmos executar
    algorithms = []
    if args.algorithm == "all":
        algorithms = ["xgboost", "lightgbm", "catboost"]
    else:
        algorithms = [args.algorithm]

    # Determinar quais estratégias executar
    strategies = []
    if args.strategy == "both":
        strategies = ["cyclic", "bagging"]
    else:
        strategies = [args.strategy]

    # Executar experimentos
    all_results = {}

    for algorithm in algorithms:
        for strategy in strategies:
            experiment_name = f"{algorithm}_{strategy}"

            try:
                if algorithm == "xgboost":
                    result = run_xgboost(config, strategy)
                elif algorithm == "lightgbm":
                    result = run_lightgbm(config, strategy)
                elif algorithm == "catboost":
                    result = run_catboost(config, strategy)

                # Verificar se o resultado existe (pode ser um objeto History do Flower)
                if result is not None:
                    all_results[experiment_name] = result
                    print(f"\n✓ Experimento {experiment_name} concluído com sucesso!\n")

            except Exception as e:
                print(f"\n✗ Erro no experimento {experiment_name}: {e}\n")
                import traceback
                traceback.print_exc()

    # Salvar resultados
    if all_results:
        save_metrics_to_file(all_results, "federated_learning_results.json")
        print(f"\n{'='*80}")
        print("TODOS OS EXPERIMENTOS CONCLUÍDOS")
        print(f"Total de experimentos bem-sucedidos: {len(all_results)}")
        print(f"{'='*80}\n")
    else:
        print("\nNenhum experimento foi concluído com sucesso.")


if __name__ == "__main__":
    main()
