"""
MAIN - Federated Learning com Tree-Based Models
Execução unificada para XGBoost, LightGBM e CatBoost
Dataset: Veículos (dataset_fl/dataset/dataset_K400_seed42/)
IMPORTANTE: Usa TODOS os 115,511 dados distribuídos entre os clientes
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from common import DataProcessor
from algorithms.xgboost import run_xgboost_experiment
from algorithms.lightgbm import run_lightgbm_experiment
from algorithms.catboost import run_catboost_experiment

import os
import platform

def get_csv_path(relative_path):
    """Converte caminho relativo para absoluto, compatível com WSL e Windows"""
    if platform.system() == "Linux" and os.path.exists("/mnt/c"):
        base = "/mnt/c/Users/candi/OneDrive/Desktop/Federated-Learning"
    else:
        base = r"C:\Users\candi\OneDrive\Desktop\Federated-Learning"

    return os.path.join(base, relative_path)

TRAIN_CSV = get_csv_path("dataset_fl/dataset/dataset_K400_seed42/dataset_all_vehicles.csv")
VALIDATION_CSV = get_csv_path("dataset_fl/dataset/dataset_K400_seed42/dataset_validation_all_vehicles.csv")

DEFAULT_CONFIG = {
    "num_clients": 3,
    "num_server_rounds": 10,
    "num_local_boost_round": 20,
    "seed": 42,
    "use_all_data": True,
    "balance_strategy": None
}


# ============================================================================
# FUNÇÕES PRINCIPAIS
# ============================================================================

def run_single_experiment(algorithm: str, strategy: str, config: dict):
    """
    Executa um único experimento (algoritmo + estratégia)

    Args:
        algorithm: 'xgboost', 'lightgbm' ou 'catboost'
        strategy: 'cyclic' ou 'bagging'
        config: Dicionário de configuração

    Returns:
        Histórico de resultados ou None em caso de erro
    """
    experiment_name = f"{algorithm}_{strategy}"

    print(f"\n{'='*80}")
    print(f"EXPERIMENTO: {algorithm.upper()} - {strategy.upper()}")
    print(f"{'='*80}")
    print(f"Dataset: {TRAIN_CSV}")
    print(f"Validação: {VALIDATION_CSV}")
    print(f"Clientes: {config['num_clients']}")
    print(f"Rodadas: {config['num_server_rounds']}")
    print(f"Boosting local: {config['num_local_boost_round']}")
    print(f"Usar todos os dados: {'Sim (115,511 amostras)' if config.get('use_all_data', True) else 'Não'}")
    print(f"Balanceamento: {config.get('balance_strategy', 'Nenhum')}")
    print(f"{'='*80}\n")

    try:
        data_processor = DataProcessor(
            num_clients=config['num_clients'],
            seed=config['seed'],
            train_csv_path=TRAIN_CSV,
            validation_csv_path=VALIDATION_CSV,
            use_all_data=config.get('use_all_data', True),
            balance_strategy=config.get('balance_strategy', None)
        )
        data_processor.load_and_prepare_data()

        if algorithm == "xgboost":
            result = run_xgboost_experiment(
                data_processor=data_processor,
                num_clients=config['num_clients'],
                num_server_rounds=config['num_server_rounds'],
                num_local_boost_round=config['num_local_boost_round'],
                train_method=strategy,
                seed=config['seed']
            )
        elif algorithm == "lightgbm":
            result = run_lightgbm_experiment(
                data_processor=data_processor,
                num_clients=config['num_clients'],
                num_server_rounds=config['num_server_rounds'],
                num_local_boost_round=config['num_local_boost_round'],
                train_method=strategy,
                seed=config['seed']
            )
        elif algorithm == "catboost":
            result = run_catboost_experiment(
                data_processor=data_processor,
                num_clients=config['num_clients'],
                num_server_rounds=config['num_server_rounds'],
                num_local_boost_round=config['num_local_boost_round'],
                train_method=strategy,
                seed=config['seed']
            )
        else:
            print(f"[ERRO] Algoritmo desconhecido: {algorithm}")
            return None

        print(f"\n[OK] Experimento {experiment_name} concluído com sucesso!\n")
        return result

    except Exception as e:
        print(f"\n[ERRO] Erro no experimento {experiment_name}: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def run_all_experiments(algorithms: list, strategies: list, config: dict):
    """
    Executa múltiplos experimentos sequencialmente

    Args:
        algorithms: Lista de algoritmos a executar
        strategies: Lista de estratégias a executar
        config: Configuração dos experimentos

    Returns:
        Dicionário com resultados de todos os experimentos
    """
    all_results = {}
    total_experiments = len(algorithms) * len(strategies)
    current_experiment = 0

    print(f"\n{'='*80}")
    print("EXECUTANDO EXPERIMENTOS DE FEDERATED LEARNING")
    print(f"{'='*80}")
    print(f"Total de experimentos: {total_experiments}")
    print(f"Algoritmos: {', '.join(algorithms)}")
    print(f"Estratégias: {', '.join(strategies)}")
    print(f"{'='*80}\n")

    for algorithm in algorithms:
        for strategy in strategies:
            current_experiment += 1
            experiment_name = f"{algorithm}_{strategy}"

            print(f"\n{'*'*80}")
            print(f"EXPERIMENTO {current_experiment}/{total_experiments}: {experiment_name.upper()}")
            print(f"{'*'*80}\n")

            result = run_single_experiment(algorithm, strategy, config)

            if result is not None and isinstance(result, dict) and result.get("success", False):
                all_results[experiment_name] = result
                print(f"[OK] {experiment_name} concluído ({current_experiment}/{total_experiments})")
            else:
                print(f"[ERRO] {experiment_name} falhou ({current_experiment}/{total_experiments})")

    return all_results


# ============================================================================
# INTERFACE DE LINHA DE COMANDO
# ============================================================================

def main():
    """Função principal com interface CLI"""
    parser = argparse.ArgumentParser(
        description="Federated Learning com Tree-Based Models (XGBoost, LightGBM, CatBoost)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Executar XGBoost com Cyclic
  python main.py --algorithm xgboost --strategy cyclic

  # Executar todos os algoritmos com ambas estratégias
  python main.py --algorithm all --strategy both

  # Executar LightGBM com Bagging (6 rodadas, 4 clientes)
  python main.py --algorithm lightgbm --strategy bagging --num-rounds 6 --num-clients 4

  # Executar com balanceamento SMOTE
  python main.py --algorithm xgboost --strategy cyclic --balance smote

  # Executar com class weights (recomendado para tree-based)
  python main.py --algorithm xgboost --strategy cyclic --balance weights

IMPORTANTE:
- Por padrão, usa TODOS os 115,511 dados do dataset distribuídos entre os clientes.
- Balanceamento de classes disponível: oversample, smote, undersample, weights
- Para usar SMOTE/oversample/undersample, instale: pip install imbalanced-learn
        """
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["xgboost", "lightgbm", "catboost", "all"],
        default="xgboost",
        help="Algoritmo a executar (padrão: xgboost)"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["cyclic", "bagging", "both"],
        default="cyclic",
        help="Estratégia de agregação (padrão: cyclic)"
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=DEFAULT_CONFIG["num_clients"],
        help=f"Número de clientes (padrão: {DEFAULT_CONFIG['num_clients']})"
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=DEFAULT_CONFIG["num_server_rounds"],
        help=f"Número de rodadas globais (padrão: {DEFAULT_CONFIG['num_server_rounds']})"
    )

    parser.add_argument(
        "--local-rounds",
        type=int,
        default=DEFAULT_CONFIG["num_local_boost_round"],
        help=f"Rodadas locais de boosting (padrão: {DEFAULT_CONFIG['num_local_boost_round']})"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help=f"Random seed (padrão: {DEFAULT_CONFIG['seed']})"
    )

    parser.add_argument(
        "--balance",
        type=str,
        choices=["oversample", "smote", "undersample", "weights"],
        default=None,
        help="Estratégia de balanceamento de classes (padrão: sem balanceamento)"
    )

    args = parser.parse_args()

    config = {
        "num_clients": args.num_clients,
        "num_server_rounds": args.num_rounds,
        "num_local_boost_round": args.local_rounds,
        "seed": args.seed,
        "use_all_data": True,
        "balance_strategy": args.balance
    }

    if args.algorithm == "all":
        algorithms = ["xgboost", "lightgbm", "catboost"]
    else:
        algorithms = [args.algorithm]

    if args.strategy == "both":
        strategies = ["cyclic", "bagging"]
    else:
        strategies = [args.strategy]

    print(f"\n{'='*80}")
    print("FEDERATED LEARNING - TREE-BASED MODELS (DATASET DE VEÍCULOS)")
    print(f"{'='*80}")
    print(f"Dataset de treino: {TRAIN_CSV}")
    print(f"Dataset de validação: {VALIDATION_CSV}")
    print(f"\nConfiguração:")
    print(f"  Algoritmos: {', '.join(algorithms)}")
    print(f"  Estratégias: {', '.join(strategies)}")
    print(f"  Clientes: {config['num_clients']}")
    print(f"  Rodadas globais: {config['num_server_rounds']}")
    print(f"  Boosting local: {config['num_local_boost_round']}")
    print(f"  Usar todos os dados: Sim (115,511 amostras)")
    print(f"  Seed: {config['seed']}")
    print(f"{'='*80}\n")

    all_results = run_all_experiments(algorithms, strategies, config)

    print(f"\n{'='*80}")
    print("RESUMO DOS EXPERIMENTOS")
    print(f"{'='*80}")
    print(f"Total de experimentos executados: {len(all_results)}")
    print(f"Experimentos bem-sucedidos:")
    for exp_name in all_results.keys():
        print(f"  [OK] {exp_name}")
    print(f"{'='*80}\n")

    if not all_results:
        print("[AVISO] Nenhum experimento foi concluído com sucesso.")
        return 1

    print("[SUCESSO] Experimentos concluídos! Verifique a pasta 'logs/' para resultados detalhados.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
