"""
Ponto de entrada para o treinamento federado de deteccao de fraude bancaria.

Uso:
    python -m baf_fl.main
    python -m baf_fl.main --data-path ../datasets/Base.csv
    python -m baf_fl.main --num-clients 5 --num-rounds 10
"""

import argparse
import os
import sys
import warnings

# Suprimir mensagens do Ray e deprecation do Flower
# DEVE ser setado antes de qualquer import de Ray/Flower
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_ENABLE_METRICS_AGENT"] = "0"
os.environ["RAY_METRICS_AGENT_ENABLED"] = "0"
os.environ["RAY_BACKEND_LOG_LEVEL"] = "fatal"
os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["RAY_verbose_spill_logs"] = "0"
warnings.filterwarnings("ignore", message=".*DEPRECATED FEATURE.*")
warnings.filterwarnings("ignore", message=".*client_fn.*")

import numpy as np
import pandas as pd

from .config import CONFIG
from .paths import DEFAULT_DATASET_PATH, EXPERIMENTS_DIR, IMAGES_DIR
from .data.processing import DataPreprocessor, DataPartitioner
from .tuning.optimization import federated_hyperparameter_optimization
from .core.runner import run_federated_training
from .reporting.plots import PlotGenerator
from .reporting.experiment_logger import ExperimentLogger
from .reporting.tcc_plots import generate_tcc_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Learning - Bank Account Fraud Detection",
    )
    parser.add_argument(
        '--data-path', type=str,
        default=DEFAULT_DATASET_PATH,
        help='Caminho para o dataset Base.csv',
    )
    parser.add_argument('--num-clients', type=int, default=3, help='Numero de clientes federados')
    parser.add_argument('--num-rounds', type=int, default=CONFIG.num_rounds, help='Numero de rounds federados')
    parser.add_argument('--optuna-trials', type=int, default=CONFIG.optuna_trials, help='Trials Optuna por cliente')
    parser.add_argument('--sample-fraction', type=float, default=1/3, help='Fracao dos dados para Optuna')
    parser.add_argument('--max-lr', type=float, default=0.05, help='Learning rate maximo (freio de seguranca)')
    return parser.parse_args()


def print_banner():
    print("#" * 80)
    print("#" + " " * 20 + "FEDERATED LEARNING COM FLOWER" + " " * 20 + "#")
    print("#" + " " * 15 + "BANK ACCOUNT FRAUD DETECTION (CORRIGIDO)" + " " * 15 + "#")
    print("#" * 80)
    print("\nCorrecoes implementadas:")
    print("  1. fl.simulation.start_simulation (motor oficial)")
    print("  2. Warm Start para Cycling (transferencia de conhecimento)")
    print("  3. Post-Processing com Limiares por Grupo (Fairness = 1.0)")
    print("  4. Serializacao correta: parameters_to_ndarrays / ndarrays_to_parameters")
    print("  5. Optuna FEDERADO: cada cliente otimiza localmente, parametros agregados")


def print_report(df_results: pd.DataFrame):
    print("\n" + "=" * 80)
    print("RELATORIO FINAL")
    print("=" * 80)

    if len(df_results) == 0:
        print("AVISO: Nenhum resultado obtido!")
        return

    print("\n--- Threshold Unico ---")
    df_single = df_results[df_results['threshold'] == 'single']
    for _, row in df_single.iterrows():
        print(f"  {row['model']}: TPR={row['tpr_at_5fpr']:.4f}, Fairness={row['fairness_ratio']:.4f}")

    print("\n--- Threshold por Grupo (Fair) ---")
    df_fair = df_results[df_results['threshold'] == 'per_group']
    for _, row in df_fair.iterrows():
        print(f"  {row['model']}: TPR={row['tpr_at_5fpr']:.4f}, Fairness={row['fairness_ratio']:.4f}")

    print("\n" + "-" * 60)
    print("BENCHMARK (TPR > 0.52, Fairness ~ 1.0)")
    print("-" * 60)

    for _, row in df_results.iterrows():
        tpr = row['tpr_at_5fpr']
        fr = row['fairness_ratio']
        tpr_ok = "[OK]" if tpr > 0.52 else "[X]"
        fr_ok = "[OK]" if 0.9 <= fr <= 1.1 else "[X]"
        print(f"{tpr_ok} {fr_ok} {row['model']}: TPR={tpr:.4f}, Fairness={fr:.4f}")


def main():
    args = parse_args()

    # Atualizar configuracao global
    CONFIG.num_clients = args.num_clients
    CONFIG.num_rounds = args.num_rounds

    print_banner()

    # Criar logger de experimento
    experiment_logger = ExperimentLogger(output_dir=EXPERIMENTS_DIR)
    experiment_logger.start_experiment()
    experiment_logger.set_config(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=CONFIG.local_epochs,
        fpr_target=CONFIG.fpr_target,
        random_state=CONFIG.random_state,
        optuna_trials=args.optuna_trials,
        sample_fraction=args.sample_fraction,
        max_learning_rate=args.max_lr,
    )

    # Pre-processamento
    preprocessor = DataPreprocessor()

    df = preprocessor.load_and_clean(args.data_path)
    df = preprocessor.create_features(df)
    df_train, df_val, df_test = preprocessor.temporal_split(df)

    (X_train, y_train, age_train,
     X_val, y_val, age_val,
     X_test, y_test, age_test) = preprocessor.encode_features(df_train, df_val, df_test)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"\nScale pos weight: {scale_pos_weight:.2f}")

    # Particionamento
    client_data = DataPartitioner.partition_balanced(
        X_train, y_train, age_train, CONFIG.num_clients,
    )

    # Registrar informacoes do dataset
    client_distributions = []
    for i, cd in enumerate(client_data):
        n_fraud = int(cd['y'].sum())
        n_total = len(cd['y'])
        client_distributions.append({
            "client_id": i,
            "total_samples": n_total,
            "fraud_samples": n_fraud,
            "fraud_rate": round(n_fraud / n_total, 6),
        })

    experiment_logger.set_dataset_info(
        total_samples=len(y_train) + len(y_val) + len(y_test),
        num_features=X_train.shape[1],
        train_size=len(y_train),
        val_size=len(y_val),
        test_size=len(y_test),
        fraud_rate_train=float(y_train.mean()),
        fraud_rate_val=float(y_val.mean()),
        fraud_rate_test=float(y_test.mean()),
        scale_pos_weight=float(scale_pos_weight),
        client_distributions=client_distributions,
    )

    # Otimizacao federada de hiperparametros
    best_params = federated_hyperparameter_optimization(
        client_data=client_data,
        X_val=X_val,
        y_val=y_val,
        scale_pos_weight=scale_pos_weight,
        n_trials_per_client=args.optuna_trials,
        sample_fraction=args.sample_fraction,
        max_learning_rate=args.max_lr,
    )

    # Registrar hiperparametros
    experiment_logger.set_hyperparameters(best_params)

    # Treinamento federado
    df_results, strategies_trained = run_federated_training(
        client_data, X_val, y_val, age_val,
        X_test, y_test, age_test,
        best_params, scale_pos_weight,
        experiment_logger=experiment_logger,
    )

    # Relatorio final
    print_report(df_results)

    # Geracao de graficos (pasta com timestamp para cada execucao)
    plot_dir = os.path.join(IMAGES_DIR, experiment_logger.experiment_id)
    plot_generator = PlotGenerator(output_dir=plot_dir)
    plot_generator.generate_all(df_results, strategies_trained)

    # Salvar registro completo do experimento
    experiment_logger.set_final_results(df_results)
    experiment_logger.end_experiment()
    experiment_logger.save()

    # Gerar graficos TCC a partir dos CSVs salvos (execucao mais recente)
    experiment_data_dir = os.path.join(EXPERIMENTS_DIR, experiment_logger.experiment_id)
    tcc_plot_dir = os.path.join(IMAGES_DIR, experiment_logger.experiment_id)
    generate_tcc_plots(experiment_data_dir, tcc_plot_dir)

    return df_results


if __name__ == "__main__":
    results = main()
