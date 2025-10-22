#!/usr/bin/env python3
"""
Script mestre para executar TODOS os experimentos
XGBoost, LightGBM e CatBoost com estratégias Cyclic e Bagging
"""

import sys
import time
from common import DataProcessor
from algorithms import run_xgboost_experiment, run_lightgbm_experiment, run_catboost_experiment


def main():
    """Executa todos os experimentos de todos os algoritmos"""

    # Configurações do experimento
    NUM_CLIENTS = 6
    NUM_SERVER_ROUNDS = 3
    NUM_LOCAL_BOOST_ROUND = 20
    SAMPLE_PER_CLIENT = 8000
    SEED = 42

    print("\n" + "="*80)
    print("EXECUÇÃO COMPLETA DE TODOS OS EXPERIMENTOS")
    print("XGBoost, LightGBM, CatBoost × Cyclic, Bagging = 6 experimentos")
    print("="*80)
    print(f"\nConfiguração:")
    print(f"  - Número de clientes: {NUM_CLIENTS}")
    print(f"  - Rodadas globais: {NUM_SERVER_ROUNDS}")
    print(f"  - Rodadas locais: {NUM_LOCAL_BOOST_ROUND}")
    print(f"  - Amostras por cliente: {SAMPLE_PER_CLIENT}")
    print(f"  - Seed: {SEED}")
    print("="*80)

    # Preparar dados (uma única vez para todos os experimentos)
    print("\n📊 Carregando e preparando dataset HIGGS...")
    data_processor = DataProcessor(
        num_clients=NUM_CLIENTS,
        sample_per_client=SAMPLE_PER_CLIENT,
        seed=SEED
    )
    data_processor.load_and_prepare_data()
    print("✅ Dataset preparado!\n")

    # Definir todos os experimentos
    experiments = [
        ("xgboost", "cyclic", run_xgboost_experiment),
        ("xgboost", "bagging", run_xgboost_experiment),
        ("lightgbm", "cyclic", run_lightgbm_experiment),
        ("lightgbm", "bagging", run_lightgbm_experiment),
        ("catboost", "cyclic", run_catboost_experiment),
        ("catboost", "bagging", run_catboost_experiment),
    ]

    results = {}
    failed = []
    start_time_total = time.time()

    # Executar todos os experimentos
    for idx, (algorithm, strategy, run_func) in enumerate(experiments, 1):
        exp_name = f"{algorithm}_{strategy}"

        print(f"\n{'#'*80}")
        print(f"# EXPERIMENTO {idx}/{len(experiments)}: {algorithm.upper()} - {strategy.upper()}")
        print(f"{'#'*80}\n")

        try:
            exp_start = time.time()

            result = run_func(
                data_processor=data_processor,
                num_clients=NUM_CLIENTS,
                num_server_rounds=NUM_SERVER_ROUNDS,
                num_local_boost_round=NUM_LOCAL_BOOST_ROUND,
                train_method=strategy,
                seed=SEED
            )

            exp_elapsed = time.time() - exp_start

            if result is not None:
                results[exp_name] = result
                print(f"\n✅ Experimento {exp_name} CONCLUÍDO! ({exp_elapsed:.2f}s)")
            else:
                failed.append(exp_name)
                print(f"\n❌ Experimento {exp_name} falhou (resultado None)")

        except Exception as e:
            failed.append(exp_name)
            print(f"\n❌ ERRO no experimento {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    # Resumo final
    elapsed_total = time.time() - start_time_total

    print("\n\n" + "="*80)
    print("RESUMO FINAL DA EXECUÇÃO")
    print("="*80)
    print(f"\n✅ Experimentos concluídos: {len(results)}/{len(experiments)}")
    print(f"❌ Experimentos falhados: {len(failed)}/{len(experiments)}")
    print(f"⏱️  Tempo total: {elapsed_total:.2f} segundos ({elapsed_total/60:.2f} minutos)")

    if results:
        print("\n📊 Experimentos executados com sucesso:")
        for exp_name in sorted(results.keys()):
            print(f"   ✓ {exp_name}")

        print("\n📁 Estrutura de logs:")
        print("   logs/")
        print("   ├── xgboost/")
        print("   │   ├── YYYYMMDD_HHMMSS_cyclic/")
        print("   │   └── YYYYMMDD_HHMMSS_bagging/")
        print("   ├── lightgbm/")
        print("   │   ├── YYYYMMDD_HHMMSS_cyclic/")
        print("   │   └── YYYYMMDD_HHMMSS_bagging/")
        print("   └── catboost/")
        print("       ├── YYYYMMDD_HHMMSS_cyclic/")
        print("       └── YYYYMMDD_HHMMSS_bagging/")
        print("\n   Cada pasta contém:")
        print("   - execution_log.txt: Log completo com métricas")
        print("   - metrics.json: Dados estruturados em JSON")
        print("   - README.md: Resumo do experimento")

    if failed:
        print(f"\n❌ Experimentos falhados:")
        for exp_name in failed:
            print(f"   ✗ {exp_name}")

    print("\n" + "="*80)
    if len(results) == len(experiments):
        print("🎉 TODOS OS EXPERIMENTOS CONCLUÍDOS COM SUCESSO!")
    elif len(results) > 0:
        print(f"⚠️  EXECUÇÃO PARCIAL: {len(results)}/{len(experiments)} experimentos concluídos")
    else:
        print("❌ FALHA TOTAL: Nenhum experimento foi concluído!")
        return 1
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
