#!/usr/bin/env python3
"""
Script para executar todos os experimentos do CatBoost
Executa tanto estratégia Cyclic quanto Bagging
"""

import sys
import time
from common import DataProcessor
from algorithms import run_catboost_experiment


def main():
    """Executa todos os experimentos do CatBoost"""

    # Configurações do experimento
    NUM_CLIENTS = 6
    NUM_SERVER_ROUNDS = 3
    NUM_LOCAL_BOOST_ROUND = 20
    SAMPLE_PER_CLIENT = 8000
    SEED = 42

    print("="*80)
    print("EXECUÇÃO COMPLETA DE EXPERIMENTOS CATBOOST")
    print("="*80)
    print(f"\nConfiguração:")
    print(f"  - Número de clientes: {NUM_CLIENTS}")
    print(f"  - Rodadas globais: {NUM_SERVER_ROUNDS}")
    print(f"  - Rodadas locais: {NUM_LOCAL_BOOST_ROUND}")
    print(f"  - Amostras por cliente: {SAMPLE_PER_CLIENT}")
    print(f"  - Seed: {SEED}")
    print("="*80)

    # Preparar dados (uma única vez)
    print("\n📊 Carregando e preparando dataset HIGGS...")
    data_processor = DataProcessor(
        num_clients=NUM_CLIENTS,
        sample_per_client=SAMPLE_PER_CLIENT,
        seed=SEED
    )
    data_processor.load_and_prepare_data()
    print("✅ Dataset preparado!\n")

    # Estratégias para executar
    strategies = ["cyclic", "bagging"]
    results = {}

    start_time_total = time.time()

    # Executar cada estratégia
    for strategy in strategies:
        print(f"\n{'#'*80}")
        print(f"# INICIANDO: CatBoost - {strategy.upper()}")
        print(f"{'#'*80}\n")

        try:
            result = run_catboost_experiment(
                data_processor=data_processor,
                num_clients=NUM_CLIENTS,
                num_server_rounds=NUM_SERVER_ROUNDS,
                num_local_boost_round=NUM_LOCAL_BOOST_ROUND,
                train_method=strategy,
                seed=SEED
            )

            if result is not None:
                results[f"catboost_{strategy}"] = result
                print(f"\n✅ Experimento CatBoost-{strategy} CONCLUÍDO COM SUCESSO!")
            else:
                print(f"\n❌ Experimento CatBoost-{strategy} falhou (resultado None)")

        except Exception as e:
            print(f"\n❌ ERRO no experimento CatBoost-{strategy}: {e}")
            import traceback
            traceback.print_exc()

    # Resumo final
    elapsed_total = time.time() - start_time_total

    print("\n" + "="*80)
    print("RESUMO DA EXECUÇÃO COMPLETA")
    print("="*80)
    print(f"\n✅ Experimentos concluídos: {len(results)}/{len(strategies)}")
    print(f"⏱️  Tempo total: {elapsed_total:.2f} segundos ({elapsed_total/60:.2f} minutos)")

    if results:
        print("\n📁 Logs salvos em:")
        print("   logs/catboost/YYYYMMDD_HHMMSS_cyclic/")
        print("   logs/catboost/YYYYMMDD_HHMMSS_bagging/")

        print("\n📊 Experimentos executados:")
        for exp_name in results.keys():
            print(f"   ✓ {exp_name}")
    else:
        print("\n❌ Nenhum experimento foi concluído com sucesso!")
        return 1

    print("\n" + "="*80)
    print("✅ EXECUÇÃO COMPLETA FINALIZADA!")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
