"""
Script de teste para validar balanceamento de classes
Executa sem precisar do Flower/Ray
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from common import DataProcessor
import numpy as np

# Caminhos dos CSVs
TRAIN_CSV = r"C:\Users\candi\OneDrive\Desktop\Federated-Learning\dataset_fl\dataset\dataset_K400_seed42\dataset_all_vehicles.csv"
VALIDATION_CSV = r"C:\Users\candi\OneDrive\Desktop\Federated-Learning\dataset_fl\dataset\dataset_K400_seed42\dataset_validation_all_vehicles.csv"

def test_balanceamento(strategy):
    """Testa uma estratégia de balanceamento"""
    print(f"\n{'='*80}")
    print(f"TESTE: Balanceamento com estratégia '{strategy}'")
    print(f"{'='*80}\n")

    try:
        # Criar DataProcessor com balanceamento
        data_processor = DataProcessor(
            num_clients=3,
            seed=42,
            train_csv_path=TRAIN_CSV,
            validation_csv_path=VALIDATION_CSV,
            use_all_data=True,
            balance_strategy=strategy
        )

        # Carregar e preparar dados
        partitions_X, partitions_y, X_test, y_test = data_processor.load_and_prepare_data()

        # Verificar distribuição final
        print(f"\n{'='*80}")
        print(f"RESULTADO: Distribuição final por cliente")
        print(f"{'='*80}\n")

        for i, y_part in enumerate(partitions_y):
            unique, counts = np.unique(y_part, return_counts=True)
            total = len(y_part)
            print(f"Cliente {i}: {total} amostras")
            for cls, count in zip(unique, counts):
                pct = count/total*100
                print(f"  Classe {cls}: {count:6d} ({pct:5.1f}%)")
            print()

        # Calcular média de amostras por classe
        all_y = np.concatenate(partitions_y)
        unique, counts = np.unique(all_y, return_counts=True)
        total = len(all_y)

        print(f"TOTAL GERAL: {total} amostras")
        for cls, count in zip(unique, counts):
            pct = count/total*100
            print(f"  Classe {cls}: {count:6d} ({pct:5.1f}%)")

        # Se weights, mostrar pesos calculados
        if strategy == 'weights' and data_processor.class_weights:
            print(f"\nPesos de classe calculados:")
            for cls, weight in data_processor.class_weights.items():
                print(f"  Classe {cls}: {weight:.4f}")

        print(f"\n{'='*80}")
        print(f"[OK] TESTE '{strategy}' CONCLUÍDO COM SUCESSO!")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"[ERRO] Erro no teste '{strategy}': {e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SUITE DE TESTES - BALANCEAMENTO DE CLASSES")
    print("="*80 + "\n")

    estrategias = ['oversample', 'smote', 'undersample', 'weights', None]
    resultados = {}

    for estrategia in estrategias:
        nome = estrategia if estrategia else "sem_balanceamento"
        resultados[nome] = test_balanceamento(estrategia)

    # Resumo
    print("\n" + "="*80)
    print("RESUMO DOS TESTES")
    print("="*80)
    for nome, passou in resultados.items():
        status = "[OK] PASSOU" if passou else "[ERRO] FALHOU"
        print(f"{nome:20s}: {status}")
    print("="*80 + "\n")

    if all(resultados.values()):
        print("[SUCESSO] TODOS OS TESTES DE BALANCEAMENTO PASSARAM!\n")
        sys.exit(0)
    else:
        print("[AVISO] ALGUNS TESTES FALHARAM\n")
        sys.exit(1)
