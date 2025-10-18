"""Script de teste para a camada de dados.

Este script demonstra o uso do sistema de datasets e valida a implementação.
"""

from config import GlobalConfig
from data import create_dataset, get_dataset_info, list_available_datasets


def test_dataset_factory():
    """Testa a factory de datasets."""
    print("\n" + "=" * 80)
    print("TESTE: Dataset Factory")
    print("=" * 80)

    # Lista datasets disponíveis
    available = list_available_datasets()
    print(f"\nDatasets disponíveis: {available}")

    # Informações do HIGGS
    info = get_dataset_info("higgs")
    print(f"\nInformações do dataset HIGGS:")
    for key, value in info.items():
        print(f"  {key}: {value}")


def test_tabular_dataset():
    """Testa o carregamento e processamento do dataset tabular (HIGGS)."""
    print("\n" + "=" * 80)
    print("TESTE: Dataset Tabular (HIGGS)")
    print("=" * 80)

    # Configuração de teste (números pequenos para teste rápido)
    config = GlobalConfig(
        num_clients=3,
        sample_per_client=1000,  # Apenas 1000 amostras por cliente para teste
        num_server_rounds=5,
        seed=42,
        test_fraction=0.2,
        dataset_name="higgs",
    )

    print(f"\nConfiguração:")
    print(f"  Número de clientes: {config.num_clients}")
    print(f"  Amostras por cliente: {config.sample_per_client}")
    print(f"  Total de amostras: {config.num_clients * config.sample_per_client}")
    print(f"  Fração de teste: {config.test_fraction}")
    print(f"  Seed: {config.seed}")

    # Cria dataset
    print("\n" + "-" * 80)
    print("Criando dataset...")
    dataset = create_dataset("higgs", config)

    # Carrega dados
    print("\n" + "-" * 80)
    print("Carregando dados...")
    dataset.load_data()

    # Informações do dataset
    print("\n" + "-" * 80)
    print("Informações do dataset:")
    info = dataset.get_dataset_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Testa dados de treino
    print("\n" + "-" * 80)
    print("Testando dados de treino...")
    X_train, y_train = dataset.get_train_data()
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_train dtype: {X_train.dtype}")
    print(f"  y_train dtype: {y_train.dtype}")
    print(f"  X_train min/max: {X_train.min():.4f} / {X_train.max():.4f}")
    print(f"  y_train unique: {sorted(set(y_train.tolist()))}")

    # Testa dados de teste
    print("\n" + "-" * 80)
    print("Testando dados de teste...")
    X_test, y_test = dataset.get_test_data()
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

    # Testa dados de validação
    print("\n" + "-" * 80)
    print("Testando dados de validação...")
    X_val, y_val = dataset.get_validation_data()
    print(f"  X_val shape: {X_val.shape}")
    print(f"  y_val shape: {y_val.shape}")

    # Testa particionamento
    print("\n" + "-" * 80)
    print(f"Testando particionamento para {config.num_clients} clientes...")
    partitions = dataset.get_partitions(config.num_clients)
    print(f"  Número de partições: {len(partitions)}")

    for i, (X_client, y_client) in enumerate(partitions):
        print(f"  Cliente {i}:")
        print(f"    Shape: X={X_client.shape}, y={y_client.shape}")
        print(f"    Classes: {sorted(set(y_client.tolist()))}")
        print(f"    Distribuição: {dict(zip(*__import__('numpy').unique(y_client, return_counts=True)))}")

    # Testa scaler
    print("\n" + "-" * 80)
    print("Testando scaler...")
    scaler = dataset.get_scaler()
    print(f"  Scaler mean: {scaler.mean_[:5]}")  # Primeiros 5 valores
    print(f"  Scaler scale: {scaler.scale_[:5]}")

    # Testa preprocess
    print("\n" + "-" * 80)
    print("Testando preprocess...")
    import numpy as np
    X_sample = np.random.randn(10, dataset.get_num_features())
    X_processed = dataset.preprocess(X_sample)
    print(f"  Input shape: {X_sample.shape}")
    print(f"  Output shape: {X_processed.shape}")
    print(f"  Output mean: {X_processed.mean():.4f}")
    print(f"  Output std: {X_processed.std():.4f}")

    # Número de features
    print("\n" + "-" * 80)
    print(f"Número de features: {dataset.get_num_features()}")


def main():
    """Função principal de testes."""
    print("\n" + "🚀 " * 40)
    print("TESTE COMPLETO DA CAMADA DE DADOS")
    print("🚀 " * 40)

    # Teste 1: Factory
    test_dataset_factory()

    # Teste 2: Tabular Dataset
    test_tabular_dataset()

    print("\n" + "✅ " * 40)
    print("TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
    print("✅ " * 40 + "\n")


if __name__ == "__main__":
    main()
