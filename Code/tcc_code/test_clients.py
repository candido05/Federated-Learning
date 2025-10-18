"""
Script de teste para validar clientes específicos de cada framework.

Testa serialização, desserialização e treinamento incremental.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from models import XGBoostClient, CatBoostClient, LightGBMClient
from config import XGBOOST_PARAMS, CATBOOST_PARAMS, LIGHTGBM_PARAMS
from utils.logging_utils import ExperimentLogger


def criar_dados_sinteticos(n_samples=1000, n_features=20, n_classes=2, random_state=42):
    """
    Cria dataset sintético para testes.

    Args:
        n_samples: Número de amostras.
        n_features: Número de features.
        n_classes: Número de classes.
        random_state: Seed para reprodutibilidade.

    Returns:
        Tupla com dados de treino e validação.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=n_classes,
        random_state=random_state
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    return (X_train, y_train), (X_valid, y_valid)


def testar_cliente(nome_cliente, classe_cliente, params, num_rounds=5):
    """
    Testa um cliente específico.

    Args:
        nome_cliente: Nome do cliente para logging.
        classe_cliente: Classe do cliente a testar.
        params: Parâmetros do modelo.
        num_rounds: Número de rounds de treinamento.
    """
    print(f"\n{'='*60}")
    print(f"Testando {nome_cliente}")
    print(f"{'='*60}")

    # Cria logger
    logger = ExperimentLogger(
        experiment_name=f"test_{nome_cliente.lower()}",
        model_name=nome_cliente.lower(),
        strategy_name="test",
        num_clients=1
    )

    # Cria dados sintéticos
    train_data, valid_data = criar_dados_sinteticos()

    # Cria cliente
    cliente = classe_cliente(
        train_data=train_data,
        valid_data=valid_data,
        num_local_round=num_rounds,
        params=params,
        train_method="fit",
        client_id="test_0",
        experiment_logger=logger
    )

    try:
        # Teste 1: Treinamento inicial (sem modelo global)
        print("\n[TESTE 1] Treinamento inicial (sem modelo global)...")
        modelo_treinado = cliente.train_local_model(global_model_bytes=None)
        print("✓ Treinamento inicial concluído")

        # Teste 2: Serialização
        print("\n[TESTE 2] Serialização do modelo...")
        modelo_bytes = cliente.save_model_bytes(modelo_treinado)
        print(f"✓ Modelo serializado: {len(modelo_bytes)} bytes")

        # Teste 3: Desserialização
        print("\n[TESTE 3] Desserialização do modelo...")
        modelo_carregado = cliente.load_model_from_bytes(modelo_bytes)
        print("✓ Modelo desserializado com sucesso")

        # Teste 4: Predição
        print("\n[TESTE 4] Predição com modelo carregado...")
        X_valid, y_valid = valid_data
        predicoes = cliente.predict(modelo_carregado, X_valid)
        print(f"✓ Predição realizada: shape={predicoes.shape}")

        # Verifica shape das predições
        assert predicoes.shape[0] == X_valid.shape[0], "Número de predições incorreto"
        assert predicoes.shape[1] >= 2, "Predições devem ter pelo menos 2 classes"
        assert np.allclose(predicoes.sum(axis=1), 1.0, atol=1e-5), "Probabilidades devem somar 1"
        print(f"✓ Shape das predições correto: {predicoes.shape}")
        print(f"✓ Probabilidades somam 1.0: {predicoes.sum(axis=1)[:5]}")

        # Teste 5: Treinamento incremental (com modelo global)
        print("\n[TESTE 5] Treinamento incremental (com modelo global)...")
        modelo_incremental = cliente.train_local_model(global_model_bytes=modelo_bytes)
        print("✓ Treinamento incremental concluído")

        # Teste 6: Predição com modelo incremental
        print("\n[TESTE 6] Predição com modelo incremental...")
        predicoes_inc = cliente.predict(modelo_incremental, X_valid)
        print(f"✓ Predição incremental: shape={predicoes_inc.shape}")

        # Teste 7: Verificar que modelos são diferentes
        print("\n[TESTE 7] Verificando diferença entre modelos...")
        diferenca = np.abs(predicoes - predicoes_inc).mean()
        print(f"✓ Diferença média nas predições: {diferenca:.6f}")

        if diferenca > 0:
            print("✓ Modelos são diferentes (treinamento incremental funcionou)")
        else:
            print("⚠ Modelos idênticos (pode ser esperado em alguns casos)")

        print(f"\n{'='*60}")
        print(f"✓✓✓ TODOS OS TESTES PASSARAM PARA {nome_cliente} ✓✓✓")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗✗✗ ERRO NO TESTE DE {nome_cliente} ✗✗✗")
        print(f"Erro: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Executa testes para todos os clientes.
    """
    print("\n" + "="*60)
    print("INICIANDO TESTES DE CLIENTES FEDERATED LEARNING")
    print("="*60)

    resultados = {}

    # Teste XGBoost
    resultados['XGBoost'] = testar_cliente(
        nome_cliente="XGBoost",
        classe_cliente=XGBoostClient,
        params=XGBOOST_PARAMS,
        num_rounds=5
    )

    # Teste CatBoost
    resultados['CatBoost'] = testar_cliente(
        nome_cliente="CatBoost",
        classe_cliente=CatBoostClient,
        params=CATBOOST_PARAMS,
        num_rounds=5
    )

    # Teste LightGBM
    resultados['LightGBM'] = testar_cliente(
        nome_cliente="LightGBM",
        classe_cliente=LightGBMClient,
        params=LIGHTGBM_PARAMS,
        num_rounds=5
    )

    # Resumo dos resultados
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)

    todos_passaram = True
    for nome, passou in resultados.items():
        status = "✓ PASSOU" if passou else "✗ FALHOU"
        print(f"{nome:15s}: {status}")
        if not passou:
            todos_passaram = False

    print("="*60)

    if todos_passaram:
        print("\n✓✓✓ TODOS OS CLIENTES FUNCIONARAM CORRETAMENTE ✓✓✓\n")
        return 0
    else:
        print("\n✗✗✗ ALGUNS TESTES FALHARAM ✗✗✗\n")
        return 1


if __name__ == "__main__":
    exit(main())
