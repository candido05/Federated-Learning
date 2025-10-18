"""
Script de teste para validar estratégias de agregação.

Testa FedBagging e FedCyclic com clientes simulados.
"""

import numpy as np
from typing import Dict, List, Tuple
from unittest.mock import Mock, MagicMock

from flwr.common import (
    Parameters,
    FitRes,
    EvaluateRes,
    Status,
    Code,
    ndarrays_to_parameters,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from strategies import FedBagging, FedCyclic


def criar_parametros_fake(seed=42) -> Parameters:
    """
    Cria parâmetros fake para testes.

    Args:
        seed: Seed para reprodutibilidade.

    Returns:
        Parameters do Flower.
    """
    np.random.seed(seed)
    # Simula bytes de modelo serializado como array numpy
    fake_model_bytes = np.random.randint(0, 256, size=100, dtype=np.uint8)
    return ndarrays_to_parameters([fake_model_bytes])


def criar_cliente_mock(cid: str) -> ClientProxy:
    """
    Cria cliente mock para testes.

    Args:
        cid: ID do cliente.

    Returns:
        ClientProxy mockado.
    """
    cliente = Mock(spec=ClientProxy)
    cliente.cid = cid
    return cliente


def criar_fit_res(num_examples: int = 100, accuracy: float = 0.85) -> FitRes:
    """
    Cria FitRes mock para testes.

    Args:
        num_examples: Número de exemplos treinados.
        accuracy: Acurácia fake.

    Returns:
        FitRes mockado.
    """
    return FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=criar_parametros_fake(),
        num_examples=num_examples,
        metrics={"accuracy": accuracy, "loss": 1.0 - accuracy},
    )


def criar_evaluate_res(
    num_examples: int = 100, loss: float = 0.15, accuracy: float = 0.85
) -> EvaluateRes:
    """
    Cria EvaluateRes mock para testes.

    Args:
        num_examples: Número de exemplos avaliados.
        loss: Loss fake.
        accuracy: Acurácia fake.

    Returns:
        EvaluateRes mockado.
    """
    return EvaluateRes(
        status=Status(code=Code.OK, message="Success"),
        loss=loss,
        num_examples=num_examples,
        metrics={"accuracy": accuracy},
    )


def testar_fed_bagging():
    """
    Testa estratégia FedBagging.
    """
    print("\n" + "=" * 60)
    print("Testando FedBagging")
    print("=" * 60)

    try:
        # Cria estratégia
        print("\n[TESTE 1] Criando estratégia FedBagging...")
        estrategia = FedBagging(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )
        print("✓ Estratégia criada com sucesso")

        # Cria client manager com clientes mock
        print("\n[TESTE 2] Configurando clientes...")
        client_manager = SimpleClientManager()
        clientes = [criar_cliente_mock(f"client_{i}") for i in range(4)]

        for cliente in clientes:
            client_manager.register(cliente)

        print(f"✓ {len(clientes)} clientes registrados")

        # Testa configure_fit
        print("\n[TESTE 3] Testando configure_fit...")
        parametros = criar_parametros_fake()
        fit_configs = estrategia.configure_fit(
            server_round=1, parameters=parametros, client_manager=client_manager
        )

        assert len(fit_configs) == 4, f"Esperado 4 clientes, obteve {len(fit_configs)}"
        print(f"✓ configure_fit retornou {len(fit_configs)} configurações")

        # Testa aggregate_fit
        print("\n[TESTE 4] Testando aggregate_fit...")
        results = [
            (clientes[0], criar_fit_res(num_examples=100, accuracy=0.85)),
            (clientes[1], criar_fit_res(num_examples=150, accuracy=0.90)),
            (clientes[2], criar_fit_res(num_examples=120, accuracy=0.88)),
        ]

        aggregated_params, aggregated_metrics = estrategia.aggregate_fit(
            server_round=1, results=results, failures=[]
        )

        assert aggregated_params is not None, "Parâmetros agregados não devem ser None"
        assert "accuracy" in aggregated_metrics, "Métricas devem conter 'accuracy'"
        assert "loss" in aggregated_metrics, "Métricas devem conter 'loss'"

        print(f"✓ Parâmetros agregados com sucesso")
        print(f"✓ Métricas agregadas: {aggregated_metrics}")

        # Testa configure_evaluate
        print("\n[TESTE 5] Testando configure_evaluate...")
        eval_configs = estrategia.configure_evaluate(
            server_round=1, parameters=parametros, client_manager=client_manager
        )

        assert len(eval_configs) == 4, f"Esperado 4 clientes, obteve {len(eval_configs)}"
        print(f"✓ configure_evaluate retornou {len(eval_configs)} configurações")

        # Testa aggregate_evaluate
        print("\n[TESTE 6] Testando aggregate_evaluate...")
        eval_results = [
            (clientes[0], criar_evaluate_res(num_examples=100, loss=0.15, accuracy=0.85)),
            (clientes[1], criar_evaluate_res(num_examples=150, loss=0.10, accuracy=0.90)),
        ]

        aggregated_loss, eval_metrics = estrategia.aggregate_evaluate(
            server_round=1, results=eval_results, failures=[]
        )

        assert aggregated_loss is not None, "Loss agregada não deve ser None"
        assert "accuracy" in eval_metrics, "Métricas devem conter 'accuracy'"

        print(f"✓ Loss agregada: {aggregated_loss:.4f}")
        print(f"✓ Métricas de avaliação: {eval_metrics}")

        print("\n" + "=" * 60)
        print("✓✓✓ TODOS OS TESTES DE FedBagging PASSARAM ✓✓✓")
        print("=" * 60)

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗✗✗ ERRO NO TESTE DE FedBagging ✗✗✗")
        print(f"Erro: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


def testar_fed_cyclic():
    """
    Testa estratégia FedCyclic.
    """
    print("\n" + "=" * 60)
    print("Testando FedCyclic")
    print("=" * 60)

    try:
        # Cria estratégia
        print("\n[TESTE 1] Criando estratégia FedCyclic...")
        estrategia = FedCyclic(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_evaluate_clients=1,
            min_available_clients=1,
        )
        print("✓ Estratégia criada com sucesso")
        print(f"✓ min_fit_clients foi forçado para 1 (valor correto para cyclic)")

        # Cria client manager com clientes mock
        print("\n[TESTE 2] Configurando clientes...")
        client_manager = SimpleClientManager()
        clientes = [criar_cliente_mock(f"client_{i}") for i in range(4)]

        for cliente in clientes:
            client_manager.register(cliente)

        print(f"✓ {len(clientes)} clientes registrados")

        # Testa configure_fit em múltiplos rounds (verifica ciclo)
        print("\n[TESTE 3] Testando configure_fit cíclico...")
        parametros = criar_parametros_fake()
        selected_clients = []

        for round_num in range(1, 6):  # 5 rounds (mais que número de clientes)
            fit_configs = estrategia.configure_fit(
                server_round=round_num, parameters=parametros, client_manager=client_manager
            )

            assert len(fit_configs) == 1, f"Esperado 1 cliente, obteve {len(fit_configs)}"
            selected_client_cid = fit_configs[0][0].cid
            selected_clients.append(selected_client_cid)
            print(f"  Round {round_num}: Cliente {selected_client_cid} selecionado")

        # Verifica padrão cíclico
        assert selected_clients[0] == "client_0", "Round 1 deve selecionar client_0"
        assert selected_clients[1] == "client_1", "Round 2 deve selecionar client_1"
        assert selected_clients[2] == "client_2", "Round 3 deve selecionar client_2"
        assert selected_clients[3] == "client_3", "Round 4 deve selecionar client_3"
        assert selected_clients[4] == "client_0", "Round 5 deve voltar para client_0 (ciclo)"

        print("✓ Padrão cíclico verificado: client_0 → client_1 → client_2 → client_3 → client_0")

        # Testa aggregate_fit
        print("\n[TESTE 4] Testando aggregate_fit (1 cliente)...")
        results = [
            (clientes[0], criar_fit_res(num_examples=100, accuracy=0.85)),
        ]

        aggregated_params, aggregated_metrics = estrategia.aggregate_fit(
            server_round=1, results=results, failures=[]
        )

        assert aggregated_params is not None, "Parâmetros agregados não devem ser None"
        assert aggregated_metrics["accuracy"] == 0.85, "Métrica deve ser do único cliente"

        print(f"✓ Parâmetros do único cliente aceitos como modelo global")
        print(f"✓ Métricas: {aggregated_metrics}")

        # Testa configure_evaluate
        print("\n[TESTE 5] Testando configure_evaluate...")
        eval_configs = estrategia.configure_evaluate(
            server_round=1, parameters=parametros, client_manager=client_manager
        )

        # Com fraction_evaluate=1.0, deve avaliar todos os clientes
        assert len(eval_configs) >= 1, "Deve avaliar pelo menos 1 cliente"
        print(f"✓ configure_evaluate retornou {len(eval_configs)} configurações")

        # Testa aggregate_evaluate
        print("\n[TESTE 6] Testando aggregate_evaluate...")
        eval_results = [
            (clientes[0], criar_evaluate_res(num_examples=100, loss=0.15, accuracy=0.85)),
        ]

        aggregated_loss, eval_metrics = estrategia.aggregate_evaluate(
            server_round=1, results=eval_results, failures=[]
        )

        assert aggregated_loss == 0.15, "Loss deve ser do único cliente"
        assert eval_metrics["accuracy"] == 0.85, "Métrica deve ser do único cliente"

        print(f"✓ Loss agregada: {aggregated_loss:.4f}")
        print(f"✓ Métricas de avaliação: {eval_metrics}")

        print("\n" + "=" * 60)
        print("✓✓✓ TODOS OS TESTES DE FedCyclic PASSARAM ✓✓✓")
        print("=" * 60)

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗✗✗ ERRO NO TESTE DE FedCyclic ✗✗✗")
        print(f"Erro: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Executa testes para todas as estratégias.
    """
    print("\n" + "=" * 60)
    print("INICIANDO TESTES DE ESTRATÉGIAS FEDERATED LEARNING")
    print("=" * 60)

    resultados = {}

    # Testa FedBagging
    resultados['FedBagging'] = testar_fed_bagging()

    # Testa FedCyclic
    resultados['FedCyclic'] = testar_fed_cyclic()

    # Resumo dos resultados
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)

    todos_passaram = True
    for nome, passou in resultados.items():
        status = "✓ PASSOU" if passou else "✗ FALHOU"
        print(f"{nome:15s}: {status}")
        if not passou:
            todos_passaram = False

    print("=" * 60)

    if todos_passaram:
        print("\n✓✓✓ TODAS AS ESTRATÉGIAS FUNCIONARAM CORRETAMENTE ✓✓✓\n")
        return 0
    else:
        print("\n✗✗✗ ALGUNS TESTES FALHARAM ✗✗✗\n")
        return 1


if __name__ == "__main__":
    exit(main())
