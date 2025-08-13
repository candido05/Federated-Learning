"""
Experimento principal de Federated Learning com modelos de ML
"""

import warnings
warnings.filterwarnings("ignore")

from flwr.simulation import run_simulation
from client import create_client_app
from server import create_server_app, AVAILABLE_STRATEGIES
from models import AVAILABLE_MODELS
from visualization import plot_results, save_results_to_csv
import time
from typing import Dict, List


# Configurações do experimento
NUM_PARTITIONS = 10
NUM_ROUNDS = 20
BATCH_SIZE = 2

# Configuração do backend
backend_config = {"client_resources": None}


def run_single_experiment(model_name: str, strategy_name: str, 
                         results: Dict[str, Dict[str, List[float]]]) -> float:
    """Executa um experimento único com um modelo e estratégia específicos"""
    
    print(f"\n{'='*80}")
    print(f"Executando: {model_name.upper()} com {strategy_name}")
    print(f"{'='*80}")
    
    # Cria chave única para os resultados
    key = f"{model_name}_{strategy_name}"
    results[key] = {"rounds": [], "loss": [], "accuracy": []}
    
    # Cria aplicações cliente e servidor
    client_app = create_client_app(model_name, NUM_PARTITIONS)
    server_app = create_server_app(strategy_name, model_name, results[key], 
                                  NUM_PARTITIONS, NUM_ROUNDS)
    
    # Mede tempo de execução
    start_time = time.time()
    
    try:
        # Executa simulação
        history = run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=NUM_PARTITIONS,
            backend_config=backend_config
        )
        
        execution_time = time.time() - start_time
        print(f"Experimento {key} concluído em {execution_time:.2f}s")
        
        return execution_time
        
    except Exception as e:
        print(f"Erro no experimento {key}: {str(e)}")
        return 0.0


def run_all_experiments():
    """Executa todos os experimentos (todos os modelos com todas as estratégias)"""
    
    print("Iniciando experimentos de Federated Learning")
    print(f"Modelos: {AVAILABLE_MODELS}")
    print(f"Estratégias: {AVAILABLE_STRATEGIES}")
    print(f"Configuração: {NUM_PARTITIONS} clientes, {NUM_ROUNDS} rodadas")
    
    results = {}
    execution_times = {}
    total_start_time = time.time()
    
    # Executa experimentos para cada combinação modelo-estratégia
    for model_name in AVAILABLE_MODELS:
        execution_times[model_name] = {}
        
        for strategy_name in AVAILABLE_STRATEGIES:
            exec_time = run_single_experiment(model_name, strategy_name, results)
            execution_times[model_name][strategy_name] = exec_time
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("TODOS OS EXPERIMENTOS CONCLUÍDOS!")
    print(f"Tempo total: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"{'='*80}")
    
    # Salva resultados
    save_results_to_csv(results, execution_times)
    
    # Gera visualizações
    plot_results(results)
    
    return results, execution_times


def run_specific_experiments(models: List[str] = None, strategies: List[str] = None):
    """Executa experimentos específicos"""
    
    if models is None:
        models = AVAILABLE_MODELS[:3]  # Primeiros 3 modelos por padrão
    
    if strategies is None:
        strategies = AVAILABLE_STRATEGIES[:3]  # Primeiras 3 estratégias por padrão
    
    print("Executando experimentos específicos:")
    print(f"Modelos: {models}")
    print(f"Estratégias: {strategies}")
    
    results = {}
    execution_times = {}
    
    for model_name in models:
        if model_name not in AVAILABLE_MODELS:
            print(f"Modelo {model_name} não disponível. Modelos disponíveis: {AVAILABLE_MODELS}")
            continue
            
        execution_times[model_name] = {}
        
        for strategy_name in strategies:
            if strategy_name not in AVAILABLE_STRATEGIES:
                print(f"Estratégia {strategy_name} não disponível. Estratégias disponíveis: {AVAILABLE_STRATEGIES}")
                continue
                
            exec_time = run_single_experiment(model_name, strategy_name, results)
            execution_times[model_name][strategy_name] = exec_time
    
    # Salva e visualiza resultados
    save_results_to_csv(results, execution_times, prefix="specific_")
    plot_results(results, save_prefix="specific_")
    
    return results, execution_times


def main():
    """Função principal"""
    print("Federated Learning com Modelos de Machine Learning")
    print("Dataset: MNIST")
    print(f"Modelos disponíveis: {AVAILABLE_MODELS}")
    print(f"Estratégias disponíveis: {AVAILABLE_STRATEGIES}")
    
    # Opções de execução
    print("\nOpções de execução:")
    print("1. Executar todos os experimentos (pode demorar muito)")
    print("2. Executar experimentos específicos (recomendado para teste)")
    print("3. Executar experimento único")
    
    choice = input("Escolha uma opção (1-3): ").strip()
    
    if choice == "1":
        results, times = run_all_experiments()
        
    elif choice == "2":
        # Experimentos específicos - modifica aqui para escolher quais executar
        selected_models = ["logistic_regression", "random_forest", "svm"]
        selected_strategies = ["FedAvg", "FedProx", "FedAdam"]
        results, times = run_specific_experiments(selected_models, selected_strategies)
        
    elif choice == "3":
        # Experimento único
        print(f"Modelos: {AVAILABLE_MODELS}")
        model = input("Escolha um modelo: ").strip()
        
        print(f"Estratégias: {AVAILABLE_STRATEGIES}")
        strategy = input("Escolha uma estratégia: ").strip()
        
        results = {}
        exec_time = run_single_experiment(model, strategy, results)
        plot_results(results, save_prefix="single_")
        
    else:
        print("Opção inválida!")
        return
    
    print("\nExperimento concluído! Verifique os arquivos gerados:")
    print("- results.csv ou specific_results.csv: Resultados em CSV")
    print("- comparison_plots.png: Gráficos de comparação")


if __name__ == "__main__":
    main()