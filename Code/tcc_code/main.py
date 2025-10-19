"""
Script principal para executar experimentos de Federated Learning.

Fornece interface CLI completa para rodar experimentos com diferentes
modelos e estratégias.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import signal

# Importa colorama se disponível
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback: cores vazias
    class Fore:
        GREEN = RED = YELLOW = CYAN = BLUE = MAGENTA = ""
    class Style:
        BRIGHT = RESET_ALL = ""

# Importa tqdm se disponível
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from config import GlobalConfig, LoggingConfig
from server import FederatedServer
from utils import MetricsCalculator


# Estado global para Ctrl+C
interrupted = False
partial_results = {}


def signal_handler(signum, frame):
    """Handler para Ctrl+C graceful."""
    global interrupted
    print(f"\n{Fore.YELLOW}⚠ Interrupção detectada. Salvando estado...{Style.RESET_ALL}")
    interrupted = True


# Registra handler
signal.signal(signal.SIGINT, signal_handler)


def colorize(text: str, color: str = "", bold: bool = False) -> str:
    """
    Coloriza texto se colorama disponível.

    Args:
        text: Texto a colorizar.
        color: Cor (atributo de Fore).
        bold: Se deve usar bold.

    Returns:
        Texto colorizado ou normal.
    """
    if not COLORAMA_AVAILABLE:
        return text

    style = Style.BRIGHT if bold else ""
    color_code = getattr(Fore, color.upper(), "")
    return f"{style}{color_code}{text}{Style.RESET_ALL}"


def list_available_models() -> List[str]:
    """
    Lista modelos disponíveis.

    Returns:
        Lista de nomes de modelos.
    """
    return ['xgboost', 'catboost', 'lightgbm']


def list_available_datasets() -> List[str]:
    """
    Lista datasets disponíveis.

    Returns:
        Lista de nomes de datasets.
    """
    return [
        'jxie/higgs',
        # Adicione outros datasets aqui conforme necessário
    ]


def list_available_strategies() -> List[str]:
    """
    Lista estratégias disponíveis.

    Returns:
        Lista de nomes de estratégias.
    """
    return ['bagging', 'cyclic']


def print_available_models():
    """Exibe modelos disponíveis e sai."""
    print(f"\n{colorize('Modelos Disponíveis:', 'cyan', bold=True)}")
    print("=" * 60)

    models = list_available_models()
    for i, model in enumerate(models, 1):
        print(f"  {i}. {colorize(model, 'green')}")

    print("=" * 60)
    print(f"Total: {colorize(str(len(models)), 'yellow', bold=True)} modelos\n")


def print_available_datasets():
    """Exibe datasets disponíveis e sai."""
    print(f"\n{colorize('Datasets Disponíveis:', 'cyan', bold=True)}")
    print("=" * 60)

    datasets = list_available_datasets()
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {colorize(dataset, 'green')}")

    print("=" * 60)
    print(f"Total: {colorize(str(len(datasets)), 'yellow', bold=True)} dataset(s)\n")


def validate_config(config: GlobalConfig) -> Tuple[bool, List[str]]:
    """
    Valida configuração.

    Args:
        config: Configuração global.

    Returns:
        Tupla (is_valid, error_messages).
    """
    errors = []

    # Valida valores positivos
    if config.num_clients <= 0:
        errors.append("num_clients deve ser > 0")

    if config.num_server_rounds <= 0:
        errors.append("num_server_rounds deve ser > 0")

    if config.num_local_boost_round <= 0:
        errors.append("num_local_boost_round deve ser > 0")

    if config.sample_per_client <= 0:
        errors.append("sample_per_client deve ser > 0")

    # Valida seed
    if config.seed < 0:
        errors.append("seed deve ser >= 0")

    # Valida relações
    total_samples = config.num_clients * config.sample_per_client
    if total_samples > 11000000:  # Dataset Higgs tem ~11M amostras
        errors.append(
            f"Total de amostras ({total_samples}) excede dataset "
            "(considere reduzir num_clients ou sample_per_client)"
        )

    return len(errors) == 0, errors


def get_experiments_to_run(
    model_arg: str,
    strategy_arg: str
) -> List[Tuple[str, str]]:
    """
    Determina quais experimentos executar.

    Args:
        model_arg: Argumento --model.
        strategy_arg: Argumento --strategy.

    Returns:
        Lista de tuplas (model, strategy).
    """
    models = list_available_models() if model_arg == 'all' else [model_arg]
    strategies = list_available_strategies() if strategy_arg == 'all' else [strategy_arg]

    experiments = []
    for model in models:
        for strategy in strategies:
            experiments.append((model, strategy))

    return experiments


def print_dry_run_info(experiments: List[Tuple[str, str]], config: GlobalConfig):
    """
    Exibe informações de dry run.

    Args:
        experiments: Lista de experimentos.
        config: Configuração global.
    """
    print(f"\n{colorize('DRY RUN - Configuração do Experimento', 'yellow', bold=True)}")
    print("=" * 60)

    print(f"\n{colorize('Configurações Globais:', 'cyan')}")
    print(f"  Clientes: {colorize(str(config.num_clients), 'green')}")
    print(f"  Rounds: {colorize(str(config.num_server_rounds), 'green')}")
    print(f"  Rounds locais: {colorize(str(config.num_local_boost_round), 'green')}")
    print(f"  Amostras/cliente: {colorize(str(config.sample_per_client), 'green')}")
    print(f"  Seed: {colorize(str(config.seed), 'green')}")

    print(f"\n{colorize('Experimentos a Executar:', 'cyan')}")
    for i, (model, strategy) in enumerate(experiments, 1):
        print(f"  {i}. {colorize(model, 'green')} + {colorize(strategy, 'blue')}")

    print(f"\n{colorize('Total:', 'yellow')} {colorize(str(len(experiments)), 'yellow', bold=True)} experimento(s)")
    print("=" * 60)
    print(f"\n{colorize('[OK] Configuracao valida. Use sem --dry-run para executar.', 'green')}\n")


def print_experiment_summary(
    model: str,
    strategy: str,
    result: Dict[str, Any],
    index: int,
    total: int
):
    """
    Exibe resumo de um experimento.

    Args:
        model: Nome do modelo.
        strategy: Nome da estratégia.
        result: Resultado do experimento.
        index: Índice do experimento.
        total: Total de experimentos.
    """
    print(f"\n{colorize(f'Experimento {index}/{total} Concluído', 'green', bold=True)}")
    print("=" * 60)
    print(f"Modelo: {colorize(model, 'cyan')}")
    print(f"Estratégia: {colorize(strategy, 'blue')}")
    print(f"Rounds: {result.get('num_rounds', 'N/A')}")
    print(f"Clientes: {result.get('num_clients', 'N/A')}")

    # Tenta extrair métricas finais do history
    history = result.get('history')
    if history and hasattr(history, 'metrics_centralized'):
        metrics = history.metrics_centralized
        if metrics:
            last_round = max(metrics.keys())
            last_metrics = metrics[last_round]

            print(f"\n{colorize(f'Métricas Finais (Round {last_round}):', 'yellow')}")
            for metric_name, value in last_metrics.items():
                print(f"  {metric_name}: {colorize(f'{value:.4f}', 'green')}")

    print("=" * 60)


def generate_comparison_report(
    results_dict: Dict[Tuple[str, str], Dict[str, Any]],
    output_path: Path
) -> Dict[str, Any]:
    """
    Gera relatório comparativo de múltiplos experimentos.

    Args:
        results_dict: Dicionário {(model, strategy): result}.
        output_path: Caminho para salvar JSON.

    Returns:
        Dicionário do relatório.
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'num_experiments': len(results_dict),
        'experiments': {},
        'comparison_table': [],
    }

    # Coleta dados de cada experimento
    for (model, strategy), result in results_dict.items():
        exp_key = f"{model}_{strategy}"

        # Extrai métricas finais
        final_metrics = {}
        history = result.get('history')

        if history and hasattr(history, 'metrics_centralized'):
            metrics = history.metrics_centralized
            if metrics:
                last_round = max(metrics.keys())
                final_metrics = metrics[last_round]

        report['experiments'][exp_key] = {
            'model': model,
            'strategy': strategy,
            'num_rounds': result.get('num_rounds'),
            'num_clients': result.get('num_clients'),
            'final_metrics': final_metrics,
        }

        # Adiciona à tabela comparativa
        report['comparison_table'].append({
            'experiment': exp_key,
            'model': model,
            'strategy': strategy,
            **final_metrics,
        })

    # Salva JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def print_comparison_summary(report: Dict[str, Any]):
    """
    Exibe resumo comparativo no console.

    Args:
        report: Relatório de comparação.
    """
    print(f"\n{colorize('Resumo Comparativo', 'cyan', bold=True)}")
    print("=" * 80)

    table = report['comparison_table']

    if not table:
        print("Nenhum resultado para comparar.")
        return

    # Cabeçalho
    print(f"\n{'Experimento':<25} {'Accuracy':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 80)

    # Linhas
    for row in table:
        exp_name = row['experiment']
        accuracy = row.get('accuracy', 0.0)
        f1 = row.get('f1', 0.0)
        auc = row.get('auc', 0.0)

        print(f"{exp_name:<25} {accuracy:<12.4f} {f1:<12.4f} {auc:<12.4f}")

    print("-" * 80)

    # Melhor por métrica
    if len(table) > 1:
        print(f"\n{colorize('Melhores Resultados:', 'green')}")

        for metric in ['accuracy', 'f1', 'auc']:
            if metric in table[0]:
                best = max(table, key=lambda x: x.get(metric, 0.0))
                print(f"  {metric.upper()}: {colorize(best['experiment'], 'yellow')} "
                      f"({best.get(metric, 0.0):.4f})")

    print("=" * 80)


def run_single_experiment(
    model: str,
    strategy: str,
    config: GlobalConfig,
    logging_config: LoggingConfig,
    dataset_source: str,
) -> Dict[str, Any]:
    """
    Executa um experimento único.

    Args:
        model: Nome do modelo.
        strategy: Nome da estratégia.
        config: Configuração global.
        logging_config: Configuração de logging.
        dataset_source: Fonte do dataset.

    Returns:
        Resultado do experimento.
    """
    # Cria servidor
    server = FederatedServer(config, logging_config)

    # Executa experimento
    result = server.run_experiment(
        model_type=model,
        strategy_type=strategy,
        dataset_source=dataset_source,
    )

    return result


def main():
    """Função principal."""
    # Parser de argumentos
    parser = argparse.ArgumentParser(
        description='Executar experimentos de Federated Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Executar XGBoost com Bagging
  python main.py --model xgboost --strategy bagging

  # Executar todos os modelos com estratégia Cyclic
  python main.py --model all --strategy cyclic

  # Executar todos os experimentos (3 modelos x 2 estratégias = 6 experimentos)
  python main.py --model all --strategy all

  # Dry run para validar configuração
  python main.py --model all --strategy all --dry-run

  # Listar modelos disponíveis
  python main.py --list-models
        """
    )

    # Argumentos principais
    parser.add_argument(
        '--model',
        choices=list_available_models() + ['all'],
        default='xgboost',
        help='Modelo a usar (padrão: xgboost)'
    )

    parser.add_argument(
        '--strategy',
        choices=list_available_strategies() + ['all'],
        default='bagging',
        help='Estratégia de agregação (padrão: bagging)'
    )

    parser.add_argument(
        '--dataset',
        default='jxie/higgs',
        help='Dataset HuggingFace (padrão: jxie/higgs)'
    )

    # Configurações de experimento
    parser.add_argument(
        '--num-clients',
        type=int,
        default=6,
        help='Número de clientes (padrão: 6)'
    )

    parser.add_argument(
        '--num-rounds',
        type=int,
        default=10,
        help='Número de rounds de treinamento (padrão: 10)'
    )

    parser.add_argument(
        '--num-local-rounds',
        type=int,
        default=5,
        help='Número de rounds locais por cliente (padrão: 5)'
    )

    parser.add_argument(
        '--sample-per-client',
        type=int,
        default=8000,
        help='Amostras por cliente (padrão: 8000)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed para reprodutibilidade (padrão: 42)'
    )

    # Ações especiais
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='Lista modelos disponíveis e sai'
    )

    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='Lista datasets disponíveis e sai'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Valida configuração sem executar'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Habilita logging detalhado'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Gera relatório comparativo (útil com --model all --strategy all)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='logs',
        help='Diretório de saída (padrão: logs)'
    )

    # Parse argumentos
    args = parser.parse_args()

    # Ações especiais que saem imediatamente
    if args.list_models:
        print_available_models()
        sys.exit(0)

    if args.list_datasets:
        print_available_datasets()
        sys.exit(0)

    # Banner
    print(f"\n{colorize('='*60, 'cyan')}")
    print(colorize('  Federated Learning - Tree Models com Flower', 'cyan', bold=True))
    print(colorize('='*60, 'cyan'))

    # Cria configurações
    config = GlobalConfig(
        num_clients=args.num_clients,
        num_server_rounds=args.num_rounds,
        num_local_boost_round=args.num_local_rounds,
        sample_per_client=args.sample_per_client,
        seed=args.seed,
    )

    logging_config = LoggingConfig(
        log_dir=args.output_dir,
        save_client_logs=True,
        save_round_logs=True,
        verbose=args.verbose,
    )

    # Valida configuração
    is_valid, errors = validate_config(config)

    if not is_valid:
        print(f"\n{colorize('Erros de Configuração:', 'red', bold=True)}")
        for error in errors:
            print(f"  {colorize('✗', 'red')} {error}")
        print()
        sys.exit(1)

    # Determina experimentos
    experiments = get_experiments_to_run(args.model, args.strategy)

    # Dry run
    if args.dry_run:
        print_dry_run_info(experiments, config)
        sys.exit(0)

    # Executa experimentos
    print(f"\n{colorize('Executando Experimentos', 'green', bold=True)}")
    print(f"Total: {colorize(str(len(experiments)), 'yellow', bold=True)} experimento(s)\n")

    global partial_results

    # Itera experimentos
    iterator = enumerate(experiments, 1)
    if TQDM_AVAILABLE:
        iterator = tqdm(list(iterator), desc="Progresso", unit="exp")

    for i, (model, strategy) in iterator:
        if interrupted:
            print(f"\n{colorize('Interrompido pelo usuário.', 'yellow')}")
            break

        try:
            if not TQDM_AVAILABLE:
                print(f"\n{colorize(f'[{i}/{len(experiments)}]', 'cyan')} "
                      f"Executando: {colorize(model, 'green')} + {colorize(strategy, 'blue')}")

            # Executa experimento
            result = run_single_experiment(
                model=model,
                strategy=strategy,
                config=config,
                logging_config=logging_config,
                dataset_source=args.dataset,
            )

            # Salva resultado
            partial_results[(model, strategy)] = result

            # Salva histórico em JSON
            try:
                json_dir = Path(args.output_dir) / "experiment_histories"
                json_dir.mkdir(parents=True, exist_ok=True)

                json_filename = f"{model}_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                json_path = json_dir / json_filename

                MetricsCalculator.save_history_to_json(
                    history=result.get('history', {}).__dict__ if hasattr(result.get('history', {}), '__dict__') else result.get('history', {}),
                    filepath=str(json_path),
                    metadata={
                        'model': model,
                        'strategy': strategy,
                        'num_clients': config.num_clients,
                        'num_rounds': config.num_server_rounds,
                        'num_local_rounds': config.num_local_boost_round,
                        'dataset': args.dataset,
                        'timestamp': datetime.now().isoformat(),
                    }
                )

                if not TQDM_AVAILABLE:
                    print(f"{colorize('✓', 'green')} Histórico salvo: {colorize(json_filename, 'cyan')}")
            except Exception as e:
                print(f"{colorize('⚠', 'yellow')} Erro ao salvar histórico JSON: {e}")

            # Imprime análise final
            if not TQDM_AVAILABLE:
                try:
                    history = result.get('history', {})
                    if hasattr(history, '__dict__'):
                        # Converte Flower History para dict
                        history_dict = {}
                        if hasattr(history, 'losses_centralized'):
                            history_dict['losses'] = [loss for _, loss in history.losses_centralized]
                        if hasattr(history, 'metrics_centralized'):
                            history_dict['metrics'] = [metrics for _, metrics in history.metrics_centralized.values()]

                        MetricsCalculator.print_final_analysis(
                            strategy_name=strategy,
                            metrics_history=history_dict,
                            num_rounds=config.num_server_rounds
                        )
                except Exception as e:
                    print(f"{colorize('⚠', 'yellow')} Erro ao imprimir análise: {e}")

            # Exibe resumo
            if not TQDM_AVAILABLE:
                print_experiment_summary(model, strategy, result, i, len(experiments))

        except Exception as e:
            print(f"\n{colorize(f'Erro no experimento {model}+{strategy}:', 'red', bold=True)}")
            print(f"{colorize(str(e), 'red')}")

            # Mostra traceback completo para debugging
            import traceback
            traceback.print_exc()

            # Continua com próximo experimento
            continue

    # Gera relatório comparativo
    if args.compare and len(partial_results) > 1:
        print(f"\n{colorize('Gerando Relatório Comparativo...', 'cyan')}")

        output_path = Path(args.output_dir) / 'comparison_report.json'
        report = generate_comparison_report(partial_results, output_path)

        print(f"{colorize('✓', 'green')} Relatório salvo em: {colorize(str(output_path), 'yellow')}")

        # Exibe resumo
        print_comparison_summary(report)

    # Resumo final
    print(f"\n{colorize('='*60, 'cyan')}")
    print(f"{colorize('Experimentos Concluídos', 'green', bold=True)}")
    print(f"Total executado: {colorize(str(len(partial_results)), 'yellow', bold=True)}/{len(experiments)}")
    print(f"Logs salvos em: {colorize(args.output_dir, 'cyan')}")
    print(colorize('='*60, 'cyan'))
    print()


if __name__ == "__main__":
    main()
