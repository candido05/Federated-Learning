from .config import ALGORITHMS, STRATEGIES, PLOTS_DIR
from .data_loader import (find_most_recent_experiment, load_metrics_json,
                          load_all_experiments, extract_metrics_by_round)
from .single_plots import plot_metrics_evolution, plot_confusion_matrix, plot_per_class_metrics
from .comparison_plots import (plot_algorithm_comparison, plot_final_metrics_comparison,
                               plot_training_time_comparison)


def generate_single_algorithm_plots(algorithm: str, strategy: str) -> bool:
    exp_dir = find_most_recent_experiment(algorithm, strategy)
    if not exp_dir:
        print(f"[AVISO] Nenhum experimento encontrado")
        return False

    print(f"Diretorio: {exp_dir.name}")

    data = load_metrics_json(exp_dir)
    if not data:
        print(f"[AVISO] metrics.json nao encontrado")
        return False

    rounds, metrics = extract_metrics_by_round(data)
    exp_info = data['experiment_info']

    exp_timestamp = exp_dir.name.split('_')[0] + '_' + exp_dir.name.split('_')[1]

    plot_file = plot_metrics_evolution(algorithm, strategy, rounds, metrics, exp_info, exp_timestamp)
    print(f"[OK] Plot salvo: {plot_file}")

    cm_file = plot_confusion_matrix(algorithm, strategy, data, exp_timestamp)
    if cm_file:
        print(f"[OK] Matriz de confusao salva: {cm_file}")

    per_class_file = plot_per_class_metrics(algorithm, strategy, rounds, data, exp_timestamp)
    print(f"[OK] Metricas por classe salvas: {per_class_file}")

    return True


def generate_comparison_plots(all_data: dict, comparison_timestamp: str = None):
    if not all_data:
        print("[AVISO] Nenhum dado disponivel para comparacao")
        return

    from datetime import datetime
    if comparison_timestamp is None:
        comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    comp_files = plot_algorithm_comparison(all_data, comparison_timestamp)
    for f in comp_files:
        print(f"[OK] Comparacao salva: {f}")

    final_file = plot_final_metrics_comparison(all_data, comparison_timestamp)
    print(f"[OK] Comparacao de metricas finais salva: {final_file}")

    time_file = plot_training_time_comparison(all_data, comparison_timestamp)
    print(f"[OK] Comparacao de tempo salva: {time_file}")


def main():
    print("="*80)
    print("GERADOR DE PLOTS DE METRICAS - FEDERATED LEARNING")
    print("="*80)
    print()

    for algorithm in ALGORITHMS:
        print(f"\n[{algorithm.upper()}]")
        for strategy in STRATEGIES:
            print(f"Estrategia: {strategy}")
            generate_single_algorithm_plots(algorithm, strategy)

    print(f"\n[COMPARACOES GERAIS]")
    all_data = load_all_experiments()
    generate_comparison_plots(all_data)

    print()
    print("="*80)
    print("[OK] GERACAO DE PLOTS CONCLUIDA!")
    print(f"[OK] Plots salvos em: {PLOTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
