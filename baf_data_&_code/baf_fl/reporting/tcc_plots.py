"""
Geracao de graficos para o TCC - Aprendizado Federado com BAF Dataset.

Gera graficos fundamentais para publicacao cientifica:
1. Curvas de Convergencia (TPR@5%FPR por rodada)
2. Monitoramento de Equidade (Fairness Ratio por rodada)
3. Trade-off Performance vs Fairness (scatter plot)
4. Eficiencia de Comunicacao (barras comparativas)
5. AUC-ROC comparativo (barras)
6. Tempo de execucao (barras)

Uso standalone:
    python -m baf_fl.reporting.tcc_plots [--experiment-dir PATH] [--output-dir PATH]

Uso integrado (chamado automaticamente pelo main.py apos salvar CSVs):
    from baf_fl.reporting.tcc_plots import generate_tcc_plots
    generate_tcc_plots(experiment_dir, output_dir)
"""

import os
import json
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ============================================================================
# Configuracao Visual (estilo para publicacao cientifica / ABNT)
# ============================================================================

MODELS = ['xgboost', 'lightgbm', 'catboost']
MODEL_LABELS = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost'}
STRATEGIES = ['bagging', 'cycling']
STRATEGY_LABELS = {'bagging': 'Bagging', 'cycling': 'Ciclica'}

MODEL_COLORS = {
    'xgboost': '#1f77b4',
    'lightgbm': '#2ca02c',
    'catboost': '#ff7f0e',
}
STRATEGY_COLORS = {
    'bagging': '#4472C4',
    'cycling': '#A5A5A5',
}

STRATEGY_MARKERS = {'bagging': 'o', 'cycling': 's'}
STRATEGY_LINESTYLES = {'bagging': '-', 'cycling': '--'}

BASELINE_TPR_UPPER = 0.55

FIGSIZE_SINGLE = (8, 5.5)
FIGSIZE_WIDE = (12, 5.5)
DPI = 300


def setup_style():
    """Configura estilo visual seaborn para publicacao cientifica."""
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


# ============================================================================
# Carregamento de Dados
# ============================================================================

def find_latest_experiment(experiments_base_dir: str) -> str:
    """
    Encontra o diretorio do experimento mais recente.

    Procura pastas com formato de timestamp (YYYYMMDD_HHMMSS) e retorna
    a mais recente.
    """
    if not os.path.isdir(experiments_base_dir):
        raise FileNotFoundError(f"Diretorio de experimentos nao encontrado: {experiments_base_dir}")

    subdirs = []
    for name in os.listdir(experiments_base_dir):
        full_path = os.path.join(experiments_base_dir, name)
        if os.path.isdir(full_path):
            # Verificar se contem arquivos de experimento
            json_path = os.path.join(full_path, 'experiment.json')
            if os.path.exists(json_path):
                subdirs.append((name, full_path))

    if not subdirs:
        raise FileNotFoundError(f"Nenhum experimento encontrado em: {experiments_base_dir}")

    # Ordenar por nome (que eh timestamp) e pegar o mais recente
    subdirs.sort(key=lambda x: x[0], reverse=True)
    return subdirs[0][1]


def load_experiment_data(experiment_dir: str):
    """
    Carrega todos os dados do experimento.

    Returns
    -------
    round_data : dict
        Dados por rodada para cada configuracao (model_strategy -> DataFrame).
    final_results : pd.DataFrame
        Resultados finais (teste) de todas as configuracoes.
    summary : pd.DataFrame
        Resumo das simulacoes (tempo, bytes, etc.).
    """
    round_data = {}
    for model in MODELS:
        for strategy in STRATEGIES:
            csv_path = os.path.join(experiment_dir, f'{model}_{strategy}_rounds.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                round_data[f'{model}_{strategy}'] = df

    json_path = os.path.join(experiment_dir, 'experiment.json')
    final_results = None
    if os.path.exists(json_path):
        with open(json_path, encoding='utf-8') as f:
            exp = json.load(f)
        if 'final_results' in exp:
            final_results = pd.DataFrame(exp['final_results'])

    summary_path = os.path.join(experiment_dir, 'simulations_summary.csv')
    summary = None
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)

    return round_data, final_results, summary


# ============================================================================
# Grafico 1: Curvas de Convergencia (TPR@5%FPR por rodada)
# ============================================================================

def plot_convergence(round_data, final_results, output_dir):
    """
    Grafico de linhas: evolucao do TPR@5%FPR ao longo das rodadas.
    Painel esquerdo: threshold standard. Painel direito: threshold fair.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)

    panels = [
        (axes[0], 'tpr_standard', 'Limiar Unico (Standard)'),
        (axes[1], 'tpr_fair', 'Limiar por Grupo (Fair)'),
    ]

    for ax, metric_col, title in panels:
        for key, df in sorted(round_data.items()):
            model, strategy = key.split('_')
            if metric_col not in df.columns:
                continue

            rounds = df['round_num'].values
            values = df[metric_col].values

            label = f"{MODEL_LABELS[model]} {STRATEGY_LABELS[strategy]}"
            ax.plot(
                rounds, values * 100,
                color=MODEL_COLORS[model],
                linestyle=STRATEGY_LINESTYLES[strategy],
                marker=STRATEGY_MARKERS[strategy],
                markersize=5, linewidth=1.8, alpha=0.85,
                label=label,
            )

        ax.axhline(
            y=BASELINE_TPR_UPPER * 100, color='red', linestyle=':',
            linewidth=1.2, alpha=0.7, label=f'Baseline centralizado ({BASELINE_TPR_UPPER*100:.0f}%)',
        )

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Rodada de Comunicacao')
        ax.set_ylabel('TPR @ 5% FPR (%)')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    fig.suptitle(
        'Convergencia: TPR@5%FPR por Rodada de Comunicacao',
        fontsize=14, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(output_dir, 'convergencia_tpr.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")

    # AUC-ROC (barras)
    if final_results is not None:
        df_single = final_results[final_results['threshold'] == 'single'].copy()
        if not df_single.empty:
            fig_auc, ax_auc = plt.subplots(figsize=FIGSIZE_SINGLE)

            colors = [MODEL_COLORS.get(m.split()[0].lower(), '#999') for m in df_single['model']]
            bars = ax_auc.bar(
                df_single['model'], df_single['roc_auc'],
                color=colors, edgecolor='black', linewidth=0.5,
            )
            ax_auc.bar_label(bars, fmt='%.4f', fontsize=9, padding=3)
            ax_auc.set_title('AUC-ROC por Modelo e Estrategia', fontweight='bold')
            ax_auc.set_ylabel('AUC-ROC')
            ax_auc.set_ylim(0.85, min(1.01, df_single['roc_auc'].max() * 1.02))
            plt.xticks(rotation=20, ha='right')
            ax_auc.grid(axis='y', alpha=0.3)
            fig_auc.tight_layout()

            path_auc = os.path.join(output_dir, 'auc_roc_comparacao.png')
            fig_auc.savefig(path_auc)
            plt.close(fig_auc)
            print(f"  [OK] {path_auc}")


# ============================================================================
# Grafico 2: Monitoramento de Equidade (Fairness Ratio por rodada)
# ============================================================================

def plot_fairness_monitoring(round_data, output_dir):
    """
    Grafico de linhas: evolucao do FPR Ratio ao longo das rodadas.
    Painel esquerdo: threshold standard. Painel direito: threshold fair.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)

    panels = [
        (axes[0], 'fairness_standard', 'Limiar Unico (Standard)'),
        (axes[1], 'fairness_fair', 'Limiar por Grupo (Fair)'),
    ]

    for ax, metric_col, title in panels:
        for key, df in sorted(round_data.items()):
            model, strategy = key.split('_')
            if metric_col not in df.columns:
                continue

            rounds = df['round_num'].values
            values = df[metric_col].values

            label = f"{MODEL_LABELS[model]} {STRATEGY_LABELS[strategy]}"
            ax.plot(
                rounds, values,
                color=MODEL_COLORS[model],
                linestyle=STRATEGY_LINESTYLES[strategy],
                marker=STRATEGY_MARKERS[strategy],
                markersize=5, linewidth=1.8, alpha=0.85,
                label=label,
            )

        ax.axhline(
            y=1.0, color='green', linestyle=':', linewidth=1.5,
            alpha=0.6, label='Fairness ideal (1,0)',
        )
        ax.axhspan(0.9, 1.05, alpha=0.08, color='green', label='Faixa aceitavel (>= 0,9)')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Rodada de Comunicacao')
        ax.set_ylabel('FPR Ratio (min/max)')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    fig.suptitle(
        'Monitoramento de Equidade: FPR Ratio por Rodada',
        fontsize=14, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(output_dir, 'fairness_monitoramento.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ============================================================================
# Grafico 3: Trade-off Performance vs Fairness (scatter plot)
# ============================================================================

def plot_tradeoff(final_results, output_dir):
    """
    Scatter plot: TPR@5%FPR (eixo Y) vs FPR Ratio (eixo X).
    Cada ponto representa uma configuracao final (modelo x estrategia x threshold).
    """
    if final_results is None or final_results.empty:
        print("  [AVISO] Sem dados finais para grafico de trade-off")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    threshold_markers = {'single': 'o', 'per_group': 's'}
    threshold_labels = {'single': 'Limiar Unico', 'per_group': 'Limiar por Grupo'}

    for _, row in final_results.iterrows():
        model_name = row['model'].replace(' (Fair)', '')
        model_key = model_name.split()[0].lower()
        threshold = row['threshold']

        color = MODEL_COLORS.get(model_key, '#999999')
        marker = threshold_markers.get(threshold, 'o')

        ax.scatter(
            row['fairness_ratio'], row['tpr_at_5fpr'] * 100,
            s=120, c=color, marker=marker,
            edgecolors='black', linewidth=0.8, zorder=5,
        )

        offset_x = 6
        offset_y = 4 if threshold == 'single' else -10
        ax.annotate(
            model_name,
            (row['fairness_ratio'], row['tpr_at_5fpr'] * 100),
            fontsize=7, ha='left', va='bottom',
            xytext=(offset_x, offset_y), textcoords='offset points',
        )

    ax.axhline(
        y=BASELINE_TPR_UPPER * 100, color='red', linestyle='--',
        linewidth=1, alpha=0.5, label=f'Baseline TPR ({BASELINE_TPR_UPPER*100:.0f}%)',
    )
    ax.axvline(
        x=1.0, color='green', linestyle='--',
        linewidth=1, alpha=0.5, label='Fairness ideal (1,0)',
    )
    ax.axvspan(0.9, 1.05, alpha=0.06, color='green')

    ax.axhspan(
        BASELINE_TPR_UPPER * 100, ax.get_ylim()[1] if ax.get_ylim()[1] > 60 else 60,
        xmin=0, xmax=1, alpha=0.03, color='blue',
    )

    from matplotlib.lines import Line2D
    legend_elements = []
    for model_key, label in MODEL_LABELS.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=MODEL_COLORS[model_key],
                   markersize=10, label=label)
        )
    for threshold, label in threshold_labels.items():
        legend_elements.append(
            Line2D([0], [0], marker=threshold_markers[threshold], color='w',
                   markerfacecolor='gray', markersize=10, label=label)
        )
    legend_elements.extend([
        Line2D([0], [0], color='red', linestyle='--', label=f'Baseline TPR ({BASELINE_TPR_UPPER*100:.0f}%)'),
        Line2D([0], [0], color='green', linestyle='--', label='Fairness ideal (1,0)'),
    ])
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

    ax.set_xlabel('FPR Ratio (min/max)', fontsize=12)
    ax.set_ylabel('TPR @ 5% FPR (%)', fontsize=12)
    ax.set_title('Trade-off: Performance vs Equidade', fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, 'tradeoff_performance_fairness.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ============================================================================
# Grafico 4: Eficiencia de Comunicacao (barras agrupadas)
# ============================================================================

def plot_communication_efficiency(summary, output_dir):
    """
    Grafico de barras agrupadas: volume de dados transferidos (MB) e tempo.
    """
    if summary is None or summary.empty:
        print("  [AVISO] Sem dados de resumo para grafico de eficiencia")
        return

    x = np.arange(len(MODELS))
    width = 0.32

    # --- Volume de comunicacao ---
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    bars_data = {}
    for strategy in STRATEGIES:
        mb_values = []
        for model in MODELS:
            row = summary[
                (summary['model_type'] == model) & (summary['strategy_type'] == strategy)
            ]
            if not row.empty:
                mb_values.append(row['total_bytes_transferred'].values[0] / (1024 * 1024))
            else:
                mb_values.append(0)
        bars_data[strategy] = mb_values

    all_values = []
    for strategy in STRATEGIES:
        for i, model in enumerate(MODELS):
            all_values.append((bars_data[strategy][i], model, strategy))
    min_val, min_model, min_strategy = min(all_values, key=lambda t: t[0])

    for idx, strategy in enumerate(STRATEGIES):
        offset = (idx - 0.5) * width
        values = bars_data[strategy]

        edgecolors = []
        linewidths = []
        for i, model in enumerate(MODELS):
            if model == min_model and strategy == min_strategy:
                edgecolors.append('#2E7D32')
                linewidths.append(2.5)
            else:
                edgecolors.append('black')
                linewidths.append(0.5)

        bars = ax.bar(
            x + offset, values, width,
            label=STRATEGY_LABELS[strategy],
            color=STRATEGY_COLORS[strategy],
            edgecolor=edgecolors, linewidth=linewidths,
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
            )

    ax.set_xlabel('')
    ax.set_ylabel('Volume Total Transferido (MB)', fontsize=12)
    ax.set_title('Eficiencia de Comunicacao por Modelo e Estrategia', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=11)
    ax.legend(title='Estrategia', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    ax.annotate(
        f'Mais eficiente:\n{MODEL_LABELS[min_model]} {STRATEGY_LABELS[min_strategy]}\n({min_val:.2f} MB)',
        xy=(0.98, 0.95), xycoords='axes fraction',
        fontsize=8, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.9),
    )

    fig.tight_layout()
    path = os.path.join(output_dir, 'eficiencia_comunicacao.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")

    # --- Tempo de execucao ---
    fig2, ax2 = plt.subplots(figsize=FIGSIZE_SINGLE)

    for idx, strategy in enumerate(STRATEGIES):
        offset = (idx - 0.5) * width
        time_values = []
        for model in MODELS:
            row = summary[
                (summary['model_type'] == model) & (summary['strategy_type'] == strategy)
            ]
            if not row.empty:
                time_values.append(row['duration_sec'].values[0])
            else:
                time_values.append(0)

        bars = ax2.bar(
            x + offset, time_values, width,
            label=STRATEGY_LABELS[strategy],
            color=STRATEGY_COLORS[strategy],
            edgecolor='black', linewidth=0.5,
        )
        for bar, val in zip(bars, time_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{val:.1f}s',
                ha='center', va='bottom', fontsize=8,
            )

    ax2.set_ylabel('Tempo de Execucao (s)', fontsize=12)
    ax2.set_title('Tempo de Execucao por Modelo e Estrategia', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=11)
    ax2.legend(title='Estrategia', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()

    path2 = os.path.join(output_dir, 'tempo_execucao.png')
    fig2.savefig(path2)
    plt.close(fig2)
    print(f"  [OK] {path2}")


# ============================================================================
# Grafico 5: Curvas ROC (Standard e Fair)
# ============================================================================

def plot_roc_curves(strategies_trained, X_test, y_test, age_test, output_dir):
    """
    Curvas ROC para cada modelo/estrategia, com dois paineis:
    - Esquerdo: curva ROC global (standard)
    - Direito: curvas ROC por grupo demografico (fair)

    Requer acesso aos objetos de estrategia treinados e dados de teste.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    from ..data.metrics import get_threshold_at_fpr, get_group_threshold_at_fpr

    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # --- Painel esquerdo: ROC global ---
    ax = axes[0]

    for key, strategy in sorted(strategies_trained.items()):
        model_type, strategy_type = key.split('_')

        if strategy_type == 'bagging':
            if not strategy.client_models:
                continue
            y_prob = strategy.predict_ensemble(X_test)
        else:
            if strategy.current_model is None:
                continue
            y_prob = strategy.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        label = f"{MODEL_LABELS[model_type]} {STRATEGY_LABELS[strategy_type]} (AUC={auc:.4f})"
        ax.plot(
            fpr, tpr,
            color=MODEL_COLORS[model_type],
            linestyle=STRATEGY_LINESTYLES[strategy_type],
            linewidth=1.8, alpha=0.85,
            label=label,
        )

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.4, label='Aleatório')
    ax.axvline(x=0.05, color='red', linestyle=':', linewidth=1, alpha=0.6, label='FPR = 5%')
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax.set_title('Curva ROC Global (Limiar Unico)', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    # --- Painel direito: ROC por grupo demografico ---
    ax = axes[1]

    # Usar o melhor modelo (XGBoost Cycling) para mostrar as curvas por grupo
    best_key = None
    best_auc = 0
    for key, strategy in strategies_trained.items():
        model_type, strategy_type = key.split('_')
        if strategy_type == 'bagging':
            if not strategy.client_models:
                continue
            y_prob = strategy.predict_ensemble(X_test)
        else:
            if strategy.current_model is None:
                continue
            y_prob = strategy.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        if auc > best_auc:
            best_auc = auc
            best_key = key

    group_colors = {'Jovens (< 50)': '#2196F3', 'Idosos (>= 50)': '#FF5722'}

    for key, strategy in sorted(strategies_trained.items()):
        model_type, strategy_type = key.split('_')

        if strategy_type == 'bagging':
            if not strategy.client_models:
                continue
            y_prob = strategy.predict_ensemble(X_test)
        else:
            if strategy.current_model is None:
                continue
            y_prob = strategy.predict(X_test)

        mask_young = age_test == 0
        mask_old = age_test == 1

        for mask, group_name, color in [
            (mask_young, 'Jovens (< 50)', group_colors['Jovens (< 50)']),
            (mask_old, 'Idosos (>= 50)', group_colors['Idosos (>= 50)']),
        ]:
            if mask.sum() == 0 or y_test[mask].sum() == 0:
                continue

            fpr_g, tpr_g, _ = roc_curve(y_test[mask], y_prob[mask])
            auc_g = roc_auc_score(y_test[mask], y_prob[mask])

            # Apenas mostrar label no melhor modelo para evitar legenda lotada
            label = None
            if key == best_key:
                label = f"{group_name} (AUC={auc_g:.4f})"

            ax.plot(
                fpr_g, tpr_g,
                color=color,
                linestyle=STRATEGY_LINESTYLES[strategy_type],
                linewidth=1.5 if key == best_key else 0.6,
                alpha=0.9 if key == best_key else 0.2,
                label=label,
            )

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.4)
    ax.axvline(x=0.05, color='red', linestyle=':', linewidth=1, alpha=0.6, label='FPR = 5%')
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax.set_title('Curva ROC por Grupo Demografico', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    fig.suptitle('Curvas ROC: Analise Global e por Grupo', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(output_dir, 'curvas_roc.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ============================================================================
# Funcao principal de integracao
# ============================================================================

def generate_tcc_plots(experiment_dir: str, output_dir: str):
    """
    Gera todos os graficos do TCC a partir dos dados de um experimento.

    Chamado automaticamente pelo main.py apos experiment_logger.save(),
    ou manualmente via linha de comando.

    Parameters
    ----------
    experiment_dir : str
        Diretorio contendo os CSVs e experiment.json do experimento.
    output_dir : str
        Diretorio de saida para os graficos (criado automaticamente).
    """
    setup_style()
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("GERACAO DE GRAFICOS TCC - Federated Learning (BAF)")
    print("=" * 60)
    print(f"  Experimento: {experiment_dir}")
    print(f"  Saida:       {output_dir}")
    print()

    round_data, final_results, summary = load_experiment_data(experiment_dir)

    if not round_data:
        print("[ERRO] Nenhum CSV de rodadas encontrado!")
        return

    print(f"  Configuracoes carregadas: {len(round_data)}")
    for key in sorted(round_data.keys()):
        n_rounds = len(round_data[key])
        print(f"    - {key}: {n_rounds} rodadas")
    print()

    print("Gerando graficos...")
    plot_convergence(round_data, final_results, output_dir)
    plot_fairness_monitoring(round_data, output_dir)
    plot_tradeoff(final_results, output_dir)
    plot_communication_efficiency(summary, output_dir)

    print()
    print(f"[OK] Todos os graficos salvos em: {output_dir}/")
    print("=" * 60)


# ============================================================================
# Ponto de entrada standalone
# ============================================================================

def main():
    """Ponto de entrada para execucao standalone via linha de comando."""
    parser = argparse.ArgumentParser(
        description='Gerar graficos do TCC - FL com BAF',
    )
    parser.add_argument(
        '--experiment-dir', type=str, default=None,
        help='Diretorio do experimento (padrao: mais recente em experiments/)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Diretorio de saida (padrao: images/{timestamp_experimento}/)',
    )
    args = parser.parse_args()

    # Determinar diretorio base do projeto
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(this_dir, '..', '..'))
    experiments_base = os.path.join(project_root, 'experiments')
    images_base = os.path.join(project_root, 'images')

    # Encontrar experimento
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
    else:
        experiment_dir = find_latest_experiment(experiments_base)
        print(f"  [Auto] Experimento mais recente: {os.path.basename(experiment_dir)}")

    # Determinar pasta de saida com timestamp do experimento
    experiment_id = os.path.basename(experiment_dir)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(images_base, experiment_id)

    generate_tcc_plots(experiment_dir, output_dir)


if __name__ == '__main__':
    main()
