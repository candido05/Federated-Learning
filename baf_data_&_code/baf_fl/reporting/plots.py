"""
Geracao de graficos de desempenho para o treinamento federado.

Gera visualizacoes de convergencia, fairness, comparacao de modelos e estrategias.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


class PlotGenerator:
    """
    Gera graficos de desempenho para analise de resultados do Federated Learning.

    Parametros
    ----------
    output_dir : str
        Diretorio onde os graficos serao salvos.
    figsize : tuple
        Tamanho padrao das figuras (largura, altura).
    dpi : int
        Resolucao das imagens salvas.
    style : str
        Estilo do seaborn para os graficos.
    """

    # Paleta de cores consistente para modelos e estrategias
    MODEL_COLORS = {
        'xgboost': '#2196F3',
        'lightgbm': '#4CAF50',
        'catboost': '#FF9800',
    }
    STRATEGY_COLORS = {
        'bagging': '#1976D2',
        'cycling': '#D32F2F',
    }
    THRESHOLD_COLORS = {
        'single': '#5C6BC0',
        'per_group': '#26A69A',
    }

    def __init__(
        self,
        output_dir: str = 'images',
        figsize: tuple = (10, 6),
        dpi: int = 150,
        style: str = 'whitegrid',
    ):
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi

        os.makedirs(output_dir, exist_ok=True)
        sns.set_style(style)
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        })

    def _save_and_close(self, fig: plt.Figure, filename: str):
        """Salva a figura e fecha para liberar memoria."""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [Plot] Salvo: {path}")

    # =========================================================================
    # 1. Convergencia por Round
    # =========================================================================

    def plot_convergence_tpr(
        self, strategies_trained: Dict, filename: str = 'convergence_tpr.png',
    ):
        """
        Grafico de convergencia: TPR@5%FPR ao longo dos rounds.

        Mostra a evolucao da metrica principal para cada combinacao modelo x estrategia,
        permitindo comparar a velocidade de convergencia.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for ax, threshold_type, title in [
            (axes[0], 'tpr_standard', 'Threshold Unico'),
            (axes[1], 'tpr_fair', 'Threshold por Grupo (Fair)'),
        ]:
            for key, strategy in strategies_trained.items():
                model_type, strategy_type = key.split('_')
                metrics = strategy.round_metrics

                if not metrics:
                    continue

                rounds = [m['round'] for m in metrics]
                tpr_values = [m[threshold_type] for m in metrics]

                color = self.MODEL_COLORS.get(model_type, '#999999')
                linestyle = '-' if strategy_type == 'bagging' else '--'
                marker = 'o' if strategy_type == 'bagging' else 's'

                ax.plot(
                    rounds, tpr_values,
                    color=color, linestyle=linestyle, marker=marker,
                    markersize=5, linewidth=1.8, alpha=0.85,
                    label=f"{model_type.upper()} {strategy_type.capitalize()}",
                )

            ax.axhline(y=0.52, color='red', linestyle=':', linewidth=1, alpha=0.6, label='Benchmark (0.52)')
            ax.set_title(title)
            ax.set_xlabel('Round')
            ax.set_ylabel('TPR @ 5% FPR')
            ax.legend(fontsize=8, loc='lower right')
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)

        fig.suptitle('Convergencia: TPR@5%FPR por Round', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_and_close(fig, filename)

    def plot_convergence_fairness(
        self, strategies_trained: Dict, filename: str = 'convergence_fairness.png',
    ):
        """
        Grafico de convergencia: Fairness Ratio ao longo dos rounds.

        Mostra como a equidade entre grupos etarios evolui durante o treinamento.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for ax, fairness_key, title in [
            (axes[0], 'fairness_standard', 'Threshold Unico'),
            (axes[1], 'fairness_fair', 'Threshold por Grupo (Fair)'),
        ]:
            for key, strategy in strategies_trained.items():
                model_type, strategy_type = key.split('_')
                metrics = strategy.round_metrics

                if not metrics:
                    continue

                rounds = [m['round'] for m in metrics]
                fairness_values = [m[fairness_key] for m in metrics]

                color = self.MODEL_COLORS.get(model_type, '#999999')
                linestyle = '-' if strategy_type == 'bagging' else '--'
                marker = 'o' if strategy_type == 'bagging' else 's'

                ax.plot(
                    rounds, fairness_values,
                    color=color, linestyle=linestyle, marker=marker,
                    markersize=5, linewidth=1.8, alpha=0.85,
                    label=f"{model_type.upper()} {strategy_type.capitalize()}",
                )

            ax.axhline(y=1.0, color='green', linestyle=':', linewidth=1.5, alpha=0.6, label='Fairness Ideal (1.0)')
            ax.axhspan(0.9, 1.1, alpha=0.08, color='green', label='Faixa Aceitavel')
            ax.set_title(title)
            ax.set_xlabel('Round')
            ax.set_ylabel('Fairness Ratio (FPR_old / FPR_young)')
            ax.legend(fontsize=8, loc='best')
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)

        fig.suptitle('Convergencia: Fairness Ratio por Round', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_and_close(fig, filename)

    # =========================================================================
    # 2. Comparacao Final entre Modelos
    # =========================================================================

    def plot_final_comparison_tpr(
        self, df_results: pd.DataFrame, filename: str = 'comparison_tpr.png',
    ):
        """
        Grafico de barras: TPR@5%FPR final por modelo e tipo de threshold.

        Compara a performance de deteccao de fraude entre todos os modelos.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        df_plot = df_results.copy()
        threshold_labels = {'single': 'Threshold Unico', 'per_group': 'Threshold por Grupo'}
        df_plot['threshold_label'] = df_plot['threshold'].map(threshold_labels)

        palette = [self.THRESHOLD_COLORS['single'], self.THRESHOLD_COLORS['per_group']]

        sns.barplot(
            data=df_plot, x='model', y='tpr_at_5fpr', hue='threshold_label',
            palette=palette, ax=ax, edgecolor='black', linewidth=0.5,
        )

        ax.axhline(y=0.52, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='Benchmark (0.52)')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)

        ax.set_title('Comparacao Final: TPR@5%FPR por Modelo', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('TPR @ 5% FPR')
        ax.legend(title='Tipo de Threshold')
        ax.set_ylim(0, min(1.05, df_plot['tpr_at_5fpr'].max() * 1.15))
        plt.xticks(rotation=25, ha='right')
        fig.tight_layout()
        self._save_and_close(fig, filename)

    def plot_final_comparison_fairness(
        self, df_results: pd.DataFrame, filename: str = 'comparison_fairness.png',
    ):
        """
        Grafico de barras: Fairness Ratio final por modelo e tipo de threshold.

        Compara a equidade entre grupos para cada modelo e estrategia.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        df_plot = df_results.copy()
        threshold_labels = {'single': 'Threshold Unico', 'per_group': 'Threshold por Grupo'}
        df_plot['threshold_label'] = df_plot['threshold'].map(threshold_labels)

        palette = [self.THRESHOLD_COLORS['single'], self.THRESHOLD_COLORS['per_group']]

        sns.barplot(
            data=df_plot, x='model', y='fairness_ratio', hue='threshold_label',
            palette=palette, ax=ax, edgecolor='black', linewidth=0.5,
        )

        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Fairness Ideal (1.0)')
        ax.axhspan(0.9, 1.1, alpha=0.08, color='green')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)

        ax.set_title('Comparacao Final: Fairness Ratio por Modelo', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Fairness Ratio (FPR_old / FPR_young)')
        ax.legend(title='Tipo de Threshold')
        plt.xticks(rotation=25, ha='right')
        fig.tight_layout()
        self._save_and_close(fig, filename)

    # =========================================================================
    # 3. Performance vs Fairness (Tradeoff)
    # =========================================================================

    def plot_tradeoff_performance_fairness(
        self, df_results: pd.DataFrame, filename: str = 'tradeoff_performance_fairness.png',
    ):
        """
        Scatter plot: TPR@5%FPR vs Fairness Ratio.

        Visualiza o tradeoff entre desempenho de deteccao e equidade.
        O quadrante ideal e canto superior direito (alto TPR, fairness ~1.0).
        """
        fig, ax = plt.subplots(figsize=(9, 7))

        markers = {'single': 'o', 'per_group': 's'}
        for _, row in df_results.iterrows():
            marker = markers.get(row['threshold'], 'o')
            ax.scatter(
                row['fairness_ratio'], row['tpr_at_5fpr'],
                s=120, marker=marker, edgecolors='black', linewidth=0.8,
                zorder=5,
            )
            ax.annotate(
                row['model'], (row['fairness_ratio'], row['tpr_at_5fpr']),
                fontsize=7, ha='left', va='bottom',
                xytext=(5, 5), textcoords='offset points',
            )

        ax.axhline(y=0.52, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Benchmark TPR (0.52)')
        ax.axvline(x=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Fairness Ideal (1.0)')
        ax.axvspan(0.9, 1.1, alpha=0.06, color='green')

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=10, label='Threshold Unico'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=10, label='Threshold por Grupo'),
            Line2D([0], [0], color='red', linestyle='--', label='Benchmark TPR (0.52)'),
            Line2D([0], [0], color='green', linestyle='--', label='Fairness Ideal (1.0)'),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

        ax.set_xlabel('Fairness Ratio (FPR_old / FPR_young)')
        ax.set_ylabel('TPR @ 5% FPR')
        ax.set_title('Tradeoff: Performance vs Fairness', fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save_and_close(fig, filename)

    # =========================================================================
    # 4. Comparacao Bagging vs Cycling
    # =========================================================================

    def plot_strategy_comparison(
        self, df_results: pd.DataFrame, filename: str = 'strategy_comparison.png',
    ):
        """
        Grafico agrupado: compara Bagging vs Cycling para cada modelo.

        Usa apenas resultados com threshold unico para comparacao justa entre estrategias.
        """
        df_single = df_results[df_results['threshold'] == 'single'].copy()

        if df_single.empty:
            print("  [Plot] AVISO: Sem dados de threshold unico para comparacao de estrategias")
            return

        # Extrair modelo base e estrategia do campo 'model'
        df_single['base_model'] = df_single['model'].apply(
            lambda x: x.split()[0]
        )
        df_single['strategy'] = df_single['model'].apply(
            lambda x: x.split()[1] if len(x.split()) > 1 else 'Unknown'
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # TPR
        sns.barplot(
            data=df_single, x='base_model', y='tpr_at_5fpr', hue='strategy',
            palette=[self.STRATEGY_COLORS['bagging'], self.STRATEGY_COLORS['cycling']],
            ax=axes[0], edgecolor='black', linewidth=0.5,
        )
        axes[0].axhline(y=0.52, color='red', linestyle='--', linewidth=1, alpha=0.6)
        axes[0].set_title('TPR@5%FPR por Estrategia', fontweight='bold')
        axes[0].set_xlabel('Modelo')
        axes[0].set_ylabel('TPR @ 5% FPR')
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.3f', fontsize=8, padding=2)

        # Fairness
        sns.barplot(
            data=df_single, x='base_model', y='fairness_ratio', hue='strategy',
            palette=[self.STRATEGY_COLORS['bagging'], self.STRATEGY_COLORS['cycling']],
            ax=axes[1], edgecolor='black', linewidth=0.5,
        )
        axes[1].axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
        axes[1].axhspan(0.9, 1.1, alpha=0.08, color='green')
        axes[1].set_title('Fairness Ratio por Estrategia', fontweight='bold')
        axes[1].set_xlabel('Modelo')
        axes[1].set_ylabel('Fairness Ratio')
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt='%.3f', fontsize=8, padding=2)

        fig.suptitle('Comparacao: Bagging vs Cycling', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_and_close(fig, filename)

    # =========================================================================
    # 5. Impacto do Threshold por Grupo (Fair vs Standard)
    # =========================================================================

    def plot_threshold_impact(
        self, df_results: pd.DataFrame, filename: str = 'threshold_impact.png',
    ):
        """
        Grafico mostrando o impacto da correcao de fairness (threshold por grupo).

        Compara TPR e Fairness antes e depois de aplicar limiares por grupo.
        """
        models_single = df_results[df_results['threshold'] == 'single'].copy()
        models_fair = df_results[df_results['threshold'] == 'per_group'].copy()

        if models_single.empty or models_fair.empty:
            print("  [Plot] AVISO: Dados insuficientes para comparacao de threshold")
            return

        # Extrair nome base (sem " (Fair)")
        models_fair['model'] = models_fair['model'].str.replace(' (Fair)', '', regex=False)

        merged = models_single.merge(
            models_fair, on='model', suffixes=('_single', '_fair'),
        )

        if merged.empty:
            print("  [Plot] AVISO: Nao foi possivel alinhar modelos single/fair")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        x = np.arange(len(merged))
        width = 0.35

        # TPR
        axes[0].bar(
            x - width / 2, merged['tpr_at_5fpr_single'], width,
            label='Threshold Unico', color=self.THRESHOLD_COLORS['single'],
            edgecolor='black', linewidth=0.5,
        )
        axes[0].bar(
            x + width / 2, merged['tpr_at_5fpr_fair'], width,
            label='Threshold por Grupo', color=self.THRESHOLD_COLORS['per_group'],
            edgecolor='black', linewidth=0.5,
        )
        axes[0].axhline(y=0.52, color='red', linestyle='--', linewidth=1, alpha=0.6)
        axes[0].set_title('Impacto no TPR@5%FPR', fontweight='bold')
        axes[0].set_ylabel('TPR @ 5% FPR')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(merged['model'], rotation=25, ha='right', fontsize=9)
        axes[0].legend()

        # Fairness
        axes[1].bar(
            x - width / 2, merged['fairness_ratio_single'], width,
            label='Threshold Unico', color=self.THRESHOLD_COLORS['single'],
            edgecolor='black', linewidth=0.5,
        )
        axes[1].bar(
            x + width / 2, merged['fairness_ratio_fair'], width,
            label='Threshold por Grupo', color=self.THRESHOLD_COLORS['per_group'],
            edgecolor='black', linewidth=0.5,
        )
        axes[1].axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
        axes[1].axhspan(0.9, 1.1, alpha=0.08, color='green')
        axes[1].set_title('Impacto no Fairness Ratio', fontweight='bold')
        axes[1].set_ylabel('Fairness Ratio')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(merged['model'], rotation=25, ha='right', fontsize=9)
        axes[1].legend()

        fig.suptitle('Impacto da Correcao de Fairness (Threshold por Grupo)', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_and_close(fig, filename)

    # =========================================================================
    # 6. Heatmap de Resultados
    # =========================================================================

    def plot_heatmap_results(
        self, df_results: pd.DataFrame, filename: str = 'heatmap_results.png',
    ):
        """
        Heatmap com TPR@5%FPR para cada modelo x threshold.

        Visao rapida e compacta de todos os resultados finais.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        pivot = df_results.pivot_table(
            values='tpr_at_5fpr', index='model', columns='threshold', aggfunc='first',
        )

        column_labels = {'single': 'Threshold Unico', 'per_group': 'Threshold por Grupo'}
        pivot = pivot.rename(columns=column_labels)

        sns.heatmap(
            pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, linecolor='gray',
            cbar_kws={'label': 'TPR @ 5% FPR'},
            vmin=0, vmax=1,
        )

        ax.set_title('Heatmap: TPR@5%FPR por Modelo e Threshold', fontweight='bold')
        ax.set_ylabel('')
        ax.set_xlabel('')
        fig.tight_layout()
        self._save_and_close(fig, filename)

    # =========================================================================
    # 7. ROC AUC Comparison
    # =========================================================================

    def plot_roc_auc_comparison(
        self, df_results: pd.DataFrame, filename: str = 'roc_auc_comparison.png',
    ):
        """
        Grafico de barras: ROC AUC por modelo.

        Usa apenas threshold unico (ROC AUC e independente do threshold).
        """
        df_single = df_results[df_results['threshold'] == 'single'].copy()

        if df_single.empty:
            print("  [Plot] AVISO: Sem dados para ROC AUC")
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = []
        for model_name in df_single['model']:
            for key, color in self.MODEL_COLORS.items():
                if key.upper() in model_name.upper():
                    colors.append(color)
                    break
            else:
                colors.append('#999999')

        bars = ax.bar(
            df_single['model'], df_single['roc_auc'],
            color=colors, edgecolor='black', linewidth=0.5,
        )

        ax.bar_label(bars, fmt='%.4f', fontsize=9, padding=3)

        ax.set_title('ROC AUC por Modelo e Estrategia', fontweight='bold')
        ax.set_ylabel('ROC AUC')
        ax.set_xlabel('')
        ax.set_ylim(0.5, min(1.02, df_single['roc_auc'].max() * 1.05))
        plt.xticks(rotation=25, ha='right')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        self._save_and_close(fig, filename)

    # =========================================================================
    # 8. Cycling: Evolucao por Cliente
    # =========================================================================

    def plot_cycling_client_contribution(
        self, strategies_trained: Dict, filename: str = 'cycling_client_contribution.png',
    ):
        """
        Grafico da evolucao do Cycling mostrando qual cliente treinou em cada round.

        Especifico para a estrategia Cycling, mostra a contribuicao sequencial.
        """
        cycling_keys = [k for k in strategies_trained if 'cycling' in k]

        if not cycling_keys:
            print("  [Plot] AVISO: Nenhuma estrategia Cycling encontrada")
            return

        n_plots = len(cycling_keys)
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5), squeeze=False)

        for idx, key in enumerate(cycling_keys):
            ax = axes[0][idx]
            strategy = strategies_trained[key]
            metrics = strategy.round_metrics
            model_type = key.split('_')[0]

            if not metrics:
                ax.set_title(f"{model_type.upper()} Cycling - Sem dados")
                continue

            rounds = [m['round'] for m in metrics]
            tpr_values = [m['tpr_standard'] for m in metrics]
            clients = [m.get('trained_client', 0) for m in metrics]

            scatter = ax.scatter(
                rounds, tpr_values, c=clients, cmap='Set1',
                s=80, edgecolors='black', linewidth=0.5, zorder=5,
            )
            ax.plot(rounds, tpr_values, color='gray', linewidth=1, alpha=0.5, zorder=1)

            ax.axhline(y=0.52, color='red', linestyle='--', linewidth=1, alpha=0.5)

            cbar = fig.colorbar(scatter, ax=ax, label='Cliente ID')
            cbar.set_ticks(sorted(set(clients)))

            ax.set_title(f'{model_type.upper()} Cycling', fontweight='bold')
            ax.set_xlabel('Round')
            ax.set_ylabel('TPR @ 5% FPR')
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)

        fig.suptitle('Cycling: Contribuicao Sequencial por Cliente', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_and_close(fig, filename)

    # =========================================================================
    # 9. Resumo Compacto (Dashboard)
    # =========================================================================

    def plot_summary_dashboard(
        self, df_results: pd.DataFrame, strategies_trained: Dict,
        filename: str = 'summary_dashboard.png',
    ):
        """
        Dashboard compacto com 4 paineis resumindo os resultados principais.

        1. TPR final por modelo (barras)
        2. Fairness final por modelo (barras)
        3. Convergencia TPR (linhas)
        4. Tradeoff Performance vs Fairness (scatter)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Painel 1: TPR Final
        ax = axes[0, 0]
        df_single = df_results[df_results['threshold'] == 'single']
        if not df_single.empty:
            colors = []
            for model_name in df_single['model']:
                for key, color in self.MODEL_COLORS.items():
                    if key.upper() in model_name.upper():
                        colors.append(color)
                        break
                else:
                    colors.append('#999999')
            bars = ax.bar(
                range(len(df_single)), df_single['tpr_at_5fpr'].values,
                color=colors, edgecolor='black', linewidth=0.5,
            )
            ax.set_xticks(range(len(df_single)))
            ax.set_xticklabels(df_single['model'].values, rotation=25, ha='right', fontsize=8)
            ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
            ax.axhline(y=0.52, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('TPR@5%FPR (Threshold Unico)', fontweight='bold')
        ax.set_ylabel('TPR')

        # Painel 2: Fairness Final
        ax = axes[0, 1]
        if not df_single.empty:
            bars = ax.bar(
                range(len(df_single)), df_single['fairness_ratio'].values,
                color=colors, edgecolor='black', linewidth=0.5,
            )
            ax.set_xticks(range(len(df_single)))
            ax.set_xticklabels(df_single['model'].values, rotation=25, ha='right', fontsize=8)
            ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
            ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
            ax.axhspan(0.9, 1.1, alpha=0.08, color='green')
        ax.set_title('Fairness Ratio (Threshold Unico)', fontweight='bold')
        ax.set_ylabel('Fairness Ratio')

        # Painel 3: Convergencia TPR
        ax = axes[1, 0]
        for key, strategy in strategies_trained.items():
            model_type, strategy_type = key.split('_')
            metrics = strategy.round_metrics
            if not metrics:
                continue
            rounds = [m['round'] for m in metrics]
            tpr_values = [m['tpr_standard'] for m in metrics]
            color = self.MODEL_COLORS.get(model_type, '#999999')
            linestyle = '-' if strategy_type == 'bagging' else '--'
            ax.plot(
                rounds, tpr_values,
                color=color, linestyle=linestyle, linewidth=1.5, alpha=0.85,
                label=f"{model_type.upper()} {strategy_type.capitalize()}",
            )
        ax.axhline(y=0.52, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_title('Convergencia TPR@5%FPR', fontweight='bold')
        ax.set_xlabel('Round')
        ax.set_ylabel('TPR')
        ax.legend(fontsize=7, loc='lower right')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        # Painel 4: Tradeoff
        ax = axes[1, 1]
        markers = {'single': 'o', 'per_group': 's'}
        for _, row in df_results.iterrows():
            marker = markers.get(row['threshold'], 'o')
            ax.scatter(
                row['fairness_ratio'], row['tpr_at_5fpr'],
                s=80, marker=marker, edgecolors='black', linewidth=0.5, zorder=5,
            )
            ax.annotate(
                row['model'], (row['fairness_ratio'], row['tpr_at_5fpr']),
                fontsize=6, ha='left', va='bottom',
                xytext=(4, 4), textcoords='offset points',
            )
        ax.axhline(y=0.52, color='red', linestyle='--', linewidth=1, alpha=0.4)
        ax.axvline(x=1.0, color='green', linestyle='--', linewidth=1, alpha=0.4)
        ax.set_title('Tradeoff: Performance vs Fairness', fontweight='bold')
        ax.set_xlabel('Fairness Ratio')
        ax.set_ylabel('TPR @ 5% FPR')
        ax.grid(True, alpha=0.3)

        fig.suptitle('Dashboard: Resultados do Federated Learning', fontsize=16, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_and_close(fig, filename)

    # =========================================================================
    # Metodo Principal: Gerar todos os graficos
    # =========================================================================

    def generate_all(
        self,
        df_results: pd.DataFrame,
        strategies_trained: Dict,
    ):
        """
        Gera todos os graficos de desempenho de uma vez.

        Parametros
        ----------
        df_results : pd.DataFrame
            DataFrame com resultados finais (model, threshold, tpr_at_5fpr, roc_auc, fairness_ratio).
        strategies_trained : Dict
            Dicionario de estrategias treinadas (chave: '{model}_{strategy}', valor: Strategy).
        """
        print("\n" + "=" * 70)
        print("GERANDO GRAFICOS DE DESEMPENHO")
        print("=" * 70)

        self.plot_convergence_tpr(strategies_trained)
        self.plot_convergence_fairness(strategies_trained)
        self.plot_final_comparison_tpr(df_results)
        self.plot_final_comparison_fairness(df_results)
        self.plot_tradeoff_performance_fairness(df_results)
        self.plot_strategy_comparison(df_results)
        self.plot_threshold_impact(df_results)
        self.plot_heatmap_results(df_results)
        self.plot_roc_auc_comparison(df_results)
        self.plot_cycling_client_contribution(strategies_trained)
        self.plot_summary_dashboard(df_results, strategies_trained)

        print(f"\n  [Plot] Todos os graficos salvos em: {self.output_dir}/")
