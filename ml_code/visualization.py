"""
Visualização dos resultados dos experimentos de Federated Learning
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List
import os


def plot_results(results: Dict[str, Dict[str, List[float]]], save_prefix: str = ""):
    """Plota os resultados dos experimentos"""
    
    if not results:
        print("Nenhum resultado para plotar!")
        return
    
    # Organiza os dados por modelo e estratégia
    models = set()
    strategies = set()
    
    for key in results.keys():
        if '_' in key:
            model, strategy = key.split('_', 1)
            models.add(model)
            strategies.add(strategy)
    
    models = sorted(list(models))
    strategies = sorted(list(strategies))
    
    if not models or not strategies:
        print("Dados insuficientes para plotar!")
        return
    
    # Cria figura com subplots
    fig, axes = plt.subplots(len(models), 2, figsize=(16, 6 * len(models)))
    if len(models) == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
    
    for i, model in enumerate(models):
        # Plot de Loss
        ax_loss = axes[i, 0]
        ax_acc = axes[i, 1]
        
        for j, strategy in enumerate(strategies):
            key = f"{model}_{strategy}"
            if key in results and results[key]["rounds"]:
                rounds = results[key]["rounds"]
                loss = results[key]["loss"]
                accuracy = results[key]["accuracy"]
                
                ax_loss.plot(rounds, loss, marker='o', label=strategy, 
                           color=colors[j], linewidth=2, markersize=4)
                ax_acc.plot(rounds, accuracy, marker='o', label=strategy,
                          color=colors[j], linewidth=2, markersize=4)
        
        # Configura plots de loss
        ax_loss.set_title(f'{model.replace("_", " ").title()} - Loss', fontsize=14, fontweight='bold')
        ax_loss.set_xlabel('Rodadas', fontsize=12)
        ax_loss.set_ylabel('Loss', fontsize=12)
        ax_loss.legend(fontsize=10)
        ax_loss.grid(True, alpha=0.3)
        
        # Configura plots de accuracy
        ax_acc.set_title(f'{model.replace("_", " ").title()} - Accuracy', fontsize=14, fontweight='bold')
        ax_acc.set_xlabel('Rodadas', fontsize=12)
        ax_acc.set_ylabel('Accuracy', fontsize=12)
        ax_acc.legend(fontsize=10)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim(0, 1)  # Accuracy vai de 0 a 1
    
    plt.tight_layout()
    
    # Salva o plot
    filename = f"{save_prefix}comparison_plots.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gráficos salvos em: {filename}")


def plot_summary_comparison(results: Dict[str, Dict[str, List[float]]], save_prefix: str = ""):
    """Plota comparação resumida de todos os modelos e estratégias"""
    
    if not results:
        return
    
    # Organiza dados
    models = set()
    strategies = set()
    
    for key in results.keys():
        if '_' in key:
            model, strategy = key.split('_', 1)
            models.add(model)
            strategies.add(strategy)
    
    models = sorted(list(models))
    strategies = sorted(list(strategies))
    
    # Cria matriz de accuracy final para cada combinação
    final_accuracies = np.zeros((len(models), len(strategies)))
    final_losses = np.zeros((len(models), len(strategies)))
    
    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            key = f"{model}_{strategy}"
            if key in results and results[key]["accuracy"]:
                final_accuracies[i, j] = results[key]["accuracy"][-1]  # Última accuracy
                final_losses[i, j] = results[key]["loss"][-1]  # Último loss
    
    # Plot heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Heatmap de Accuracy
    im1 = ax1.imshow(final_accuracies, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Accuracy Final por Modelo e Estratégia', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Estratégias', fontsize=14)
    ax1.set_ylabel('Modelos', fontsize=14)
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, rotation=45)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels([m.replace('_', ' ').title() for m in models])
    
    # Adiciona valores nas células
    for i in range(len(models)):
        for j in range(len(strategies)):
            text = ax1.text(j, i, f'{final_accuracies[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, label='Accuracy')
    
    # Heatmap de Loss
    im2 = ax2.imshow(final_losses, cmap='RdYlGn_r', aspect='auto')
    ax2.set_title('Loss Final por Modelo e Estratégia', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Estratégias', fontsize=14)
    ax2.set_ylabel('Modelos', fontsize=14)
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels(strategies, rotation=45)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels([m.replace('_', ' ').title() for m in models])
    
    # Adiciona valores nas células
    for i in range(len(models)):
        for j in range(len(strategies)):
            text = ax2.text(j, i, f'{final_losses[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, label='Loss')
    
    plt.tight_layout()
    
    # Salva
    filename = f"{save_prefix}summary_heatmap.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Heatmap salvo em: {filename}")


def plot_convergence_analysis(results: Dict[str, Dict[str, List[float]]], save_prefix: str = ""):
    """Analisa a convergência dos algoritmos"""
    
    if not results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cores para diferentes modelos
    models = set()
    for key in results.keys():
        if '_' in key:
            model, _ = key.split('_', 1)
            models.add(model)
    
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_color_map = dict(zip(sorted(models), model_colors))
    
    # Plot de convergência de accuracy
    for key, data in results.items():
        if '_' in key and data["rounds"]:
            model, strategy = key.split('_', 1)
            rounds = data["rounds"]
            accuracy = data["accuracy"]
            
            ax1.plot(rounds, accuracy, marker='o', markersize=3, 
                    label=f'{model}_{strategy}', color=model_color_map[model], alpha=0.7)
    
    ax1.set_title('Convergência da Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Rodadas')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot de convergência de loss
    for key, data in results.items():
        if '_' in key and data["rounds"]:
            model, strategy = key.split('_', 1)
            rounds = data["rounds"]
            loss = data["loss"]
            
            ax2.plot(rounds, loss, marker='o', markersize=3, 
                    label=f'{model}_{strategy}', color=model_color_map[model], alpha=0.7)
    
    ax2.set_title('Convergência da Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Rodadas')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Salva
    filename = f"{save_prefix}convergence_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Análise de convergência salva em: {filename}")


def save_results_to_csv(results: Dict[str, Dict[str, List[float]]], 
                       execution_times: Dict[str, Dict[str, float]] = None,
                       prefix: str = ""):
    """Salva os resultados em arquivo CSV"""
    
    # Prepara dados para CSV
    csv_data = []
    
    for key, data in results.items():
        if '_' in key and data["rounds"]:
            model, strategy = key.split('_', 1)
            
            # Pega tempo de execução se disponível
            exec_time = None
            if execution_times and model in execution_times and strategy in execution_times[model]:
                exec_time = execution_times[model][strategy]
            
            for i in range(len(data["rounds"])):
                row = {
                    'Model': model,
                    'Strategy': strategy,
                    'Round': data["rounds"][i],
                    'Loss': data["loss"][i],
                    'Accuracy': data["accuracy"][i],
                    'Execution_Time': exec_time
                }
                csv_data.append(row)
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        filename = f"{prefix}results.csv"
        df.to_csv(filename, index=False)
        print(f"Resultados salvos em: {filename}")
        
        # Salva também um resumo
        summary_data = []
        for key, data in results.items():
            if '_' in key and data["rounds"]:
                model, strategy = key.split('_', 1)
                
                exec_time = None
                if execution_times and model in execution_times and strategy in execution_times[model]:
                    exec_time = execution_times[model][strategy]
                
                summary_row = {
                    'Model': model,
                    'Strategy': strategy,
                    'Final_Loss': data["loss"][-1] if data["loss"] else None,
                    'Final_Accuracy': data["accuracy"][-1] if data["accuracy"] else None,
                    'Best_Accuracy': max(data["accuracy"]) if data["accuracy"] else None,
                    'Best_Loss': min(data["loss"]) if data["loss"] else None,
                    'Execution_Time_Seconds': exec_time,
                    'Total_Rounds': len(data["rounds"])
                }
                summary_data.append(summary_row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_filename = f"{prefix}summary.csv"
            summary_df.to_csv(summary_filename, index=False)
            print(f"Resumo salvo em: {summary_filename}")
    
    else:
        print("Nenhum dado para salvar!")


def generate_all_plots(results: Dict[str, Dict[str, List[float]]], save_prefix: str = ""):
    """Gera todos os tipos de plot"""
    print("Gerando visualizações...")
    
    plot_results(results, save_prefix)
    plot_summary_comparison(results, save_prefix)
    plot_convergence_analysis(results, save_prefix)
    
    print("Todas as visualizações foram geradas!")


def print_best_results(results: Dict[str, Dict[str, List[float]]]):
    """Imprime os melhores resultados por métrica"""
    
    if not results:
        print("Nenhum resultado disponível!")
        return
    
    best_accuracy = {"key": "", "value": 0}
    best_loss = {"key": "", "value": float('inf')}
    
    print("\n" + "="*60)
    print("RESUMO DOS MELHORES RESULTADOS")
    print("="*60)
    
    for key, data in results.items():
        if '_' in key and data["accuracy"] and data["loss"]:
            final_acc = data["accuracy"][-1]
            final_loss = data["loss"][-1]
            best_acc = max(data["accuracy"])
            best_loss_val = min(data["loss"])
            
            model, strategy = key.split('_', 1)
            
            print(f"\n{model.replace('_', ' ').title()} + {strategy}:")
            print(f"  Accuracy final: {final_acc:.4f} | Melhor: {best_acc:.4f}")
            print(f"  Loss final: {final_loss:.4f} | Melhor: {best_loss_val:.4f}")
            
            if best_acc > best_accuracy["value"]:
                best_accuracy = {"key": key, "value": best_acc}
            
            if best_loss_val < best_loss["value"]:
                best_loss = {"key": key, "value": best_loss_val}
    
    print(f"\n🏆 MELHOR ACCURACY: {best_accuracy['key']} = {best_accuracy['value']:.4f}")
    print(f"🏆 MELHOR LOSS: {best_loss['key']} = {best_loss['value']:.4f}")
    print("="*60)