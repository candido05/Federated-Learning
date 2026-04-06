import matplotlib.pyplot as plt
import os
import numpy as np
import json

def plot_strategy_comparison(bagging_hist, cycling_hist, dataset, output_dir="plots"):
    """Gera gráficos comparativos entre Bagging e Cycling."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ["accuracy", "f1_macro", "auc"]
    # Usando subplots para organizar os 3 gráficos em linha
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, metric in enumerate(metrics):
        b_vals = [r[metric] for r in bagging_hist]
        c_vals = [r[metric] for r in cycling_hist]
        rounds = range(1, len(b_vals) + 1)
        
        axes[i].plot(rounds, b_vals, 'o-', label='Bagging', color='#2196F3', linewidth=2)
        axes[i].plot(rounds, c_vals, 's--', label='Cycling', color='#FF9800', linewidth=2)
        
        axes[i].set_title(f"Métrica: {metric.upper()}")
        axes[i].set_xlabel("Round")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f"XGBoost: Bagging vs Cycling - Dataset: {dataset}", fontsize=14)
    path = os.path.join(output_dir, f"comparison_{dataset}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Comparação salva em: {path}")

def compare_results(dataset):
    """Lê os resultados salvos em JSON e gera a comparação final."""
    try:
        path_bag = f"output/{dataset}_xgboost_bagging/results.json"
        path_cyc = f"output/{dataset}_xgboost_cycling/results.json"
        
        with open(path_bag, "r") as f:
            bag_hist = json.load(f)["rounds"]
        with open(path_cyc, "r") as f:
            cyc_hist = json.load(f)["rounds"]
        
        plot_strategy_comparison(bag_hist, cyc_hist, dataset)
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado ({e.filename}). Execute os scripts .sh primeiro.")