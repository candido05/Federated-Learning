#!/usr/bin/env python3
"""
Add Section 11: Fairness Analysis to the notebook
"""

import json

notebook_path = r"c:\Users\candi\OneDrive\Desktop\Federated-Learning\Code\novo_tcc_code\bank_account_fraud_federated_learning.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Current notebook has {len(nb['cells'])} cells")

# Create new cells for Section 11: Fairness Analysis

fairness_cells = []

# Cell 1: Section header (markdown)
fairness_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## Seção 11: Análise de Fairness no Federated Learning\n",
        "\n",
        "### Objetivo:\n",
        "Avaliar se os modelos federalizados apresentam viés contra clientes com idade > 50 anos.\n",
        "\n",
        "### Métricas de Fairness:\n",
        "- **FPR (False Positive Rate)**: Taxa de clientes legítimos marcados como fraude\n",
        "- **Fairness Ratio**: FPR_old / FPR_young (ideal = 1.0, sem viés)\n",
        "- **Meta**: Fairness Ratio próximo de 1.0 (sem discriminação)\n",
        "\n",
        "### Estratégias de Mitigação:\n",
        "1. **Threshold Decoupling**: Thresholds diferentes por grupo etário\n",
        "2. **Sample Weighting**: Aumentar peso de clientes legítimos > 50 anos no treino\n",
        "3. **Combinação**: Aplicar ambas as estratégias"
    ]
})

# Cell 2: Utility functions (code)
fairness_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# FUNÇÕES AUXILIARES PARA ANÁLISE DE FAIRNESS\n",
        "# ============================================================================\n",
        "\n",
        "def calc_fairness_metrics(y_true, y_prob, age_flag, fpr_target=0.05):\n",
        "    \"\"\"\n",
        "    Calcula métricas de fairness para um modelo.\n",
        "    \n",
        "    Args:\n",
        "        y_true: Labels verdadeiros (0=legítimo, 1=fraude)\n",
        "        y_prob: Probabilidades preditas\n",
        "        age_flag: Flag binária (0=<=50 anos, 1=>50 anos)\n",
        "        fpr_target: FPR alvo para threshold global\n",
        "    \n",
        "    Returns:\n",
        "        dict: Métricas de fairness\n",
        "    \"\"\"\n",
        "    # Threshold global\n",
        "    threshold = get_threshold_at_fpr(y_true, y_prob, fpr_target)\n",
        "    y_pred = (y_prob >= threshold).astype(int)\n",
        "    \n",
        "    # Máscaras de idade\n",
        "    mask_old = age_flag == 1      # Idade > 50\n",
        "    mask_young = age_flag == 0    # Idade <= 50\n",
        "    \n",
        "    # FPR por grupo\n",
        "    def calc_fpr(y_true_g, y_pred_g):\n",
        "        neg_mask = y_true_g == 0\n",
        "        if neg_mask.sum() == 0:\n",
        "            return 0.0\n",
        "        fp = ((y_pred_g == 1) & (y_true_g == 0)).sum()\n",
        "        return fp / neg_mask.sum()\n",
        "    \n",
        "    fpr_old = calc_fpr(y_true[mask_old], y_pred[mask_old])\n",
        "    fpr_young = calc_fpr(y_true[mask_young], y_pred[mask_young])\n",
        "    \n",
        "    fairness_ratio = fpr_old / fpr_young if fpr_young > 0 else np.inf\n",
        "    \n",
        "    # Métricas gerais\n",
        "    auc = roc_auc_score(y_true, y_prob)\n",
        "    tpr = calc_tpr_at_fpr(y_true, y_prob, fpr_target)\n",
        "    \n",
        "    return {\n",
        "        'threshold': threshold,\n",
        "        'auc': auc,\n",
        "        'tpr_at_5fpr': tpr,\n",
        "        'fpr_old': fpr_old,\n",
        "        'fpr_young': fpr_young,\n",
        "        'fairness_ratio': fairness_ratio\n",
        "    }\n",
        "\n",
        "\n",
        "def get_group_threshold_at_fpr(y_true, y_prob, group_mask, fpr_target=0.05):\n",
        "    \"\"\"\n",
        "    Calcula threshold específico para um grupo que resulta em FPR <= fpr_target.\n",
        "    \"\"\"\n",
        "    y_true_g = y_true[group_mask]\n",
        "    y_prob_g = y_prob[group_mask]\n",
        "    \n",
        "    if len(y_true_g) == 0:\n",
        "        return 0.5\n",
        "    \n",
        "    fpr, tpr, thresholds = roc_curve(y_true_g, y_prob_g)\n",
        "    valid_indices = np.where(fpr <= fpr_target)[0]\n",
        "    \n",
        "    if len(valid_indices) == 0:\n",
        "        return 1.0\n",
        "    \n",
        "    best_idx = valid_indices[np.argmax(tpr[valid_indices])]\n",
        "    return thresholds[best_idx]\n",
        "\n",
        "\n",
        "def apply_decoupled_thresholds(y_true, y_prob, age_flag, fpr_target=0.05):\n",
        "    \"\"\"\n",
        "    Aplica thresholds diferentes para grupos etários.\n",
        "    \"\"\"\n",
        "    mask_old = age_flag == 1\n",
        "    mask_young = age_flag == 0\n",
        "    \n",
        "    # Calcular thresholds por grupo\n",
        "    threshold_old = get_group_threshold_at_fpr(y_true, y_prob, mask_old, fpr_target)\n",
        "    threshold_young = get_group_threshold_at_fpr(y_true, y_prob, mask_young, fpr_target)\n",
        "    \n",
        "    # Aplicar thresholds\n",
        "    y_pred = np.zeros(len(y_true), dtype=int)\n",
        "    y_pred[mask_old] = (y_prob[mask_old] >= threshold_old).astype(int)\n",
        "    y_pred[mask_young] = (y_prob[mask_young] >= threshold_young).astype(int)\n",
        "    \n",
        "    return y_pred, threshold_old, threshold_young\n",
        "\n",
        "\n",
        "print(\"Funções de fairness definidas!\")"
    ]
})

# Cell 3: Subsection 11.1 - Evaluate FL models (markdown)
fairness_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "### 11.1 - Avaliação de Fairness dos Modelos FL\n",
        "\n",
        "Primeiro, vamos recuperar os modelos globais finais de cada experimento FL e avaliar suas métricas de fairness no conjunto de teste."
    ]
})

# Cell 4: Combine test data and age_above_50 (code)
fairness_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# PREPARAR DADOS DE TESTE PARA ANÁLISE DE FAIRNESS\n",
        "# ============================================================================\n",
        "\n",
        "print(\"Combinando dados de teste de todos os clientes...\")\n",
        "\n",
        "# Combinar X_test e y_test de todos os clientes\n",
        "X_test_all = np.vstack([test_partitions[i][0] for i in range(NUM_CLIENTS)])\n",
        "y_test_all = np.concatenate([test_partitions[i][1] for i in range(NUM_CLIENTS)])\n",
        "\n",
        "print(f\"X_test_all shape: {X_test_all.shape}\")\n",
        "print(f\"y_test_all shape: {y_test_all.shape}\")\n",
        "\n",
        "# IMPORTANTE: Recuperar age_above_50 para o conjunto de teste\n",
        "# Usamos age_above_50_test_full que foi salvo antes do particionamento\n",
        "age_above_50_test_all = age_above_50_test_full.values\n",
        "\n",
        "print(f\"age_above_50_test_all shape: {age_above_50_test_all.shape}\")\n",
        "print(f\"\\nDistribuição etária no teste:\")\n",
        "print(f\"  <= 50 anos: {(age_above_50_test_all == 0).sum()} ({(age_above_50_test_all == 0).mean()*100:.1f}%)\")\n",
        "print(f\"  > 50 anos: {(age_above_50_test_all == 1).sum()} ({(age_above_50_test_all == 1).mean()*100:.1f}%)\")\n",
        "\n",
        "# Verificar taxa de fraude por grupo etário\n",
        "mask_old = age_above_50_test_all == 1\n",
        "mask_young = age_above_50_test_all == 0\n",
        "\n",
        "fraud_rate_old = y_test_all[mask_old].mean()\n",
        "fraud_rate_young = y_test_all[mask_young].mean()\n",
        "\n",
        "print(f\"\\nTaxa de fraude por grupo:\")\n",
        "print(f\"  <= 50 anos: {fraud_rate_young*100:.2f}%\")\n",
        "print(f\"  > 50 anos: {fraud_rate_old*100:.2f}%\")"
    ]
})

# Cell 5: Note about model evaluation (markdown)
fairness_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**NOTA IMPORTANTE**: Para avaliar os modelos FL, precisaríamos:\n",
        "\n",
        "1. Salvar os modelos globais finais durante a simulação FL\n",
        "2. Carregar esses modelos e fazer predições no X_test_all\n",
        "3. Calcular as métricas de fairness\n",
        "\n",
        "Como a implementação atual do Flower não retorna diretamente o modelo global final da simulação, aqui está o código para **quando você implementar a salvação dos modelos**:\n",
        "\n",
        "```python\n",
        "# Exemplo de como avaliar (quando tiver os modelos salvos)\n",
        "# XGBoost Bagging\n",
        "y_prob_xgb_bagging = xgb_bagging_model.predict(xgb.DMatrix(X_test_all))\n",
        "fairness_xgb_bagging = calc_fairness_metrics(y_test_all, y_prob_xgb_bagging, age_above_50_test_all)\n",
        "\n",
        "# XGBoost Cyclic\n",
        "y_prob_xgb_cyclic = xgb_cyclic_model.predict(xgb.DMatrix(X_test_all))\n",
        "fairness_xgb_cyclic = calc_fairness_metrics(y_test_all, y_prob_xgb_cyclic, age_above_50_test_all)\n",
        "\n",
        "# ... e assim por diante para LightGBM e CatBoost\n",
        "```\n",
        "\n",
        "Por enquanto, vamos demonstrar as estratégias de mitigação de fairness usando um modelo de exemplo."
    ]
})

# Cell 6: Demonstration with dummy model (code)
fairness_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# DEMONSTRAÇÃO: ANÁLISE DE FAIRNESS COM MODELO DE EXEMPLO\n",
        "# ============================================================================\n",
        "\n",
        "print(\"DEMONSTRAÇÃO: Treinando modelo XGBoost local para análise de fairness...\")\n",
        "print(\"(Em produção, você usaria os modelos globais salvos do FL)\\n\")\n",
        "\n",
        "# Treinar modelo local para demonstração\n",
        "dtrain_demo = xgb.DMatrix(X_train_final, label=y_train.values)\n",
        "dtest_demo = xgb.DMatrix(X_test_all, label=y_test_all)\n",
        "\n",
        "demo_model = xgb.train(\n",
        "    xgb_params,\n",
        "    dtrain_demo,\n",
        "    num_boost_round=100,\n",
        "    verbose_eval=False\n",
        ")\n",
        "\n",
        "# Predições no teste\n",
        "y_prob_demo = demo_model.predict(dtest_demo)\n",
        "\n",
        "# Avaliar fairness do modelo baseline\n",
        "print(\"=\" * 70)\n",
        "print(\"MODELO BASELINE (SEM MITIGAÇÃO DE FAIRNESS)\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "fairness_baseline = calc_fairness_metrics(y_test_all, y_prob_demo, age_above_50_test_all)\n",
        "\n",
        "print(f\"\\nMétricas Gerais:\")\n",
        "print(f\"  AUC: {fairness_baseline['auc']:.4f}\")\n",
        "print(f\"  TPR @ 5% FPR: {fairness_baseline['tpr_at_5fpr']:.4f}\")\n",
        "print(f\"  Threshold: {fairness_baseline['threshold']:.4f}\")\n",
        "\n",
        "print(f\"\\nMétricas de Fairness:\")\n",
        "print(f\"  FPR (> 50 anos): {fairness_baseline['fpr_old']*100:.2f}%\")\n",
        "print(f\"  FPR (<= 50 anos): {fairness_baseline['fpr_young']*100:.2f}%\")\n",
        "print(f\"  Fairness Ratio: {fairness_baseline['fairness_ratio']:.2f}\")\n",
        "\n",
        "if fairness_baseline['fairness_ratio'] > 1.2:\n",
        "    print(\"\\n  VIÉS DETECTADO: Modelo discrimina clientes > 50 anos!\")\n",
        "elif fairness_baseline['fairness_ratio'] < 0.8:\n",
        "    print(\"\\n  VIÉS DETECTADO: Modelo discrimina clientes <= 50 anos!\")\n",
        "else:\n",
        "    print(\"\\n  OK: Fairness ratio aceitável (próximo de 1.0)\")"
    ]
})

# Cell 7: Subsection 11.2 - Threshold Decoupling (markdown)
fairness_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "### 11.2 - Estratégia A: Threshold Decoupling\n",
        "\n",
        "Aplicar thresholds diferentes para cada grupo etário, mantendo FPR = 5% **dentro de cada grupo**."
    ]
})

# Cell 8: Apply threshold decoupling (code)
fairness_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# ESTRATÉGIA A: THRESHOLD DECOUPLING\n",
        "# ============================================================================\n",
        "\n",
        "print(\"=\" * 70)\n",
        "print(\"ESTRATÉGIA A: THRESHOLD DECOUPLING\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "# Aplicar thresholds decoupled\n",
        "y_pred_decoupled, threshold_old, threshold_young = apply_decoupled_thresholds(\n",
        "    y_test_all, y_prob_demo, age_above_50_test_all, fpr_target=0.05\n",
        ")\n",
        "\n",
        "print(f\"\\nThresholds por grupo:\")\n",
        "print(f\"  > 50 anos: {threshold_old:.4f}\")\n",
        "print(f\"  <= 50 anos: {threshold_young:.4f}\")\n",
        "\n",
        "# Calcular métricas com thresholds decoupled\n",
        "mask_old = age_above_50_test_all == 1\n",
        "mask_young = age_above_50_test_all == 0\n",
        "\n",
        "def calc_metrics_with_pred(y_true, y_pred, mask):\n",
        "    y_t = y_true[mask]\n",
        "    y_p = y_pred[mask]\n",
        "    \n",
        "    # FPR\n",
        "    neg_mask = y_t == 0\n",
        "    fpr = ((y_p == 1) & (y_t == 0)).sum() / neg_mask.sum() if neg_mask.sum() > 0 else 0\n",
        "    \n",
        "    # TPR\n",
        "    pos_mask = y_t == 1\n",
        "    tpr = ((y_p == 1) & (y_t == 1)).sum() / pos_mask.sum() if pos_mask.sum() > 0 else 0\n",
        "    \n",
        "    return {'fpr': fpr, 'tpr': tpr}\n",
        "\n",
        "metrics_old_decoupled = calc_metrics_with_pred(y_test_all, y_pred_decoupled, mask_old)\n",
        "metrics_young_decoupled = calc_metrics_with_pred(y_test_all, y_pred_decoupled, mask_young)\n",
        "\n",
        "fairness_ratio_decoupled = metrics_old_decoupled['fpr'] / metrics_young_decoupled['fpr'] if metrics_young_decoupled['fpr'] > 0 else np.inf\n",
        "\n",
        "print(f\"\\nMétricas após Threshold Decoupling:\")\n",
        "print(f\"  FPR (> 50 anos): {metrics_old_decoupled['fpr']*100:.2f}%\")\n",
        "print(f\"  FPR (<= 50 anos): {metrics_young_decoupled['fpr']*100:.2f}%\")\n",
        "print(f\"  Fairness Ratio: {fairness_ratio_decoupled:.2f}\")\n",
        "print(f\"\\n  TPR (> 50 anos): {metrics_old_decoupled['tpr']*100:.2f}%\")\n",
        "print(f\"  TPR (<= 50 anos): {metrics_young_decoupled['tpr']*100:.2f}%\")\n",
        "\n",
        "# Overall TPR\n",
        "overall_tpr_decoupled = (y_pred_decoupled[y_test_all == 1] == 1).sum() / (y_test_all == 1).sum()\n",
        "print(f\"\\n  TPR Geral: {overall_tpr_decoupled*100:.2f}%\")"
    ]
})

# Cell 9: Subsection 11.3 - Sample Weighting (markdown)
fairness_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "### 11.3 - Estratégia B: Sample Weighting\n",
        "\n",
        "Re-treinar o modelo aumentando o peso de clientes legítimos (classe 0) com idade > 50 anos."
    ]
})

# Cell 10: Apply sample weighting (code)
fairness_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# ESTRATÉGIA B: SAMPLE WEIGHTING\n",
        "# ============================================================================\n",
        "\n",
        "print(\"=\" * 70)\n",
        "print(\"ESTRATÉGIA B: SAMPLE WEIGHTING\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "# Criar sample weights para o treino\n",
        "# IMPORTANTE: Usamos age_above_50_train_full salvo anteriormente\n",
        "sample_weights_train = np.ones(len(y_train))\n",
        "\n",
        "# Aumentar peso de clientes LEGÍTIMOS (y=0) com idade > 50\n",
        "WEIGHT_OLD_LEGIT = 3.0  # Peso 3x maior\n",
        "\n",
        "age_above_50_train = age_above_50_train_full.values\n",
        "mask_old_legit = (y_train.values == 0) & (age_above_50_train == 1)\n",
        "\n",
        "sample_weights_train[mask_old_legit] = WEIGHT_OLD_LEGIT\n",
        "\n",
        "print(f\"\\nSample weights criados:\")\n",
        "print(f\"  Amostras com peso 1.0: {(sample_weights_train == 1.0).sum()}\")\n",
        "print(f\"  Amostras com peso {WEIGHT_OLD_LEGIT}: {(sample_weights_train == WEIGHT_OLD_LEGIT).sum()}\")\n",
        "print(f\"    (clientes legítimos > 50 anos)\")\n",
        "\n",
        "# Re-treinar modelo com sample weights\n",
        "print(f\"\\nRe-treinando modelo XGBoost com sample weighting...\")\n",
        "\n",
        "dtrain_weighted = xgb.DMatrix(X_train_final, label=y_train.values, weight=sample_weights_train)\n",
        "\n",
        "model_weighted = xgb.train(\n",
        "    xgb_params,\n",
        "    dtrain_weighted,\n",
        "    num_boost_round=100,\n",
        "    verbose_eval=False\n",
        ")\n",
        "\n",
        "# Predições no teste\n",
        "y_prob_weighted = model_weighted.predict(dtest_demo)\n",
        "\n",
        "# Avaliar fairness\n",
        "fairness_weighted = calc_fairness_metrics(y_test_all, y_prob_weighted, age_above_50_test_all)\n",
        "\n",
        "print(f\"\\nMétricas Gerais (com sample weighting):\")\n",
        "print(f\"  AUC: {fairness_weighted['auc']:.4f}\")\n",
        "print(f\"  TPR @ 5% FPR: {fairness_weighted['tpr_at_5fpr']:.4f}\")\n",
        "print(f\"  Threshold: {fairness_weighted['threshold']:.4f}\")\n",
        "\n",
        "print(f\"\\nMétricas de Fairness (com sample weighting):\")\n",
        "print(f\"  FPR (> 50 anos): {fairness_weighted['fpr_old']*100:.2f}%\")\n",
        "print(f\"  FPR (<= 50 anos): {fairness_weighted['fpr_young']*100:.2f}%\")\n",
        "print(f\"  Fairness Ratio: {fairness_weighted['fairness_ratio']:.2f}\")\n",
        "\n",
        "if fairness_weighted['fairness_ratio'] > 1.2:\n",
        "    print(\"\\n  VIÉS DETECTADO: Modelo ainda discrimina clientes > 50 anos\")\n",
        "elif fairness_weighted['fairness_ratio'] < 0.8:\n",
        "    print(\"\\n  VIÉS DETECTADO: Modelo agora discrimina clientes <= 50 anos\")\n",
        "else:\n",
        "    print(\"\\n  MELHORIA: Fairness ratio melhorou!\")"
    ]
})

# Cell 11: Subsection 11.4 - Comparison (markdown)
fairness_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "### 11.4 - Comparação de Todas as Estratégias"
    ]
})

# Cell 12: Comparison table (code)
fairness_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# COMPARAÇÃO FINAL DE ESTRATÉGIAS\n",
        "# ============================================================================\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"COMPARAÇÃO DE ESTRATÉGIAS DE FAIRNESS\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Criar tabela comparativa\n",
        "comparison_data = {\n",
        "    'Estratégia': [\n",
        "        'Baseline (sem mitigação)',\n",
        "        'Threshold Decoupling',\n",
        "        'Sample Weighting'\n",
        "    ],\n",
        "    'AUC': [\n",
        "        fairness_baseline['auc'],\n",
        "        roc_auc_score(y_test_all, y_prob_demo),  # Mesmo modelo, só muda threshold\n",
        "        fairness_weighted['auc']\n",
        "    ],\n",
        "    'TPR@5%FPR': [\n",
        "        fairness_baseline['tpr_at_5fpr'],\n",
        "        overall_tpr_decoupled,\n",
        "        fairness_weighted['tpr_at_5fpr']\n",
        "    ],\n",
        "    'FPR >50': [\n",
        "        fairness_baseline['fpr_old'],\n",
        "        metrics_old_decoupled['fpr'],\n",
        "        fairness_weighted['fpr_old']\n",
        "    ],\n",
        "    'FPR <=50': [\n",
        "        fairness_baseline['fpr_young'],\n",
        "        metrics_young_decoupled['fpr'],\n",
        "        fairness_weighted['fpr_young']\n",
        "    ],\n",
        "    'Fairness Ratio': [\n",
        "        fairness_baseline['fairness_ratio'],\n",
        "        fairness_ratio_decoupled,\n",
        "        fairness_weighted['fairness_ratio']\n",
        "    ]\n",
        "}\n",
        "\n",
        "comparison_df = pd.DataFrame(comparison_data)\n",
        "\n",
        "print(\"\\n\")\n",
        "print(comparison_df.to_string(index=False))\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"INTERPRETAÇÃO\")\n",
        "print(\"=\"*70)\n",
        "print(\"\\nFairness Ratio ideal = 1.0 (sem discriminação)\")\n",
        "print(\"Fairness Ratio > 1.2: Viés contra clientes > 50 anos\")\n",
        "print(\"Fairness Ratio < 0.8: Viés contra clientes <= 50 anos\")\n",
        "print(\"\\nMelhor estratégia: Aquela com Fairness Ratio mais próximo de 1.0\")\n",
        "print(\"                     mantendo TPR@5%FPR alto\")"
    ]
})

# Cell 13: Visualization (code)
fairness_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# VISUALIZAÇÃO: COMPARAÇÃO DE FAIRNESS\n",
        "# ============================================================================\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# Plot 1: Fairness Ratio\n",
        "strategies = comparison_df['Estratégia'].values\n",
        "fairness_ratios = comparison_df['Fairness Ratio'].values\n",
        "\n",
        "colors = ['red' if r > 1.2 or r < 0.8 else 'green' for r in fairness_ratios]\n",
        "\n",
        "axes[0].bar(range(len(strategies)), fairness_ratios, color=colors, edgecolor='black', linewidth=1.5)\n",
        "axes[0].axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Ideal (1.0)')\n",
        "axes[0].axhline(y=1.2, color='orange', linestyle=':', linewidth=1, label='Threshold (1.2)')\n",
        "axes[0].axhline(y=0.8, color='orange', linestyle=':', linewidth=1)\n",
        "axes[0].set_xticks(range(len(strategies)))\n",
        "axes[0].set_xticklabels(strategies, rotation=15, ha='right')\n",
        "axes[0].set_ylabel('Fairness Ratio', fontweight='bold')\n",
        "axes[0].set_title('Fairness Ratio por Estratégia', fontweight='bold')\n",
        "axes[0].legend()\n",
        "axes[0].grid(axis='y', alpha=0.3)\n",
        "\n",
        "# Plot 2: FPR por grupo\n",
        "x = np.arange(len(strategies))\n",
        "width = 0.35\n",
        "\n",
        "fpr_old = comparison_df['FPR >50'].values * 100\n",
        "fpr_young = comparison_df['FPR <=50'].values * 100\n",
        "\n",
        "axes[1].bar(x - width/2, fpr_old, width, label='> 50 anos', color='salmon', edgecolor='black')\n",
        "axes[1].bar(x + width/2, fpr_young, width, label='<= 50 anos', color='skyblue', edgecolor='black')\n",
        "axes[1].set_xticks(x)\n",
        "axes[1].set_xticklabels(strategies, rotation=15, ha='right')\n",
        "axes[1].set_ylabel('FPR (%)', fontweight='bold')\n",
        "axes[1].set_title('FPR por Grupo Etário', fontweight='bold')\n",
        "axes[1].legend()\n",
        "axes[1].grid(axis='y', alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\nGráficos gerados com sucesso!\")"
    ]
})

# Cell 14: Conclusion (markdown)
fairness_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "### 11.5 - Conclusões da Análise de Fairness\n",
        "\n",
        "**Principais Descobertas:**\n",
        "\n",
        "1. **Baseline**: Modelo sem mitigação provavelmente apresenta viés contra clientes > 50 anos\n",
        "2. **Threshold Decoupling**: Equaliza FPR entre grupos aplicando thresholds diferentes\n",
        "3. **Sample Weighting**: Melhora fairness durante o treino, mas pode impactar performance geral\n",
        "\n",
        "**Recomendações para FL:**\n",
        "\n",
        "1. **Durante o Treino FL**:\n",
        "   - Aplicar sample weighting em cada cliente\n",
        "   - Garantir que cada cliente tenha representação balanceada de grupos etários\n",
        "\n",
        "2. **Durante a Inferência**:\n",
        "   - Aplicar threshold decoupling como post-processing\n",
        "   - Monitorar fairness ratio continuamente\n",
        "\n",
        "3. **Governança**:\n",
        "   - Documentar métricas de fairness junto com métricas de performance\n",
        "   - Estabelecer limites aceitáveis para fairness ratio (ex: 0.9 < ratio < 1.1)\n",
        "   - Auditorias regulares de fairness\n",
        "\n",
        "**Próximos Passos:**\n",
        "\n",
        "- Implementar salvação de modelos globais no FL\n",
        "- Avaliar fairness de todos os experimentos FL (XGBoost, LightGBM, CatBoost x Bagging/Cyclic)\n",
        "- Comparar fairness: FL vs. Centralizado\n",
        "- Investigar se estratégias FL (Bagging vs Cyclic) afetam fairness"
    ]
})

# Add all fairness cells to the notebook
print(f"\nAdding {len(fairness_cells)} new cells for Section 11: Fairness Analysis")
nb['cells'].extend(fairness_cells)

print(f"New total cells: {len(nb['cells'])}")

# Save the updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nSection 11: Fairness Analysis added successfully!")
print("\nNotebook structure:")
print("  Sections 1-10: Existing FL implementation")
print("  Section 11: NEW - Fairness Analysis")
print("    11.1: Evaluate FL models fairness")
print("    11.2: Threshold Decoupling")
print("    11.3: Sample Weighting")
print("    11.4: Comparison of strategies")
print("    11.5: Conclusions")
