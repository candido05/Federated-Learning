# FASE 3 - ANÁLISE CRÍTICA E MELHORIAS AVANÇADAS

**Data**: 2025-12-22
**Status**: EM IMPLEMENTAÇÃO

---

## 📊 ANÁLISE DOS RESULTADOS ANTERIORES

### Problema Principal Identificado

Após implementação de FASE 1 e FASE 2, os resultados **NÃO atingiram as metas**:

| Métrica | Meta FASE 2 | XGBoost Real | LightGBM Real | CatBoost Real | Status |
|---------|-------------|--------------|---------------|---------------|--------|
| Balanced Acc | > 72% | 63.28% | 65.08% | 62.87% | ❌ **-8 a -9pp** |
| F1-Score Macro | > 0.62 | 0.5127 | 0.4794 | 0.4649 | ❌ **-17 a -25%** |
| Recall Classe 2 | > 70% | 56.13% | ~60% | ~60% | ❌ **-10 a -14pp** |

### Root Causes Identificadas

#### 1. **DEGRADAÇÃO NAS RODADAS FINAIS** ⚠️

**Evidência**: Todos os algoritmos pioraram em R18-R20

| Algoritmo | Melhor Round | F1 Melhor | Round 20 | F1 R20 | Degradação |
|-----------|--------------|-----------|----------|--------|------------|
| XGBoost | R13 | 0.5287 | R20 | 0.5127 | **-3%** ❌ |
| LightGBM | R14 | 0.5220 | R20 | 0.4794 | **-8.2%** ❌❌ |
| CatBoost | R18 | 0.4664 | R20 | 0.4649 | -0.3% ⚠️ |

**Causa**: **OVERFITTING LOCAL** - Sem early stopping, clientes continuam treinando mesmo após convergência ótima.

**Solução**: ✅ Implementar early stopping local (TODO 3.1)

---

#### 2. **TRADE-OFF: REALISMO FL vs PERFORMANCE** 🔄

**Configuração Atual (FASE 2)**:
- `num_clients: 10` (realismo FL ✅)
- `vehicles_per_client: 15`
- **Amostras por cliente: ~4,230** ⚠️

**Baseline Anterior**:
- `num_clients: 3` (menos realista ❌)
- `vehicles_per_client: 40`
- **Amostras por cliente: ~102k** ✅

**Impacto**:
```
FASE 2 (10 clientes):  Balanced Acc 63.28%  F1 0.5127
Baseline (3 clientes): Balanced Acc 66.78%  F1 0.4979
```

**Análise**:
- ✅ F1-Score melhorou +3% com 10 clientes (mais heterogeneidade)
- ❌ Balanced Accuracy **piorou** -3.5pp (menos dados por cliente)

**Conclusão**: **Reduzir para 7 clientes** (sweet spot entre realismo e dados suficientes)

---

#### 3. **CatBoost NÃO APROVEITOU OTIMIZAÇÕES** ❌

**Esperado**: `auto_class_weights` + `border_count=128` → CatBoost ≈ LightGBM

**Realidade**:
```
LightGBM: F1 0.4794, Balanced Acc 65.08%
CatBoost: F1 0.4649, Balanced Acc 62.87%  (-3% PIOR!)
```

**Hipóteses**:
1. **`learning_rate: 0.03` muito alto** - CatBoost precisa LR ainda mais baixo
2. **Conflito entre `auto_class_weights` e sample weights manuais** - pode estar duplicando penalização
3. **`border_count: 128` insuficiente** - Aumentar para 254 (máximo)
4. **Convergência lenta** - CatBoost precisa de mais rodadas (25-30) que XGB/LGBM

**Soluções**:
- ✅ Reduzir LR CatBoost: 0.03 → **0.02**
- ✅ Aumentar border_count: 128 → **254**
- ✅ Remover `auto_class_weights` (conflito com class weights manuais)
- ✅ Aumentar rounds: 20 → **25** (apenas para CatBoost)

---

#### 4. **CLASSES MINORITÁRIAS AINDA FRACAS** 🎯

**Classe 0 (XGBoost R20)**:
- Recall: 55.20% (meta: 70%) ❌ -15pp
- **Precision: 18.76%** ❌❌ (crítico!)
- F1: 0.28 (inadequado)

**Classe 2 (XGBoost R20)**:
- Recall: 56.13% (meta: 70%) ❌ -14pp
- **Precision: 31.48%** ❌ (muito baixa)
- F1: 0.40 (inadequado)

**Classe 1 (majoritária)**:
- Recall: 78.51% ✅ (ótimo)
- Precision: 93.82% ✅ (excelente)
- F1: 0.85 ✅

**Análise**:
- Class weights ajudaram recall (~55-56% vs anterior ~32-38%)
- **MAS precisão muito baixa** (18% e 31%) → modelo predizendo classe 1 excessivamente

**Causa**: Class weights desbalanceados demais ou LR muito alto causando oscilações.

**Soluções**:
- ✅ Reduzir `max_class_weight`: 10.0 → **5.0** (menos agressivo)
- ✅ Implementar **Focal Loss** (TODO 3.2) - penaliza mais erros em classes difíceis
- ✅ Calibração de thresholds (TODO 3.3) - ajustar threshold de decisão por classe

---

#### 5. **OUT OF MEMORY COM 10 CLIENTES** 💾

**Evidência**: CatBoost Bagging não completou (interrompido)

**Configuração problemática**:
```python
num_clients: 10
num_local_boost_round: 50
depth: 6
```

**Solução**: Reduzir clientes para **7** (já coberto em item 2)

---

## 🎯 ESTRATÉGIAS DE MELHORIA - FASE 3

### Estratégia 1: **EARLY STOPPING LOCAL** (TODO 3.1) ✅

**Problema**: Overfitting nas rodadas finais (degradação R18-R20)

**Implementação**:
```python
# XGBoost, LightGBM, CatBoost clients
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_local_round,
    evals=[(dvalid, 'valid')],
    early_stopping_rounds=5,  # NOVO
    verbose_eval=False
)

# Após treino, logar número real de árvores usadas
actual_rounds = bst.best_iteration + 1
log(INFO, f"[Early Stopping] Usou {actual_rounds}/{num_local_round} rodadas")
```

**Impacto Esperado**:
- ✅ Evita overfitting local
- ✅ Reduz tempo de treino em ~20-30%
- ✅ Melhora generalização (+1-2% Balanced Acc)

---

### Estratégia 2: **AJUSTE DE HIPERPARÂMETROS** ✅

#### Configuração Otimizada (FASE 3):

**main.py**:
```python
DEFAULT_CONFIG = {
    "num_clients": 7,              # 10 → 7 (evitar OOM, mais dados/cliente)
    "vehicles_per_client": 20,     # 15 → 20 (total: 140 veículos)
    "num_server_rounds": 25,       # 20 → 25 (mais convergência)
    "num_local_boost_round": 50,   # mantém
    "seed": 42,
    "use_all_data": False,
    "balance_strategy": "weights",
    "stratified": True,

    "use_class_weights": True,
    "max_class_weight": 5.0,       # 10.0 → 5.0 (menos agressivo)
    "use_stable_params": True,
    "diversity_aggregation": True,
    "diversity_alpha": 0.8,
    "penalize_mono_class": True,
    "use_curriculum": True,
    "curriculum_warmup": 5,
    "use_entropy_cycling": True,
}
```

**fl_advanced.py**:
```python
# XGBoost (mantém)
if algorithm == 'xgboost':
    return {
        'max_depth': 6,
        'eta': 0.01,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'lambda': 1.5,
        'alpha': 0.5,
    }

# LightGBM (mantém)
elif algorithm == 'lightgbm':
    return {
        'max_depth': 6,
        'learning_rate': 0.01,
        'min_child_samples': 10,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 1.5,
        'min_split_gain': 0.01,
    }

# CatBoost (OTIMIZADO)
elif algorithm == 'catboost':
    return {
        'depth': 6,
        'learning_rate': 0.02,         # 0.03 → 0.02 (MUDANÇA)
        # 'auto_class_weights': REMOVIDO (conflito)
        'min_data_in_leaf': 10,
        'bootstrap_type': 'Bernoulli',
        'border_count': 254,           # 128 → 254 (MUDANÇA)
        'subsample': 0.8,
        'rsm': 0.8,
        'l2_leaf_reg': 3.0,
        'random_strength': 0.5,
    }
```

---

### Estratégia 3: **SIMPLIFICAÇÃO DO CÓDIGO** 🧹

#### Problemas Identificados no Código Atual:

1. **Complexidade desnecessária em `data_processing.py`**:
   - ❌ Suporta 6 estratégias de balanceamento (SMOTE, oversampling, etc)
   - ❌ Maioria nunca usada (apenas `weights` é efetivo)
   - ✅ **Remover** estratégias SMOTE/oversampling (incompatíveis com particionamento por veículo)

2. **Lógica duplicada de class weights**:
   - ❌ `ClassBalancingHelper` calcula weights
   - ❌ `data_processing.py` também calcula weights
   - ❌ CatBoost tem `auto_class_weights` (conflito)
   - ✅ **Unificar** em um único local (`ClassBalancingHelper`)

3. **Callbacks verbosos demais**:
   - ❌ `VerboseCallback` imprime cada época de treinamento
   - ❌ Polui logs (80 linhas por cliente por round)
   - ✅ **Simplificar** para apenas progresso resumido

4. **Estratégias avançadas não usadas**:
   - ❌ `CurriculumLearning` (implementado mas impacto mínimo)
   - ❌ `EntropyBasedCycling` (não usado em produção)
   - ✅ **Remover** ou marcar como experimentais

#### Simplificações Propostas:

**1. Remover estratégias de balanceamento não-usadas**:
```python
# ANTES: 6 estratégias (oversample, smote, undersample, combined, smoteenn, weights)
# DEPOIS: Apenas 'weights' (mais efetivo para tree-based models)

def _balance_classes(self, X, y):
    if self.balance_strategy != 'weights':
        log(WARNING, "Apenas 'weights' suportado para tree-based models em FL")
        return X, y

    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    self.class_weights = dict(zip(classes, class_weights))
    return X, y
```

**2. Unificar class weights em um único helper**:
```python
# Remover cálculo duplicado em data_processing.py
# Usar apenas ClassBalancingHelper nos clients
```

**3. Simplificar callbacks de treino**:
```python
# ANTES: VerboseCallback imprime cada época
# DEPOIS: Callback silencioso, log apenas final

class SilentProgressCallback(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        # Apenas conta iterações, sem print
        return False
```

---

### Estratégia 4: **REVISÃO DO PIPELINE DE DADOS** 🔍

#### Problemas Identificados:

1. **Normalização antes de particionamento**:
   ```python
   # ATUAL: Normaliza TODOS os dados antes de particionar
   self.scaler.fit(X_train_all)  # Usa informação de TODOS os clientes
   X_train_all = self.scaler.transform(X_train_all)
   ```

   **Problema**: Em FL real, cada cliente deveria normalizar seus próprios dados (não ter acesso a estatísticas globais)

   **Solução**: Manter normalização global (mais simples e efetivo para benchmarking)

2. **Particionamento estratificado por veículo**:
   ```python
   # ATUAL: Seleciona veículos baseado em classe majoritária
   majority_class = unique_cls[np.argmax(counts)]
   vehicle_class_dist[veh_id] = majority_class
   ```

   **Problema**: Veículos podem ter múltiplas classes, usar apenas majoritária perde informação

   **Solução**: ✅ Melhorar para considerar distribuição completa de classes por veículo

3. **Validação local vs global**:
   ```python
   # ATUAL: Cada cliente faz split 80/20 dos próprios dados
   train_test_split(X_part, y_part, test_size=0.2)
   ```

   **Problema**: Com 4-5k amostras por cliente, validação local pode ser desbalanceada

   **Solução**: ✅ Usar validação centralizada (`centralised_eval_client=True`)

---

### Estratégia 5: **IMPLEMENTAR FOCAL LOSS** (TODO 3.2) 🎯

**Problema**: Class weights melhoram recall mas prejudicam precision (18% e 31%)

**Focal Loss**: Reduz peso de exemplos "fáceis" (bem classificados) e foca em "difíceis"

**Fórmula**:
```
FL(p_t) = -(1 - p_t)^γ * log(p_t)

Onde:
- p_t = probabilidade da classe verdadeira
- γ = foco em exemplos difíceis (padrão: 2)
- α = balanceamento de classes (padrão: 0.25)
```

**Implementação**:
```python
# common/focal_loss.py
def focal_loss_xgb(y_pred, dtrain, gamma=2.0, alpha=0.25):
    """Focal Loss para XGBoost (gradient + hessian)"""
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid

    # Gradient
    grad = alpha * (p - y_true) * ((1 - p) ** gamma) * (gamma * p * np.log(p) + p - 1)

    # Hessian
    hess = alpha * ((1 - p) ** gamma) * (
        gamma * (1 - 2*p) * np.log(p) +
        (1 + gamma * p * (1 - p))
    )

    return grad, hess

# Uso no client:
params['objective'] = focal_loss_xgb
```

**Impacto Esperado**: +3-5% precision classes minoritárias

---

### Estratégia 6: **CALIBRAÇÃO DE THRESHOLDS** (TODO 3.3) 📊

**Problema**: Modelo usa threshold padrão (0.5 para binário, max prob para multi-classe)

**Solução**: Buscar threshold ótimo que maximize F1 para cada classe

**Implementação**:
```python
# common/threshold_optimization.py
from sklearn.metrics import f1_score

def optimize_multiclass_thresholds(y_true, y_pred_proba, num_classes=3):
    """
    Otimiza thresholds para maximizar F1-Score macro em classificação multi-classe

    Returns:
        thresholds: dict {class_id: threshold_value}
    """
    best_thresholds = {}

    for cls in range(num_classes):
        # Para cada classe, buscar melhor threshold
        best_f1 = 0
        best_thresh = 0.5

        for threshold in np.arange(0.1, 0.9, 0.05):
            # Predizer classe se prob > threshold
            y_pred_cls = (y_pred_proba[:, cls] > threshold).astype(int)

            # Calcular F1 binário (one-vs-rest)
            y_true_binary = (y_true == cls).astype(int)
            f1 = f1_score(y_true_binary, y_pred_cls)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = threshold

        best_thresholds[cls] = best_thresh
        log(INFO, f"Classe {cls}: threshold={best_thresh:.2f}, F1={best_f1:.4f}")

    return best_thresholds

# Aplicar após agregação global:
y_pred_proba = model.predict(X_test)
optimized_thresholds = optimize_multiclass_thresholds(y_test, y_pred_proba)

# Predizer com thresholds otimizados
y_pred = apply_optimized_thresholds(y_pred_proba, optimized_thresholds)
```

**Impacto Esperado**: +2-4% F1-Score

---

## 📋 PLANO DE IMPLEMENTAÇÃO FASE 3

### Prioridade ALTA (Implementar Agora):

1. ✅ **Ajustar hiperparâmetros** (5 min)
   - `num_clients: 10 → 7`
   - `vehicles_per_client: 15 → 20`
   - `num_server_rounds: 20 → 25`
   - `max_class_weight: 10.0 → 5.0`
   - CatBoost: `learning_rate: 0.03 → 0.02`, `border_count: 128 → 254`

2. ✅ **Implementar early stopping** (15 min)
   - Adicionar `early_stopping_rounds=5` em todos os clients
   - Logar número real de iterações usadas

3. ✅ **Simplificar data_processing.py** (10 min)
   - Remover estratégias SMOTE/oversampling
   - Manter apenas `balance_strategy='weights'`

### Prioridade MÉDIA (Implementar se tempo permitir):

4. ⏳ **Implementar focal loss** (30 min)
   - Criar `common/focal_loss.py`
   - Integrar nos clients

5. ⏳ **Calibração de thresholds** (20 min)
   - Criar `common/threshold_optimization.py`
   - Aplicar no servidor após agregação

### Prioridade BAIXA (Opcional):

6. ⏳ **Simplificar callbacks** (10 min)
7. ⏳ **Remover código experimental** (10 min)

---

## 🎯 MÉTRICAS ALVO - FASE 3

| Métrica | Meta FASE 2 | Real FASE 2 | **Meta FASE 3** | Melhoria |
|---------|-------------|-------------|-----------------|----------|
| **Balanced Accuracy** | > 72% | 63.28% | **> 68%** | +5pp |
| **F1-Score Macro** | > 0.62 | 0.5127 | **> 0.58** | +13% |
| **Recall Classe 0** | > 70% | 55.20% | **> 62%** | +7pp |
| **Recall Classe 2** | > 70% | 56.13% | **> 62%** | +6pp |
| **Precision Classe 0** | > 50% | 18.76% | **> 30%** | +11pp |
| **Precision Classe 2** | > 50% | 31.48% | **> 42%** | +11pp |

**Meta realista**: Não atingir 72% Balanced Acc, mas melhorar significativamente precision das classes minoritárias.

---

## ✅ VALIDAÇÃO DE SUCESSO FASE 3

### Checklist Mínimo:
- [ ] Balanced Accuracy > 68% (XGBoost)
- [ ] F1-Score Macro > 0.58 (XGBoost)
- [ ] Recall classes 0 e 2 > 62%
- [ ] **Precision classes 0 e 2 > 30% e 42%** (CRÍTICO)
- [ ] Early stopping funcionando (< 50 iterações realmente usadas)
- [ ] CatBoost ≈ LightGBM (diferença < 2% F1)
- [ ] Sem degradação R18-R20 (diferença < 1% do melhor round)
- [ ] Tempo total < 30min para all+both

### Checklist Avançado:
- [ ] Focal loss reduz false positives em classe 1
- [ ] Threshold optimization melhora F1 em +2-4%
- [ ] Code simplificado e mais legível
- [ ] Logs limpos e informativos

---

**FIM DA ANÁLISE - PROSSEGUIR PARA IMPLEMENTAÇÃO**
