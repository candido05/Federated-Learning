# FASE 2 - OTIMIZAÇÕES IMPORTANTES ✅

## Status: CÓDIGO MODIFICADO - PRONTO PARA EXECUÇÃO

**Data de Implementação**: 2025-12-22
**Tempo Estimado**: 1-2 horas
**Impacto Esperado**: +5-10% nas métricas

---

## ✅ TODO 2.1: AUTO_CLASS_WEIGHTS NO CATBOOST - COMPLETO

**Arquivo modificado**: `Code/tcc_code/common/fl_advanced.py` (linha 430)

```python
# ANTES:
elif algorithm == 'catboost':
    return {
        'depth': 4,
        'learning_rate': 0.03,
        'min_data_in_leaf': 20,
        ...

# DEPOIS:
elif algorithm == 'catboost':
    return {
        'depth': 6,
        'learning_rate': 0.03,
        'auto_class_weights': 'Balanced',  # <-- NOVO
        'min_data_in_leaf': 10,
        ...
```

**Justificativa**: CatBoost tem suporte nativo para class weights automáticos, calculando pesos balanceados sem necessidade de implementação manual.

**Impacto Esperado**: CatBoost +1-2% (alcançar LightGBM)

**Status**: ✅ Implementado

---

## ✅ TODO 2.2: AUMENTAR PROFUNDIDADE DAS ÁRVORES - COMPLETO

**Arquivo modificado**: `Code/tcc_code/common/fl_advanced.py`

### XGBoost (linha 403):
```python
# ANTES:
'max_depth': 4,

# DEPOIS:
'max_depth': 6,
```

### LightGBM (linha 415):
```python
# ANTES:
'max_depth': 4,

# DEPOIS:
'max_depth': 6,
```

### CatBoost (linha 428):
```python
# ANTES:
'depth': 4,

# DEPOIS:
'depth': 6,
```

**Justificativa**: Profundidade 4 é muito rasa para 22 features. Depth 6 é o sweet spot que não overfita nem underfita.

**Impacto Esperado**: +1-3% F1-Score

**Status**: ✅ Implementado

---

## ✅ TODO 2.3: AUMENTAR BORDER_COUNT NO CATBOOST - COMPLETO

**Arquivo modificado**: `Code/tcc_code/common/fl_advanced.py` (linha 433)

```python
# ANTES:
# Não existia border_count

# DEPOIS:
'border_count': 128,
```

**Justificativa**: CatBoost usa histogramas para splits. 32 bins (default) = baixa granularidade. 128 bins = alta precisão recomendada para classes desbalanceadas.

**Impacto Esperado**: CatBoost +0.5-1% precisão nos splits

**Status**: ✅ Implementado

---

## ✅ TODO 2.4: REDUZIR MIN_DATA_IN_LEAF - COMPLETO

**Arquivo modificado**: `Code/tcc_code/common/fl_advanced.py`

### LightGBM (linha 417):
```python
# ANTES:
'min_child_samples': 20,

# DEPOIS:
'min_child_samples': 10,
```

### CatBoost (linha 431):
```python
# ANTES:
'min_data_in_leaf': 20,

# DEPOIS:
'min_data_in_leaf': 10,
```

**Justificativa**: Classes minoritárias têm poucas amostras. min_data_in_leaf=20 impedia splits finos. 10 permite mais granularidade sem overfitting.

**Impacto Esperado**: +0.5-2% Recall classes minoritárias

**Status**: ✅ Implementado

---

## ✅ TODO 2.5: AUMENTAR NÚMERO DE CLIENTES - COMPLETO

**Arquivo modificado**: `Code/tcc_code/main.py` (linhas 38, 43)

```python
# ANTES:
"num_clients": 3,
"vehicles_per_client": 40,

# DEPOIS:
"num_clients": 10,
"vehicles_per_client": 15,
```

**Justificativa**:
- 3 clientes com 102k amostras cada = quase centralizado
- FL real precisa de 10-100 clientes com dados heterogêneos
- Heterogeneidade real só aparece com mais clientes
- Total de veículos: 10 × 15 = 150 veículos (antes: 3 × 40 = 120)

**Impacto Esperado**: +1-3% generalização, FL mais realista

**Status**: ✅ Implementado

---

## ✅ TODO 2.6: ATIVAR DIVERSITY AGGREGATION - COMPLETO

**Arquivo modificado**: `Code/tcc_code/main.py` (linha 51)

```python
# ANTES:
"diversity_alpha": 0.5,

# DEPOIS:
"diversity_alpha": 0.8,
```

**Justificativa**:
- Alpha=0.5 = pouco peso na diversidade de classes
- Alpha=0.8 = prioriza clientes com classes minoritárias
- Já implementado, só precisava de ajuste fino

**Impacto Esperado**: +0.5-1.5% em agregação ponderada

**Status**: ✅ Implementado

---

## 📊 Resumo das Mudanças - FASE 2

### Arquivo: `common/fl_advanced.py`

```python
# XGBoost
'max_depth': 4 → 6

# LightGBM
'max_depth': 4 → 6
'min_child_samples': 20 → 10

# CatBoost
'depth': 4 → 6
'auto_class_weights': 'Balanced'  # NOVO
'min_data_in_leaf': 20 → 10
'border_count': 128  # NOVO
```

### Arquivo: `main.py`

```python
"num_clients": 3 → 10
"vehicles_per_client": 40 → 15
"diversity_alpha": 0.5 → 0.8
```

---

## 🔧 Comandos para Executar

### 1. Limpar Cache Python (CRÍTICO!)

**WSL/Linux**:
```bash
cd /mnt/c/Users/candi/OneDrive/Desktop/Federated-Learning/Code/tcc_code
./clean_cache.sh
```

**Windows PowerShell**:
```powershell
cd C:\Users\candi\OneDrive\Desktop\Federated-Learning\Code\tcc_code
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter *.pyc | Remove-Item -Force
```

### 2. Executar Experimentos

#### Opção 1: Testar apenas CatBoost (Rápido - 5-10 min)
```bash
python main.py --algorithm catboost --strategy both
```

#### Opção 2: Executar todos os algoritmos (Completo - 30-50 min)
```bash
python main.py --algorithm all --strategy both
```

---

## 📈 Métricas Alvo - FASE 2

### Após FASE 1 (Baseline):
```
XGBoost Cyclic Round 10:
- Balanced Accuracy: 66.78%
- F1-Score Macro:    0.4979
- Recall Classe 2:   57.22%
```

### Após FASE 2 (Esperado):
```
XGBoost Cyclic Round 20:
- Balanced Accuracy: > 72%     (+5pp)
- F1-Score Macro:    > 0.62    (+24%)
- Recall Classe 2:   > 70%     (+13pp)

CatBoost Cyclic Round 20:
- Balanced Accuracy: ≈ LightGBM (diferença < 0.5%)
- F1-Score Macro:    > 0.62
- Recall Classe 2:   > 70%
```

### Comparação CatBoost vs LightGBM (OBJETIVO):
```
Antes (FASE 1):
- LightGBM:  Balanced Acc 67.25%, F1 0.4984
- CatBoost:  Balanced Acc 66.23%, F1 0.4831  (PIOR -1%)

Depois (FASE 2):
- LightGBM:  Balanced Acc 72%,    F1 0.62
- CatBoost:  Balanced Acc 72%,    F1 0.62    (IGUAL!)
```

---

## 🎯 Validação de Sucesso

### Checklist Mínimo:
- [ ] CatBoost Balanced Acc ≈ LightGBM (diferença < 0.5%)
- [ ] F1-Score Macro > 0.62 (todos os algoritmos)
- [ ] Recall Classe 2 > 70% (todos os algoritmos)
- [ ] 10 clientes executando (não 3)
- [ ] 20 rodadas completadas
- [ ] Tempo total < 1h para all+both

### Checklist Detalhado:
- [ ] ✅ **auto_class_weights**: CatBoost usando pesos balanceados
- [ ] ✅ **max_depth=6**: Todos os algoritmos com árvores mais profundas
- [ ] ✅ **border_count=128**: CatBoost com granularidade alta
- [ ] ✅ **min_data_in_leaf=10**: LightGBM e CatBoost permitindo splits finos
- [ ] ✅ **10 clientes**: FL mais realista e heterogêneo
- [ ] ✅ **diversity_alpha=0.8**: Priorizando clientes com classes minoritárias

---

## 📊 Análise Esperada nos Logs

### Round 1 (Início):
```
[SERVER] Round 1 Métricas de Performance:
  Acurácia:           ~0.86-0.89  (alta por oversample inicial)
  Balanced Accuracy:  ~0.62-0.65  [ESPERADO]
  F1-Score Macro:     ~0.42-0.45  [ESPERADO]
  Recall Classe 2:    ~0.55-0.60  [ESPERADO]
```

### Round 10 (Meio):
```
[SERVER] Round 10 Métricas de Performance:
  Acurácia:           ~0.68-0.72
  Balanced Accuracy:  ~0.68-0.71  [MELHORIA!]
  F1-Score Macro:     ~0.58-0.61  [MELHORIA!]
  Recall Classe 2:    ~0.65-0.68  [MELHORIA!]
```

### Round 20 (Final):
```
[SERVER] Round 20 Métricas de Performance:
  Acurácia:           ~0.70-0.74
  Balanced Accuracy:  > 0.72      [ALVO ATINGIDO!]
  F1-Score Macro:     > 0.62      [ALVO ATINGIDO!]
  Recall Classe 2:    > 0.70      [ALVO ATINGIDO!]
```

---

## 🔍 Troubleshooting

### Problema 1: Métricas não melhoraram
**Sintoma**: F1-Score Macro ainda < 0.60

**Diagnóstico**:
```bash
# Verificar se cache foi limpo
ls -la common/__pycache__/

# Verificar configuração aplicada
grep "num_clients" logs/xgboost/*/execution_log.txt | tail -1
grep "auto_class_weights" logs/catboost/*/execution_log.txt | tail -1
```

**Solução**:
1. Limpar cache novamente
2. Confirmar que `fl_advanced.py` tem as mudanças
3. Re-executar experimentos

### Problema 2: CatBoost ainda pior que LightGBM
**Sintoma**: CatBoost Balanced Acc < LightGBM -1%

**Diagnóstico**:
```bash
# Verificar auto_class_weights sendo usado
grep -A 5 "CatBoost.*params" logs/catboost/*/execution_log.txt
```

**Solução**:
- Verificar que `auto_class_weights: 'Balanced'` está nos params
- Se não estiver, cache não foi limpo

### Problema 3: Out of Memory
**Sintoma**: Erro de memória com 10 clientes

**Solução**:
```python
# Reduzir temporariamente
"num_clients": 7,
"vehicles_per_client": 20,
```

### Problema 4: Convergência muito lenta
**Sintoma**: Round 20 ainda melhorando significativamente

**Solução**:
- Aumentar `num_server_rounds` para 30
- Considerar implementar early stopping (FASE 3)

---

## 📝 Comparação: FASE 1 vs FASE 2

| Parâmetro              | FASE 1      | FASE 2      | Mudança    |
|------------------------|-------------|-------------|------------|
| **num_clients**        | 3           | 10          | +233%      |
| **vehicles_per_client**| 40          | 15          | -62%       |
| **max_depth (XGB)**    | 4           | 6           | +50%       |
| **max_depth (LGBM)**   | 4           | 6           | +50%       |
| **depth (CatBoost)**   | 4           | 6           | +50%       |
| **min_child_samples**  | 20          | 10          | -50%       |
| **min_data_in_leaf**   | 20          | 10          | -50%       |
| **border_count**       | ausente     | 128         | NOVO       |
| **auto_class_weights** | ausente     | Balanced    | NOVO       |
| **diversity_alpha**    | 0.5         | 0.8         | +60%       |

---

## 🎯 Próximos Passos

### Se FASE 2 for bem-sucedida (F1 > 0.62, Balanced Acc > 72%):
✅ Marcar TODOs 2.1-2.7 como completos
📊 Gerar plots comparativos
📈 Documentar melhorias obtidas
🚀 **OPCIONAL**: Prosseguir para FASE 3 (early stopping, focal loss, etc.)

### Se FASE 2 não atingir métricas (F1 < 0.60):
⚠️ **NÃO prosseguir para FASE 3**
🔍 Revisar implementação de class_weights
🐛 Debugar aggregation strategies
📧 Relatar problema com logs completos

---

## 📊 Comparação Esperada: Execuções

### Execução ANTES (21/12 18:02 - FASE 1 parcial):
```
XGBoost Cyclic Round 10:
- Balanced Accuracy: 66.78%
- F1-Score Macro:    0.4979
- Recall Classe 0:   74.95%
- Recall Classe 1:   68.18%
- Recall Classe 2:   57.22%
- AUC:               0.8157
- Num Clients:       3
- Samples/Client:    102213
```

### Execução DEPOIS (FASE 2 completa - esperado):
```
XGBoost Cyclic Round 20:
- Balanced Accuracy: > 72%      (+5.2pp)
- F1-Score Macro:    > 0.62     (+24%)
- Recall Classe 0:   ~77%       (+2pp)
- Recall Classe 1:   ~72%       (+4pp)
- Recall Classe 2:   > 70%      (+13pp)
- AUC:               > 0.84      (+2%)
- Num Clients:       10         (realismo)
- Samples/Client:    ~11k        (heterogêneo)
```

---

## 💾 Backup e Versionamento

**IMPORTANTE**: Antes de executar, faça backup da configuração anterior!

```bash
# Backup dos logs antigos
cp -r logs/ logs_backup_fase1/

# Ou criar um git commit
git add main.py common/fl_advanced.py
git commit -m "feat: FASE 2 - Otimizações importantes (+10 clientes, depth 6, auto_class_weights)"
```

---

## 📞 Suporte

Se encontrar problemas:

1. **Salve os logs**: `logs/[algorithm]/[timestamp]_[strategy]/execution_log.txt`
2. **Compare com FASE 1**: Use tabelas acima
3. **Verifique configuração**: Confirme que todas as mudanças foram aplicadas
4. **Limpe cache**: Sempre limpe antes de executar

---

**Tempo Total de Implementação**: 15 minutos
**Tempo Estimado de Execução**: 30-50 minutos (all+both)
**Impacto Esperado**: +5-10% nas métricas principais
**Próxima Fase**: FASE 3 (opcional) ou FASE 4 (validação e documentação)

🚀 **BOA SORTE COM A EXECUÇÃO DA FASE 2!**
