# Bank Account Fraud Detection - Federated Learning

## Visão Geral

Este notebook implementa **Federated Learning (FL)** para detecção de fraude bancária usando o dataset BAF (Bank Account Fraud), mantendo 100% do preprocessamento e estrutura do notebook original `bank_account_fraud_sota_benchmark`.

## Características Principais

✅ **100% do preprocessamento original**
- Conversão de -1 para NaN
- Remoção de ruído (hard drop)
- Imputação com mediana
- Feature engineering avançado
- One-Hot Encoding

✅ **3 Modelos de Árvore Federalizados**
- XGBoost
- LightGBM
- CatBoost

✅ **2 Estratégias de FL**
- **Bagging (FedAvg)**: Todos os clientes treinam em paralelo, modelos são agregados
- **Cyclic**: Um cliente por round, em sequência cíclica

✅ **Particionamento Balanceado**
- 3 clientes com quantidades semelhantes de fraude/não-fraude
- Particionamento estratificado para manter distribuição de classes
- Mantém split temporal (Meses 0-5: Treino | Mês 6: Validação | Mês 7: Teste)

✅ **Código Simples e Elegante**
- Fácil de entender
- Sem complexidade desnecessária
- Comentários e documentação clara

## Estrutura do Notebook

### Seções 1-5: Preprocessamento Original
1. **Imports e Configuração**
2. **Carregamento e Pré-processamento**
3. **Feature Engineering**
4. **Validação Temporal**
5. **Métricas Customizadas** (TPR @ 5% FPR)

### Seções 6-10: Federated Learning
6. **Particionamento Federado** (3 clientes balanceados)
7. **FL - XGBoost** (Bagging + Cyclic)
8. **FL - LightGBM** (Bagging + Cyclic)
9. **FL - CatBoost** (Bagging + Cyclic)
10. **Visualizações e Comparação** (Plots de convergência e resultados)

## Requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost
pip install flwr[simulation]
```

## Como Executar

### 1. Abrir o Notebook

```bash
cd Code/novo_tcc_code
jupyter notebook bank_account_fraud_federated_learning.ipynb
```

### 2. Executar Células Sequencialmente

O notebook é projetado para ser executado de cima para baixo:

1. **Células 1-5**: Preprocessamento (idêntico ao original)
2. **Célula 6**: Particionamento federado
3. **Células 7-9**: Treinamento FL (XGBoost, LightGBM, CatBoost)
4. **Célula 10**: Visualizações e comparação

### 3. Customizar Configurações

Você pode ajustar os seguintes parâmetros:

```python
# Configuração FL
NUM_CLIENTS = 3           # Número de clientes
NUM_ROUNDS = 10           # Rounds de FL
NUM_LOCAL_ROUNDS = 20     # Boosting rounds por cliente
RANDOM_STATE = 42         # Seed para reprodutibilidade
```

## Métricas e Avaliação

### Métrica Principal: TPR @ 5% FPR

- **TPR (True Positive Rate / Recall)**: % de fraudes corretamente identificadas
- **FPR (False Positive Rate)**: % de clientes legítimos marcados como fraude
- **Meta do Benchmark**: TPR > 0.52 @ FPR = 5%

### Métricas Adicionais

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-Score

## Particionamento de Dados

### Estratégia

Cada um dos 3 clientes recebe:
- ~1/3 das fraudes
- ~1/3 dos não-fraudulentos

Isso garante que:
- ✅ Todos os clientes têm dados balanceados
- ✅ Distribuição de classes é semelhante entre clientes
- ✅ Mantém split temporal (treino/validação/teste)

### Visualização

O notebook gera gráficos mostrando a distribuição de classes por cliente:

```
Cliente 0: 33.3% dos dados | 0.9% fraudes
Cliente 1: 33.3% dos dados | 0.9% fraudes
Cliente 2: 33.4% dos dados | 0.9% fraudes
```

## Estratégias de FL

### 1. Bagging (FedAvg)

**Como funciona:**
1. Todos os 3 clientes treinam em paralelo
2. Cada cliente treina modelo local por 20 rounds
3. Servidor agrega modelos (média ponderada)
4. Modelo global é distribuído para próximo round

**Vantagens:**
- ✅ Treinamento paralelo (mais rápido)
- ✅ Utiliza todos os clientes a cada round
- ✅ Convergência mais rápida

### 2. Cyclic

**Como funciona:**
1. Um cliente por round (sequência: 0 → 1 → 2 → 0 → ...)
2. Cliente treina modelo local por 20 rounds
3. Modelo atualizado é passado para próximo cliente
4. Processo se repete ciclicamente

**Vantagens:**
- ✅ Menor sobrecarga de comunicação
- ✅ Modelo evolui sequencialmente
- ✅ Útil quando recursos são limitados

## Visualizações

O notebook gera os seguintes plots:

1. **Distribuição de Classes por Cliente**
   - Gráfico de barras mostrando fraudes vs não-fraudes

2. **Convergência de Métricas**
   - Loss por round (Bagging vs Cyclic)
   - AUC por round (se disponível)

3. **Comparação Final**
   - Barras comparando loss final de todos os experimentos
   - XGBoost (Bagging/Cyclic) vs LightGBM vs CatBoost

## Logs de Treinamento

Durante a execução, você verá logs detalhados:

```
[Cliente 0] Round 1 - Treinando...
[Cliente 0] Treino: AUC=0.8234 | TPR@5%FPR=0.5123
[Cliente 0] Validação: AUC=0.8156 | TPR@5%FPR=0.4987

[Cliente 1] Round 1 - Treinando...
[Cliente 1] Treino: AUC=0.8301 | TPR@5%FPR=0.5234
...
```

## Diferenças do Notebook Original

| Aspecto | Original | Federalizado |
|---------|----------|--------------|
| **Treinamento** | Centralizado | Distribuído (3 clientes) |
| **Dados** | Todos em um lugar | Particionados entre clientes |
| **Agregação** | Não aplicável | FedAvg / Cyclic |
| **Comunicação** | Não aplicável | Flower framework |
| **Preprocessamento** | ✅ Mantido 100% | ✅ Mantido 100% |
| **Feature Engineering** | ✅ Mantido 100% | ✅ Mantido 100% |
| **Métricas** | ✅ Mantidas | ✅ Mantidas + métricas locais |

## Próximos Passos

Após executar o notebook:

1. **Avaliar modelos agregados no teste** (Mês 7)
2. **Comparar com resultados centralizados** (notebook original)
3. **Análise de fairness** no contexto federado
4. **Experimentar com mais clientes** (5, 10, 20...)
5. **Testar diferentes particionamentos** (non-IID, heterogêneo)

## Referências

- **Paper BAF**: "Turning Lemons into Lemonade" (NeurIPS 2022)
- **Flower Framework**: https://flower.ai/docs/
- **Notebook Original**: `bank_account_fraud_sota_benchmark.ipynb`

## Autor

Criado para o TCC sobre "Optimization of Federated Learning Models with SDN"

## Licença

Este código é fornecido "as is" para fins educacionais e de pesquisa.
