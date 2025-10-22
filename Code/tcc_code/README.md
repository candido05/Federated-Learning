# Federated Learning - Tree-Based Models

Implementação modular e refatorada de Federated Learning com modelos baseados em árvore (XGBoost, LightGBM, CatBoost) usando o framework Flower.

## 📁 Estrutura do Projeto

```
tcc_code/
├── common/                    # 🔧 Módulos compartilhados
│   ├── __init__.py
│   ├── data_processing.py    # Processamento e particionamento do dataset HIGGS
│   └── metrics_logger.py     # Cálculo de métricas (AUC, F1, etc.) e logging
│
├── algorithms/                # 🤖 Implementações de FL para cada algoritmo
│   ├── __init__.py
│   └── xgboost_fl.py         # Cliente, servidor e execução para XGBoost
│
├── archive/                   # 📦 Códigos funcionais originais (PRESERVADOS)
│   ├── xgboost.py            # Código original XGBoost (funcional em Colab)
│   ├── ligthGBM.py           # Código original LightGBM (funcional em Colab)
│   └── catbbost.py           # Código original CatBoost (funcional em Colab)
│
├── run_experiments.py         # ▶️ Script principal de execução
├── requirements.txt           # 📋 Dependências do projeto
└── README.md                  # 📖 Este arquivo
```

## 🚀 Instalação

### 1. Criar ambiente virtual

```bash
cd Code/tcc_code
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## 💻 Uso

### ⚡ Execução Rápida - Scripts All-in-One

**Para executar todos os experimentos de um algoritmo (Cyclic + Bagging):**

```bash
# CatBoost - Todas as estratégias
PYTHONPATH=. python run_catboost_all.py

# XGBoost - Todas as estratégias
PYTHONPATH=. python run_xgboost_all.py

# LightGBM - Todas as estratégias
PYTHONPATH=. python run_lightgbm_all.py
```

**Para executar TODOS os 6 experimentos (XGBoost, LightGBM, CatBoost × Cyclic, Bagging):**

```bash
PYTHONPATH=. python run_all_experiments.py
```

### Execução Individual via CLI

```bash
# XGBoost com estratégia Cyclic
PYTHONPATH=. python run_experiments.py --algorithm xgboost --strategy cyclic

# XGBoost com estratégia Bagging
PYTHONPATH=. python run_experiments.py --algorithm xgboost --strategy bagging

# Executar ambas estratégias
PYTHONPATH=. python run_experiments.py --algorithm xgboost --strategy both
```

### Parâmetros Disponíveis

```bash
python run_experiments.py --help

Opções:
  --algorithm {xgboost,lightgbm,catboost,all}
                        Algoritmo a executar (padrão: xgboost)
  --strategy {cyclic,bagging,both}
                        Estratégia de agregação (padrão: cyclic)
  --num-clients NUM     Número de clientes (padrão: 6)
  --num-rounds NUM      Número de rodadas do servidor (padrão: 6)
  --local-rounds NUM    Rodadas locais de boosting (padrão: 20)
  --samples NUM         Amostras por cliente (padrão: 8000)
  --seed NUM            Random seed (padrão: 42)
```

### Exemplos Avançados

```bash
# Experimento customizado - mais clientes e rodadas
python run_experiments.py \
    --algorithm xgboost \
    --strategy both \
    --num-clients 10 \
    --num-rounds 10 \
    --local-rounds 30 \
    --samples 5000

# Teste rápido com poucos dados
python run_experiments.py \
    --num-clients 3 \
    --num-rounds 3 \
    --samples 2000
```

## 📊 Métricas Coletadas

Para cada rodada, são calculadas automaticamente:
- ✅ **Acurácia** (Accuracy)
- ✅ **Precisão** (Precision)
- ✅ **Revocação** (Recall)
- ✅ **F1-Score**
- ✅ **AUC-ROC**
- ✅ **Especificidade**
- ✅ **Matriz de Confusão** (TN, FP, FN, TP)

## 📈 Outputs

### Sistema de Logging Estruturado

Todos os experimentos são automaticamente salvos na pasta `logs/` organizada por **algoritmo → data/hora → estratégia**:

```
logs/
├── xgboost/
│   ├── 20251021_143052_cyclic/       # Pasta única por execução
│   │   ├── execution_log.txt         # Log completo com todas as métricas
│   │   ├── metrics.json              # Dados estruturados em JSON
│   │   └── README.md                 # Resumo do experimento
│   └── 20251021_144120_bagging/
│       ├── execution_log.txt
│       ├── metrics.json
│       └── README.md
├── lightgbm/
│   └── ...
└── catboost/
    └── ...
```

**Vantagens dessa estrutura:**
- ✅ Cada execução tem sua própria pasta com timestamp
- ✅ Fácil identificar quando o experimento foi executado
- ✅ Nunca sobrescreve resultados anteriores
- ✅ README.md em cada pasta para navegação rápida

### Arquivo de Log de Texto (.txt)

Contém output formatado de cada experimento:

```
================================================================================
INICIANDO EXPERIMENTO: XGBOOST - CYCLIC
================================================================================
Configuração:
  - Algoritmo: xgboost
  - Estratégia: cyclic
  - Número de clientes: 6
  - Rodadas globais: 3
  - Rodadas locais: 20
  - Amostras por cliente: 2000
================================================================================

[SERVER] Round 1 Métricas de Performance:
  Acurácia:    0.6954
  Precisão:    0.6987
  Revocação:   0.6880
  F1-Score:    0.6933
  AUC:         0.7651
  Especific.:  0.7028
  Matriz de Confusão:
    TN:  281 | FP:  119
    FN:  125 | TP:  275

...

================================================================================
EXPERIMENTO CONCLUÍDO: XGBOOST - CYCLIC
Tempo total: 45.32 segundos
================================================================================
```

### Arquivo JSON (.json)

Contém dados estruturados completos:

```json
{
  "experiment_info": {
    "algorithm": "xgboost",
    "strategy": "cyclic",
    "num_clients": 6,
    "num_rounds": 3,
    "num_local_rounds": 20,
    "samples_per_client": 2000,
    "total_time_seconds": 45.32
  },
  "metrics_by_round": {
    "1": {
      "accuracy": 0.6954,
      "precision": 0.6987,
      "recall": 0.6880,
      "f1_score": 0.6933,
      "auc": 0.7651,
      "specificity": 0.7028,
      "confusion_matrix": {
        "tn": 281, "fp": 119,
        "fn": 125, "tp": 275
      }
    },
    ...
  },
  "detailed_logs": [...],
  "final_history": "History(...)"
}
```

### Console
Métricas são impressas em tempo real (e salvas no arquivo .txt):

```
[SERVER] Round 1 Métricas de Performance:
  Acurácia:    0.8542
  Precisão:    0.8331
  Revocação:   0.8765
  F1-Score:    0.8543
  AUC:         0.9102
  Especific.:  0.8319
  Matriz de Confusão:
    TN: 3421 | FP:  679
    FN:  498 | TP: 3402
```

## 🏗️ Arquitetura Modular

### 1. `common/data_processing.py`
- **DataProcessor**: Classe para carregar e particionar dataset HIGGS
- **replace_keys()**: Utilitário para converter configurações

### 2. `common/metrics_logger.py`
- **calculate_comprehensive_metrics()**: Calcula todas as métricas (AUC, F1, Precision, Recall, etc.)
- **print_metrics_summary()**: Imprime métricas formatadas no console
- **ExperimentLogger**: Gerencia logging completo de experimentos
  - Cria diretórios automaticamente em `logs/{algorithm}/`
  - Salva logs em tempo real (.txt) e dados estruturados (.json)
  - Rastreia tempo de execução e histórico completo
  - Métodos principais:
    - `start_experiment()`: Inicializa experimento e logging
    - `log_round_metrics()`: Registra métricas de cada rodada
    - `log_aggregated_metrics()`: Registra métricas agregadas
    - `end_experiment()`: Finaliza e salva todos os dados
- **evaluate_metrics_aggregation()**: Agrega métricas de múltiplos clientes

### 3. `algorithms/xgboost_fl.py`
- **XGBoostClient**: Cliente FL para XGBoost
- **run_xgboost_experiment()**: Função principal para executar experimento
- Suporte automático para GPU/CPU
- Estratégias Cyclic e Bagging

## 🎯 Estratégias de Agregação

### Cyclic (Cíclica)
- ⚡ Treina **um cliente por rodada**, sequencialmente
- 🔄 Modelo passa de cliente em cliente
- ✅ Melhor para convergência gradual
- 💾 Menor uso de memória

### Bagging
- 🚀 **Todos os clientes treinam em paralelo**
- 🔀 Modelos são agregados no servidor
- ⚡ Mais rápido (processamento paralelo)
- 💻 Requer mais recursos computacionais

## 📦 Códigos Originais (Archive)

Os códigos funcionais originais estão **preservados** em `archive/`:
- `xgboost.py` - Código original do XGBoost (testado e funcional no Colab)
- `ligthGBM.py` - Código original do LightGBM (testado e funcional no Colab)
- `catbbost.py` - Código original do CatBoost (testado e funcional no Colab)

**⚠️ Estes arquivos são referência e NÃO devem ser modificados.**

## ✅ Status de Implementação

- ✅ **XGBoost**: Totalmente funcional e modularizado para VSCode
- ⏳ **LightGBM**: Em desenvolvimento (use `archive/ligthGBM.py` temporariamente)
- ⏳ **CatBoost**: Em desenvolvimento (use `archive/catbbost.py` temporariamente)

## 🐛 Troubleshooting

### Erro: "Module not found"
Certifique-se de executar a partir do diretório correto:
```bash
cd Code/tcc_code
python run_experiments.py ...
```

### Erro: "CUDA out of memory"
Reduza o número de clientes ou amostras:
```bash
python run_experiments.py --num-clients 4 --samples 5000
```

### Dataset HIGGS não baixa
- O dataset é baixado automaticamente do HuggingFace
- Certifique-se de ter conexão com internet
- Pode demorar na primeira execução (~1GB)

### Import errors
Reinstale as dependências:
```bash
pip install --upgrade -r requirements.txt
```

## 🔬 Para Desenvolvedores

### Estrutura de um Módulo de Algoritmo

Cada algoritmo em `algorithms/` deve implementar:

1. **Cliente FL** (classe que herda de `flwr.client.Client`)
   - `fit()`: Treino local com modelo
   - `evaluate()`: Avaliação local

2. **Funções Factory**
   - `create_client_fn()`: Cria função de cliente
   - `create_server_fn()`: Cria função de servidor
   - `get_evaluate_fn()`: Cria função de avaliação centralizada

3. **Função Principal**
   - `run_{algorithm}_experiment()`: Orquestra experimento completo

### Exemplo: Template para Novo Algoritmo

```python
# algorithms/new_algorithm_fl.py

from ..common import DataProcessor, calculate_comprehensive_metrics
from flwr.client import Client, ClientApp
from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes

class NewAlgorithmClient(Client):
    def fit(self, ins: FitIns) -> FitRes:
        # Implementar treino local
        pass

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Implementar avaliação local
        pass

def run_new_algorithm_experiment(data_processor, num_clients, ...):
    # Implementar lógica completa do experimento
    pass
```

## 📚 Referências

- **Flower Framework**: https://flower.ai/
- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **CatBoost**: https://catboost.ai/
- **Dataset HIGGS**: https://huggingface.co/datasets/jxie/higgs

## 📝 Licença

Este código faz parte de um projeto de TCC (Trabalho de Conclusão de Curso) sobre **"Optimization of Federated Learning Models with SDN (Software-Defined Networking)"**.

---

**Desenvolvido com 🤖 para TCC - Federated Learning com SDN**
