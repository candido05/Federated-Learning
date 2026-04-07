# Federated Learning com Análise de Fairness

**Trabalho de Conclusão de Curso (TCC)**
**"Otimização de Modelos de Aprendizado Federado com SDN (Software-Defined Networking)"**

Bacharel em Ciência da Computação — Universidade Federal da Paraíba (UFPB)

**Autor**: Cândido Leandro de Queiroga Bisneto
**Orientador**: Prof. Fernando Menezes Matos

---

## Sobre o Projeto

Este repositório contém a implementação completa de experimentos de **Aprendizado Federado (FL)** com modelos baseados em árvores de decisão aplicados ao dataset **Bank Account Fraud (BAF)**. O foco principal é avaliar como diferentes estratégias de agregação impactam a **detecção de fraude** e a **equidade (fairness)** entre grupos demográficos.

Foram avaliados três algoritmos de boosting (**XGBoost**, **LightGBM**, **CatBoost**) com duas estratégias de agregação federada (**Bagging** e **Cyclic**), totalizando **6 configurações experimentais**, cada uma com **50 rodadas** de treinamento federado.

---

## Resultados

### Desempenho Final (Teste Standard — FPR ≤ 5%)

| Algoritmo | Estratégia | TPR | AUC | Bytes Transferidos | Duração |
|-----------|-----------|-----|-----|-------------------|---------|
| XGBoost | **Cycling** | **0.5574** | **0.8907** | 339 MB | 943s |
| XGBoost | Bagging | 0.4671 | 0.8548 | 1.013 GB | 1079s |
| LightGBM | **Cycling** | **0.5588** | **0.8897** | 367 MB | 764s |
| LightGBM | Bagging | 0.4580 | 0.8474 | 1.078 GB | 2574s |
| CatBoost | **Cycling** | **0.5518** | **0.8881** | 136 MB | 104s |
| CatBoost | Bagging | 0.4573 | 0.8355 | 407 MB | 248s |

### Com Pós-processamento de Fairness (FPR ≤ 5% por grupo)

| Algoritmo | Estratégia | TPR | AUC | Fairness Score |
|-----------|-----------|-----|-----|---------------|
| XGBoost | Cycling | 0.5308 | 0.8907 | **0.9987** |
| LightGBM | Cycling | 0.5350 | 0.8897 | 0.9865 |
| CatBoost | Cycling | 0.5329 | 0.8881 | 0.9717 |
| XGBoost | Bagging | 0.4545 | 0.8548 | 0.9884 |
| LightGBM | Bagging | 0.4447 | 0.8474 | 0.9867 |
| CatBoost | Bagging | 0.4489 | 0.8355 | 0.9916 |

**Conclusão principal**: A estratégia **Cyclic supera Bagging** em TPR e AUC em todos os algoritmos, com consumo de comunicação 3–10x menor. O pós-processamento de fairness eleva o Fairness Score para ~0.99 em todos os casos, com custo moderado em TPR (~2–3%).

---

## Dataset

**Bank Account Fraud (BAF)** — dataset de detecção de fraudes em abertura de contas bancárias.

| Propriedade | Valor |
|------------|-------|
| Total de amostras | 999.641 |
| Features | 99 |
| Taxa de fraude (treino) | ~1,03% |
| Partições (clientes FL) | 3 (IID) |
| Amostras por cliente | ~264.893 |

O dataset possui atributo sensível de grupo demográfico, permitindo análise de fairness via equalização da taxa de falsos positivos (FPR) por grupo.

---

## Estrutura do Repositório

```
Federated-Learning/
├── Code/
│   └── tcc_code/               # Implementação principal (modular)
│       ├── algorithms/         # Um módulo por algoritmo
│       │   ├── xgboost/        # client.py, server.py, runner.py, __init__.py
│       │   ├── lightgbm/       # client.py, server.py, runner.py, __init__.py
│       │   └── catboost/       # client.py, server.py, runner.py, __init__.py
│       ├── common/             # Utilitários compartilhados
│       │   ├── data_processing.py   # DataProcessor (carrega e particiona BAF)
│       │   ├── metrics.py           # Métricas, fairness, AUC
│       │   ├── logger.py            # ExperimentLogger (JSON + TXT)
│       │   └── utils.py             # Helpers gerais
│       ├── main.py             # Ponto de entrada (CLI)
│       └── requirements.txt
│
├── experiments/                # Resultados dos experimentos executados
│   └── experiments/
│       ├── 20260317_112359/    # Execução 1
│       ├── 20260323_203608/    # Execução 2
│       └── 20260331_135100/    # Execução final (50 rodadas, todos os algoritmos)
│           ├── summary.txt          # Relatório completo
│           ├── simulations_summary.csv
│           ├── {algorithm}_{strategy}_rounds.csv
│           └── experiment.json
│
├── doc/                        # Monografia em LaTeX
│   ├── main.tex
│   └── chapters/               # Capítulos 01–09
│
├── baf_data_&_code/            # Notebooks de análise exploratória do BAF
├── dataset_fl/                 # Dataset local de veículos (experimentos anteriores)
├── Artigos/                    # Referências bibliográficas
└── images/                     # Figuras e gráficos
```

---

## Como Executar

### Requisitos

```bash
# Python 3.9+ em ambiente Linux/WSL (recomendado para Ray/Flower)
python -m venv ~/fl-venv
source ~/fl-venv/bin/activate
pip install -U "flwr[simulation]"
pip install -r Code/tcc_code/requirements.txt
```

### Executar experimentos

```bash
cd Code/tcc_code

# Executar um algoritmo com uma estratégia
python main.py --algorithm xgboost --strategy cyclic

# Executar todos os 6 experimentos
python main.py --algorithm all --strategy both

# Customizar parâmetros
python main.py --algorithm lightgbm --strategy bagging \
    --num-clients 3 --num-rounds 50 --local-rounds 25
```

### Parâmetros disponíveis

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--algorithm` | `xgboost`, `lightgbm`, `catboost`, `all` | `xgboost` |
| `--strategy` | `cyclic`, `bagging`, `both` | `cyclic` |
| `--num-clients` | Número de clientes FL | `3` |
| `--num-rounds` | Rodadas de treinamento federado | `50` |
| `--local-rounds` | Rodadas locais de boosting | `25` |
| `--samples` | Amostras por cliente | `8000` |
| `--seed` | Semente aleatória | `42` |

---

## Estratégias de Agregação

**Cyclic (Cíclica)**: Um cliente por rodada, em sequência. O modelo global passa de cliente em cliente. Menor consumo de comunicação, melhor convergência.

**Bagging**: Todos os clientes treinam em paralelo e os modelos são agregados no servidor. Maior consumo de comunicação; neste trabalho apresentou TPR e AUC inferiores ao Cyclic.

---

## Arquitetura Modular

Cada algoritmo implementa três componentes separados:

- **`client.py`**: Classe FL client (`fit` + `evaluate`), serialização do modelo
- **`server.py`**: Função de avaliação centralizada e configuração de estratégia
- **`runner.py`**: Orquestração do experimento (criação de clientes, execução da simulação)

O módulo `common/` provê processamento de dados, cálculo de métricas (incluindo fairness), e logging estruturado em JSON.

---

## Dependências

```
flwr[simulation]>=1.6.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
ray>=2.6.0
```

---

## Licença

MIT License — veja [LICENSE](LICENSE).

---

**Última atualização**: Março/2026
