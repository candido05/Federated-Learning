# Federated Learning com Análise de Fairness

**Trabalho de Conclusão de Curso (TCC)**
**"Otimização de Modelos de Aprendizado Federado com SDN (Software-Defined Networking)"**

Bacharel em Ciência da Computação — Universidade Federal da Paraíba (UFPB)

**Autor**: Cândido Leandro de Queiroga Bisneto
**Orientador**: Prof. Fernando Menezes Matos

---

## Sobre o Projeto

Implementação completa de experimentos de **Aprendizado Federado (FL)** com modelos baseados em árvores de decisão aplicados ao dataset **Bank Account Fraud (BAF)**. O projeto avalia como diferentes estratégias de agregação federada impactam a **detecção de fraude** e a **equidade (fairness)** entre grupos demográficos, sem que os clientes compartilhem dados brutos entre si.

Foram avaliados três algoritmos de boosting (**XGBoost**, **LightGBM**, **CatBoost**) com duas estratégias de agregação (**Bagging** e **Cyclic**), totalizando **6 configurações**, cada uma com **50 rodadas** de treinamento federado e **otimização de hiperparâmetros via Optuna**.

---

## Resultados

### Desempenho Final — Teste Standard (FPR ≤ 5%)

| Algoritmo | Estratégia | TPR | AUC | Comunicação | Duração |
|-----------|-----------|-----|-----|-------------|---------|
| XGBoost | **Cycling** | **0.5574** | **0.8907** | 339 MB | 943s |
| XGBoost | Bagging | 0.4671 | 0.8548 | 1.013 GB | 1.079s |
| LightGBM | **Cycling** | **0.5588** | **0.8897** | 367 MB | 764s |
| LightGBM | Bagging | 0.4580 | 0.8474 | 1.078 GB | 2.574s |
| CatBoost | **Cycling** | **0.5518** | **0.8881** | 136 MB | 104s |
| CatBoost | Bagging | 0.4573 | 0.8355 | 407 MB | 248s |

### Com Pós-processamento de Fairness (FPR ≤ 5% por grupo demográfico)

| Algoritmo | Estratégia | TPR | AUC | Fairness Score |
|-----------|-----------|-----|-----|---------------|
| XGBoost | Cycling | 0.5308 | 0.8907 | **0.9987** |
| LightGBM | Cycling | 0.5350 | 0.8897 | 0.9865 |
| CatBoost | Cycling | 0.5329 | 0.8881 | 0.9717 |
| XGBoost | Bagging | 0.4545 | 0.8548 | 0.9884 |
| LightGBM | Bagging | 0.4447 | 0.8474 | 0.9867 |
| CatBoost | Bagging | 0.4489 | 0.8355 | 0.9916 |

**Conclusões principais:**
- **Cyclic supera Bagging** em TPR e AUC em todos os algoritmos, com 3–10x menos bytes transferidos
- O pós-processamento de fairness eleva o Fairness Score para ~0.97–0.999 com custo moderado em TPR (~2–3 p.p.)
- **CatBoost Cycling** foi o mais eficiente computacionalmente (104s, 136 MB)

---

## Dataset

**Bank Account Fraud (BAF)** — detecção de fraudes em abertura de contas bancárias com atributo sensível de grupo demográfico.

| Propriedade | Valor |
|------------|-------|
| Total de amostras | 999.641 |
| Features | 99 |
| Taxa de fraude | ~1,03% (treino) |
| Clientes FL | 3 (distribuição IID) |
| Amostras por cliente | ~264.893 |

O dataset não está incluído no repositório por ser grande demais. Consulte o [artigo original](baf_data_&_code/BAF_paper.pdf) incluído no repositório para mais detalhes.

---

## Estrutura do Repositório

```
Federated-Learning/
│
├── baf_data_&_code/                    # Implementação principal (BAF)
│   ├── baf_fl/                         # Módulo FL para o dataset BAF
│   │   ├── core/                       # Cliente FL, runner, serialização de modelos
│   │   ├── data/                       # Processamento de dados e métricas
│   │   ├── strategies/                 # Estratégias Bagging e Cycling
│   │   ├── tuning/                     # Otimização de hiperparâmetros (Optuna)
│   │   ├── reporting/                  # Logging, geração de gráficos e plots TCC
│   │   ├── config.py                   # Configuração central
│   │   ├── paths.py                    # Caminhos do projeto
│   │   └── main.py                     # Ponto de entrada
│   ├── notebooks/                      # Análise exploratória e benchmarks
│   │   ├── bank_account_fraud_baf_benchmark.ipynb
│   │   ├── bank_account_fraud_sota_benchmark.ipynb
│   │   └── bank_fraud_benchmark_models.ipynb
│   ├── BAF_paper.pdf                   # Artigo original do dataset
│   └── requirements.txt
│
├── Code/
│   ├── tcc_code/                       # Implementação modular (dataset de veículos)
│   │   ├── algorithms/                 # XGBoost, LightGBM, CatBoost (client/server/runner)
│   │   ├── common/                     # DataProcessor, métricas, logger, utils
│   │   └── main.py                     # CLI unificado
│   ├── fl_veiculos/                    # Implementação FL com dataset de veículos
│   ├── novo_tcc_code/                  # Notebooks intermediários do BAF (v1, v2)
│   ├── ml_code/                        # Protótipo inicial com modelos ML tradicionais
│   └── nn_code/                        # Referência: FL com redes neurais (PyTorch)
│
├── fl_simple_demo/                     # Demo mínima de Federated Learning com Flower
├── sdn_fl_test/                        # Teste de integração SDN com FL
├── Artigos/                            # Referências bibliográficas (PDFs)
├── doc/                                # Monografia em LaTeX (capítulos 01–09)
├── .gitattributes                      # Configura Python como linguagem principal
└── .gitignore                          # Exclui datasets, experimentos e imagens
```

> **Não rastreados pelo git** (excluídos por tamanho ou por serem regeneráveis):
> `experiments/`, `images/`, `dataset_fl/`, `baf_data_&_code/datasets/`, `baf_data_&_code/experiments/`

---

## Como Executar

### Requisitos

```bash
# Recomendado: Linux ou WSL (melhor compatibilidade com Ray/Flower)
python -m venv ~/fl-venv
source ~/fl-venv/bin/activate
pip install -r baf_data_&_code/requirements.txt
```

### Executar experimentos (BAF)

```bash
cd baf_data_&_code
python -m baf_fl.main
```

### Executar experimentos (dataset de veículos)

```bash
cd Code/tcc_code

# Um algoritmo
python main.py --algorithm xgboost --strategy cyclic

# Todos os 6 experimentos
python main.py --algorithm all --strategy both

# Customizar parâmetros
python main.py --algorithm lightgbm --strategy bagging \
    --num-clients 3 --num-rounds 50 --local-rounds 25
```

---

## Estratégias de Agregação

**Cyclic**: Um cliente por rodada, em sequência circular. O modelo global é passado de cliente em cliente. Menor overhead de comunicação e melhores métricas de desempenho neste trabalho.

**Bagging**: Todos os clientes treinam em paralelo e os modelos são agregados no servidor. Maior volume de bytes transferidos; apresentou TPR e AUC inferiores ao Cyclic em todos os algoritmos avaliados.

---

## Dependências Principais

```
flwr[simulation]>=1.6.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
scikit-learn>=1.3.0
optuna>=3.0.0
numpy>=1.24.0
pandas>=2.0.0
ray>=2.6.0
```

---

## Licença

MIT License — veja [LICENSE](LICENSE).

---

**Última atualização**: Março/2026
