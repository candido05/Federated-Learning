# Aprendizado Federado com Modelos Baseados em Árvores - Projeto TCC

## Visão Geral

Este projeto implementa **Aprendizado Federado (FL)** com modelos baseados em árvores (**XGBoost**, **LightGBM**, **CatBoost**) usando o framework **Flower**. Faz parte de um TCC (Trabalho de Conclusão de Curso) investigando a otimização de modelos de aprendizado federado com SDN (Software-Defined Networking).

### Foco da Pesquisa

- **Objetivo Principal**: Avaliar modelos baseados em árvores em ambientes federados
- **Framework**: Flower (flwr) para orquestração federada
- **Integração SDN**: Otimizar tráfego de rede e comunicação entre clientes e servidor FL
- **Cenários**: Comparar desempenho em distribuições de dados IID vs. non-IID
- **Métricas**: Acurácia de classificação, tempo de convergência, overhead de comunicação, latência, consumo de banda

## Estrutura do Projeto

```
tcc_code/
├── config/                  # Arquivos de configuração
│   ├── __init__.py
│   ├── config.py           # Dataclasses GlobalConfig e LoggingConfig
│   └── model_params.py     # Hiperparâmetros dos modelos
├── data/                    # Carregamento e particionamento de dados
│   └── __init__.py
├── models/                  # Implementações dos modelos
│   └── __init__.py
├── strategies/              # Estratégias FL (FedAvg, FedProx, etc.)
│   └── __init__.py
├── utils/                   # Funções utilitárias
│   └── __init__.py
├── server/                  # Implementação do servidor FL
│   └── __init__.py
├── logs/                    # Arquivos de log (criados em tempo de execução)
├── main.py                  # Ponto de entrada
├── requirements.txt         # Dependências Python
├── .gitignore              # Regras do Git
└── README.md               # Este arquivo
```

## Instalação

### 1. Criar Ambiente Virtual

```bash
# Navegar para o diretório do projeto
cd Code/tcc_code

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

## Configuração

### Configuração Global

Edite `config/config.py` para modificar as configurações do experimento:

```python
GlobalConfig(
    num_clients=6,              # Número de clientes FL
    sample_per_client=8000,     # Amostras por cliente
    num_server_rounds=6,        # Rodadas de comunicação
    num_local_boost_round=20,   # Rodadas de boosting local
    seed=42,                    # Semente aleatória para reprodutibilidade
    test_fraction=0.2,          # Fração dos dados de teste
    dataset_name="higgs"        # Dataset a ser usado
)
```

### Hiperparâmetros dos Modelos

Modifique `config/model_params.py` para ajustar os hiperparâmetros dos modelos:

- **XGBoost**: `XGBOOST_PARAMS` (seed=42)
- **CatBoost**: `CATBOOST_PARAMS` (random_seed=42)
- **LightGBM**: `LIGHTGBM_PARAMS` (seed=42)

### Configuração de Logging

Controle o comportamento de logging em `config/config.py`:

```python
LoggingConfig(
    log_dir="logs",             # Diretório de logs
    save_client_logs=True,      # Salvar logs individuais dos clientes
    save_round_logs=True,       # Salvar logs rodada por rodada
    verbose=True                # Imprimir logs detalhados
)
```

## Uso

### Executando Experimentos

```bash
# Executar experimento principal
python main.py
```

### Argumentos de Linha de Comando

```bash
# Especificar modelo
python main.py --model xgboost

# Especificar estratégia FL
python main.py --strategy FedAvg

# Configuração customizada
python main.py --num-clients 10 --num-rounds 10

# Configurar semente aleatória
python main.py --seed 42

# Habilitar logging verboso
python main.py --verbose
```

## Modelos Suportados

1. **XGBoost** - Gradient boosting com algoritmo baseado em histograma
2. **LightGBM** - Framework rápido de gradient boosting
3. **CatBoost** - Gradient boosting com suporte a features categóricas

## Estratégias de Aprendizado Federado

- **FedAvg**: Média federada padrão
- **FedProx**: Termo proximal para dados heterogêneos
- **FedAdam**: Taxa de aprendizado adaptativa no servidor
- **FedAdagrad**: Agregação de gradiente adaptativa
- **FedYogi**: Otimização adaptativa
- **FedMedian**: Robusto a outliers

## Dataset

Atualmente configurado para o dataset **HIGGS** (classificação binária).

### Particionamento de Dados

- **IID**: Distribuição independente e idêntica entre clientes
- **Non-IID**: Particionamento heterogêneo baseado em Dirichlet

## Logging e Resultados

Logs são salvos no diretório `logs/`:

- **Logs de clientes**: Logs de treinamento individual dos clientes
- **Logs de rodadas**: Métricas por rodada de comunicação
- **Resultados**: Arquivos CSV com acurácia, loss e métricas de convergência

## Desenvolvimento

### Estilo de Código

- **Type hints**: Todas as funções usam anotações de tipo
- **Docstrings**: Docstrings estilo Google para todos os módulos, classes e funções
- **Formatação**: Segue diretrizes PEP 8

### Adicionando Novos Modelos

1. Adicionar parâmetros do modelo em `config/model_params.py`
2. Implementar wrapper do modelo em `models/`
3. Atualizar imports de configuração

### Adicionando Novas Estratégias

1. Criar classe de estratégia em `strategies/`
2. Herdar da estratégia base do Flower
3. Registrar na configuração do servidor

## Solução de Problemas

### Problemas Comuns

1. **Erros de import**: Certifique-se de que o ambiente virtual está ativado
2. **Erros CUDA**: XGBoost/LightGBM podem requerer instalação específica para GPU
3. **Erros de memória**: Reduza `sample_per_client` ou `num_clients`

### Problemas de Logging

Se os logs não estão sendo salvos:
- Verifique as permissões do diretório `logs/`
- Verifique o caminho `LoggingConfig.log_dir`
- Certifique-se de que `save_client_logs` e `save_round_logs` estão habilitados

## Referências

- [Documentação do Flower](https://flower.dev/docs/)
- [Documentação do XGBoost](https://xgboost.readthedocs.io/)
- [Documentação do LightGBM](https://lightgbm.readthedocs.io/)
- [Documentação do CatBoost](https://catboost.ai/docs/)