# Gerenciador do Servidor Federated Learning

Implementação do servidor FL que coordena experimentos, estratégias, clientes e avaliação centralizada.

## Visão Geral

O módulo `server/` fornece:
- **FederatedServer**: Gerenciador principal que orquestra experimentos FL
- **get_evaluate_fn()**: Cria funções de avaliação centralizada compatível com Flower
- Detecção automática de GPU
- Fallback para diferentes versões do Flower
- Logging integrado com ExperimentLogger

## Arquitetura

```
server/
├── __init__.py           - Exporta FederatedServer e get_evaluate_fn
├── server_manager.py     - Classe FederatedServer
└── evaluation.py         - Funções de avaliação centralizada
```

---

## get_evaluate_fn()

Cria função de avaliação centralizada que roda no servidor.

### Assinatura

```python
def get_evaluate_fn(
    test_data: Tuple[np.ndarray, np.ndarray],
    model_type: str,
    params: dict,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
```

### Parâmetros

- **test_data**: Tupla `(X_test, y_test)` para avaliação
- **model_type**: `'xgboost'`, `'catboost'` ou `'lightgbm'`
- **params**: Parâmetros do modelo (não usado atualmente, para compatibilidade futura)

### Retorna

Função `evaluate_fn(server_round, parameters_ndarrays, config)` que:
1. Desserializa modelo dos bytes recebidos
2. Faz predições no dataset de teste
3. Calcula métricas comprehensivas (accuracy, precision, recall, F1, AUC)
4. Retorna `(loss, metrics_dict)`

### Exemplo de Uso

```python
from server import get_evaluate_fn

# Dados de teste
X_test, y_test = dataset.X_test, dataset.y_test
test_data = (X_test, y_test)

# Cria função de avaliação
evaluate_fn = get_evaluate_fn(
    test_data=test_data,
    model_type='xgboost',
    params={},
)

# Usa em estratégia Flower
strategy = FedBagging(evaluate_fn=evaluate_fn)
```

### Funcionamento Interno

```python
# Dentro de evaluate_fn retornada:
def evaluate_fn(server_round, parameters_ndarrays, config):
    # 1. Converte NDArrays para bytes
    model_bytes = parameters_ndarrays[0].tobytes()

    # 2. Carrega modelo específico
    if model_type == 'xgboost':
        model = _load_xgboost_model(model_bytes)   # Arquivo temp .json
        predictions = _predict_xgboost(model, X_test)

    elif model_type == 'catboost':
        model = _load_catboost_model(model_bytes)  # Arquivo temp .cbm
        predictions = _predict_catboost(model, X_test)

    elif model_type == 'lightgbm':
        model = _load_lightgbm_model(model_bytes) # Arquivo temp .txt
        predictions = _predict_lightgbm(model, X_test)

    # 3. Calcula métricas
    metrics = MetricsCalculator().calculate_comprehensive_metrics(
        y_true=y_test,
        y_pred_proba=predictions
    )

    # 4. Retorna loss e métricas
    loss = 1.0 - metrics['accuracy']
    return loss, {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
    }
```

### Suporte a Frameworks

| Framework | Formato | Função de Load | Função de Predict | Conversão |
|-----------|---------|----------------|-------------------|-----------|
| XGBoost | JSON | `_load_xgboost_model()` | `_predict_xgboost()` | Binário → `[n, 2]` |
| CatBoost | CBM | `_load_catboost_model()` | `_predict_catboost()` | Automático |
| LightGBM | Texto | `_load_lightgbm_model()` | `_predict_lightgbm()` | Binário → `[n, 2]` |

Todas as funções de load/predict usam arquivos temporários com timestamps únicos e limpeza automática.

---

## FederatedServer

Classe principal que gerencia experimentos FL.

### Construtor

```python
def __init__(self, config: GlobalConfig, logging_config: LoggingConfig)
```

**Argumentos**:
- `config`: Configuração global (num_clients, num_rounds, etc.)
- `logging_config`: Configuração de logging

**Exemplo**:
```python
from config import GlobalConfig, LoggingConfig
from server import FederatedServer

config = GlobalConfig(
    num_clients=10,
    num_rounds=20,
    sample_per_client=5000,
    num_local_rounds=10,
    seed=42,
)

logging_config = LoggingConfig(
    base_dir="logs",
    log_to_file=True,
    log_to_console=True,
)

server = FederatedServer(config, logging_config)
```

### Atributos

| Atributo | Tipo | Descrição |
|----------|------|-----------|
| `config` | GlobalConfig | Configuração global |
| `logging_config` | LoggingConfig | Configuração de logging |
| `logger` | Logger | Logger básico do servidor |
| `experiment_logger` | ExperimentLogger | Logger de experimentos (criado no setup) |
| `dataset` | BaseDataset | Dataset carregado (criado no setup) |
| `current_model_type` | str | Tipo de modelo atual |
| `current_strategy_type` | str | Tipo de estratégia atual |

---

## Métodos do FederatedServer

### 1. setup_experiment()

Configura experimento FL (logger, dataset, metadados).

```python
def setup_experiment(
    self,
    model_type: str,
    strategy_type: str,
    dataset_source: str = "jxie/higgs",
) -> None
```

**O que faz**:
1. Valida `model_type` e `strategy_type`
2. Cria `ExperimentLogger` para este experimento
3. Carrega dataset do HuggingFace
4. Particiona dados entre clientes
5. Armazena metadados

**Exemplo**:
```python
server.setup_experiment(
    model_type='xgboost',
    strategy_type='bagging',
    dataset_source='jxie/higgs',
)

# Agora server.dataset está disponível
print(f"Treino: {len(server.dataset.X_train)} amostras")
print(f"Teste: {len(server.dataset.X_test)} amostras")
```

---

### 2. create_strategy()

Cria estratégia de agregação FL.

```python
def create_strategy(
    self,
    strategy_type: str,
    model_type: str,
    evaluate_fn: Optional[Callable] = None,
    params: Optional[Dict] = None,
) -> BaseStrategy
```

**Parâmetros**:
- `strategy_type`: `'bagging'` ou `'cyclic'`
- `model_type`: Para logging (não afeta lógica da estratégia)
- `evaluate_fn`: Função de avaliação centralizada
- `params`: Parâmetros customizados (fraction_fit, etc.)

**Retorna**: Estratégia configurada (`FedBagging` ou `FedCyclic`)

**Exemplo**:
```python
# Cria função de avaliação
evaluate_fn = get_evaluate_fn(test_data, 'xgboost', {})

# Cria estratégia Bagging
strategy = server.create_strategy(
    strategy_type='bagging',
    model_type='xgboost',
    evaluate_fn=evaluate_fn,
    params={
        'fraction_fit': 0.5,        # 50% dos clientes
        'fraction_evaluate': 1.0,   # Avalia em todos
    }
)

# Cria estratégia Cyclic
strategy = server.create_strategy(
    strategy_type='cyclic',
    model_type='catboost',
    evaluate_fn=evaluate_fn,
)
```

**Parâmetros padrão**:
```python
# FedBagging
{
    'fraction_fit': 1.0,               # Todos os clientes
    'fraction_evaluate': 1.0,          # Avalia em todos
    'min_fit_clients': config.num_clients,
    'min_evaluate_clients': 2,
    'min_available_clients': config.num_clients,
}

# FedCyclic
{
    'fraction_evaluate': 1.0,
    'min_evaluate_clients': 1,
    'min_available_clients': 1,
}
```

---

### 3. create_client_fn()

Cria função `client_fn` compatível com Flower.

```python
def create_client_fn(
    self,
    model_type: str,
    dataset,
    params: Optional[Dict] = None,
) -> Callable[[str], Any]
```

**Parâmetros**:
- `model_type`: `'xgboost'`, `'catboost'` ou `'lightgbm'`
- `dataset`: Dataset com partições de clientes
- `params`: Parâmetros do modelo (usa padrão se None)

**Retorna**: Função `client_fn(cid: str)` que cria clientes

**Exemplo**:
```python
# Cria função de criação de clientes
client_fn = server.create_client_fn(
    model_type='lightgbm',
    dataset=server.dataset,
    params=LIGHTGBM_PARAMS,  # Opcional
)

# Flower usa essa função internamente
client_0 = client_fn("0")  # Cria LightGBMClient para cliente 0
client_1 = client_fn("1")  # Cria LightGBMClient para cliente 1
```

**Lógica interna**:
```python
def client_fn(cid: str):
    client_idx = int(cid)
    train_data, valid_data = dataset.get_client_data(client_idx)

    if model_type == 'xgboost':
        return XGBoostClient(train_data, valid_data, ...)
    elif model_type == 'catboost':
        return CatBoostClient(train_data, valid_data, ...)
    elif model_type == 'lightgbm':
        return LightGBMClient(train_data, valid_data, ...)
```

---

### 4. run_experiment()

Executa experimento FL completo (método principal).

```python
def run_experiment(
    self,
    model_type: str,
    strategy_type: str,
    dataset_source: str = "jxie/higgs",
) -> Dict[str, Any]
```

**Parâmetros**:
- `model_type`: Tipo do modelo
- `strategy_type`: Tipo da estratégia
- `dataset_source`: Dataset HuggingFace

**Retorna**: Dicionário com resultados
```python
{
    'history': History,        # Histórico do Flower
    'model_type': str,
    'strategy_type': str,
    'num_rounds': int,
    'num_clients': int,
}
```

**Exemplo completo**:
```python
from config import GlobalConfig, LoggingConfig
from server import FederatedServer

# Configuração
config = GlobalConfig(num_clients=10, num_rounds=20, ...)
logging_config = LoggingConfig(...)

# Cria servidor
server = FederatedServer(config, logging_config)

# Executa experimento (tudo automático!)
results = server.run_experiment(
    model_type='xgboost',
    strategy_type='bagging',
    dataset_source='jxie/higgs',
)

# Acessa resultados
print(f"Rounds executados: {results['num_rounds']}")
print(f"Histórico: {results['history']}")
```

**Fluxo interno**:
```
1. setup_experiment()
   ├─ Cria ExperimentLogger
   ├─ Carrega dataset
   └─ Particiona dados

2. get_evaluate_fn()
   └─ Cria função de avaliação centralizada

3. create_strategy()
   └─ Configura FedBagging ou FedCyclic

4. create_client_fn()
   └─ Função para criar XGBoost/CatBoost/LightGBM clients

5. _detect_gpu_and_configure()
   └─ Detecta GPU e configura backend

6. _safe_run_simulation()
   ├─ Tenta start_simulation (Flower moderno)
   ├─ Fallback para versão legada se necessário
   └─ Retorna History

7. experiment_logger.save_summary()
   └─ Salva resultados em JSON
```

---

### 5. _safe_run_simulation()

Executa simulação FL com fallback para diferentes versões do Flower.

```python
def _safe_run_simulation(
    self,
    client_fn: Callable,
    num_clients: int,
    config: ServerConfig,
    strategy: BaseStrategy,
    backend_config: Optional[Dict] = None,
) -> Any
```

**Estratégia de fallback**:
1. **Tenta**: `flwr.simulation.start_simulation` (versão moderna)
2. **Fallback**: Método legado se ImportError

**Exemplo de uso interno**:
```python
history = self._safe_run_simulation(
    client_fn=client_fn,
    num_clients=10,
    config=ServerConfig(num_rounds=20),
    strategy=strategy,
    backend_config={'client_resources': {'num_cpus': 2, 'num_gpus': 0}},
)
```

---

### 6. _detect_gpu_and_configure()

Detecta GPU NVIDIA e configura recursos do backend.

```python
def _detect_gpu_and_configure(self) -> Optional[Dict]
```

**Retorna**:
```python
# Se GPU detectada:
{
    'client_resources': {
        'num_cpus': 1,
        'num_gpus': 0.2,  # Fração de GPU por cliente
    }
}

# Se apenas CPU:
{
    'client_resources': {
        'num_cpus': 2,
        'num_gpus': 0.0,
    }
}
```

**Detecção**:
- Executa `nvidia-smi --query-gpu=name,memory.total`
- Se bem-sucedido: configura GPU
- Se falhar: fallback para CPU

**Logs gerados**:
```
INFO - GPU detectada: NVIDIA GeForce RTX 3080, 10240 MiB
INFO - Usando CPU para simulação
```

---

### 7. _get_model_params()

Obtém parâmetros padrão para tipo de modelo.

```python
def _get_model_params(self, model_type: str) -> Dict
```

**Retorna**:
- `'xgboost'` → `XGBOOST_PARAMS`
- `'catboost'` → `CATBOOST_PARAMS`
- `'lightgbm'` → `LIGHTGBM_PARAMS`

---

## Exemplo Completo de Uso

```python
from config import GlobalConfig, LoggingConfig
from server import FederatedServer

# 1. Configuração
config = GlobalConfig(
    num_clients=10,
    num_rounds=20,
    sample_per_client=5000,
    num_local_rounds=10,
    seed=42,
)

logging_config = LoggingConfig(
    base_dir="logs",
    log_to_file=True,
    log_to_console=True,
)

# 2. Cria servidor
server = FederatedServer(config, logging_config)

# 3. Executa experimento (método simples)
results = server.run_experiment(
    model_type='xgboost',
    strategy_type='bagging',
    dataset_source='jxie/higgs',
)

# 4. Acessa resultados
print(f"Experimento concluído!")
print(f"Modelo: {results['model_type']}")
print(f"Estratégia: {results['strategy_type']}")
print(f"Rounds: {results['num_rounds']}")
```

**Ou controle granular**:

```python
# 1. Setup manual
server.setup_experiment('catboost', 'cyclic')

# 2. Cria componentes manualmente
test_data = (server.dataset.X_test, server.dataset.y_test)
evaluate_fn = get_evaluate_fn(test_data, 'catboost', {})

strategy = server.create_strategy('cyclic', 'catboost', evaluate_fn)
client_fn = server.create_client_fn('catboost', server.dataset)

# 3. Executa simulação manualmente
from flwr.simulation import start_simulation
from flwr.server import ServerConfig

history = start_simulation(
    client_fn=client_fn,
    num_clients=server.config.num_clients,
    config=ServerConfig(num_rounds=server.config.num_rounds),
    strategy=strategy,
)
```

---

## Logging

### Logs do Servidor

```
INFO - Servidor Federado inicializado
INFO - Configuração: 10 clientes, 20 rounds
INFO - Configurando experimento: xgboost + bagging
INFO - Carregando dataset: jxie/higgs
INFO - Criando estratégia: bagging
INFO - Estratégia bagging criada com sucesso
INFO - GPU detectada: NVIDIA GeForce RTX 3080, 10240 MiB
INFO - Iniciando simulação FL: 20 rounds, 10 clientes
INFO - Usando flwr.simulation.start_simulation
INFO - Simulação FL concluída com sucesso
```

### Logs do ExperimentLogger

Salvos em `logs/{model_type}/{strategy_type}/{timestamp}/`:
```
experiment.log          - Logs textuais
summary.json           - Resumo do experimento
round_X_metrics.json   - Métricas por round
```

---

## Detecção de GPU

### Como Funciona

1. **Tenta executar**: `nvidia-smi --query-gpu=name,memory.total`
2. **Se sucesso**: Configura `num_gpus=0.2` (20% da GPU por cliente)
3. **Se falha**: Configura `num_gpus=0.0` (apenas CPU)

### Saída de nvidia-smi

```bash
$ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NVIDIA GeForce RTX 3080, 10240 MiB
```

### Configuração Resultante

```python
# GPU disponível
backend_config = {
    'client_resources': {
        'num_cpus': 1,
        'num_gpus': 0.2,  # 5 clientes simultâneos em 1 GPU
    }
}

# Apenas CPU
backend_config = {
    'client_resources': {
        'num_cpus': 2,
        'num_gpus': 0.0,
    }
}
```

---

## Tratamento de Erros

### Validação de Inputs

```python
# Modelo inválido
server.setup_experiment('invalid_model', 'bagging')
# ValueError: Modelo 'invalid_model' inválido. Use: ['xgboost', 'catboost', 'lightgbm']

# Estratégia inválida
server.setup_experiment('xgboost', 'invalid_strategy')
# ValueError: Estratégia 'invalid_strategy' inválida. Use: ['bagging', 'cyclic']
```

### Falhas em Avaliação Centralizada

```python
# Em evaluate_fn retornada por get_evaluate_fn():
try:
    model = load_model(model_bytes)
    predictions = predict(model, X_test)
    metrics = calculate_metrics(y_test, predictions)
    return loss, metrics
except Exception as e:
    print(f"[ERRO] Avaliação centralizada falhou no round {server_round}: {e}")
    return None  # Não interrompe treinamento
```

### Fallback do Flower

```python
try:
    from flwr.simulation import start_simulation
    # Usa versão moderna
except ImportError:
    # Fallback para versão legada
    from flwr.server import start_server
    # ...
```

---

## Compatibilidade com Flower

### Versões Suportadas

| Versão Flower | Método | Status |
|---------------|--------|--------|
| >= 1.5.0 | `start_simulation()` | ✅ Preferencial |
| < 1.5.0 | Método legado | ✅ Fallback automático |

### Assinatura de start_simulation

```python
from flwr.simulation import start_simulation

start_simulation(
    client_fn=client_fn,           # Função que cria clientes
    num_clients=10,                 # Número total de clientes
    config=ServerConfig(num_rounds=20),  # Configuração do servidor
    strategy=strategy,              # Estratégia de agregação
    client_resources={              # Recursos por cliente (opcional)
        'num_cpus': 2,
        'num_gpus': 0.2,
    }
)
```

---

## Testes

Execute `test_server.py` para validar componentes:

```bash
cd Code/tcc_code
python test_server.py
```

### Testes Realizados

1. ✅ **get_evaluate_fn**: Cria funções para XGBoost, CatBoost, LightGBM
2. ✅ **FederatedServer**: Construtor e atributos
3. ✅ **create_strategy**: Cria FedBagging e FedCyclic
4. ✅ **_detect_gpu_and_configure**: Detecta GPU ou fallback CPU
5. ✅ **_get_model_params**: Retorna parâmetros corretos
6. ✅ **setup_experiment**: Carrega dataset e configura logger

### Saída Esperada

```
============================================================
INICIANDO TESTES DO SERVIDOR FEDERATED LEARNING
============================================================

============================================================
Testando get_evaluate_fn
============================================================

[TESTE] Criando evaluate_fn para xgboost...
✓ evaluate_fn criada para xgboost
✓ Tipo: <class 'function'>
✓ Callable: True

[TESTE] Criando evaluate_fn para catboost...
✓ evaluate_fn criada para catboost
[...]

============================================================
✓✓✓ TESTES DE evaluate_fn PASSARAM ✓✓✓
============================================================

============================================================
Testando FederatedServer
============================================================

[TESTE 1] Criando configurações...
✓ Configurações criadas

[TESTE 2] Criando FederatedServer...
✓ FederatedServer criado com sucesso
✓ Número de clientes: 4
✓ Número de rounds: 3

[TESTE 3] Testando create_strategy...
✓ Estratégia bagging criada: FedBagging
✓ Estratégia cyclic criada: FedCyclic

[TESTE 4] Testando detecção de GPU...
✓ Backend configurado: {'client_resources': {'num_cpus': 2, 'num_gpus': 0.0}}
✓ CPU configurada (GPU não detectada)

[TESTE 5] Testando _get_model_params...
✓ Parâmetros xgboost: 5 chaves
✓ Parâmetros catboost: 6 chaves
✓ Parâmetros lightgbm: 5 chaves

============================================================
✓✓✓ TESTES DE FederatedServer PASSARAM ✓✓✓
============================================================

[...]

============================================================
RESUMO DOS TESTES
============================================================
evaluate_fn         : ✓ PASSOU
FederatedServer     : ✓ PASSOU
setup_experiment    : ✓ PASSOU
============================================================

✓✓✓ TODOS OS TESTES DO SERVIDOR PASSARAM ✓✓✓
```

---

## Próximos Passos

Com o servidor implementado, você pode:

1. **Criar script de execução principal** (`main.py`)
2. **Executar experimentos completos** com diferentes combinações
3. **Analisar resultados** do `History` retornado
4. **Visualizar métricas** ao longo dos rounds
5. **Comparar estratégias** (Bagging vs Cyclic)
6. **Comparar modelos** (XGBoost vs CatBoost vs LightGBM)

Veja `CLAUDE.md` para roadmap completo do TCC.
