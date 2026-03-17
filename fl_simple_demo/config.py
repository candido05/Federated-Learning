"""
Configuracoes centralizadas do FL Simple Demo.
Modifique os valores abaixo conforme necessario.
"""

# ---------------------------------------------------------------------------
# Rede
# ---------------------------------------------------------------------------
SERVER_HOST = "0.0.0.0"       # IP em que o servidor escuta (0.0.0.0 = todas as interfaces)
SERVER_PORT = 8080             # Porta gRPC do servidor
SERVER_ADDRESS = f"{SERVER_HOST}:{SERVER_PORT}"

# Endereco que os CLIENTES usam para conectar ao servidor
# Se rodar tudo na mesma maquina, use 127.0.0.1
# Se rodar em maquinas diferentes, coloque o IP real do servidor
CLIENT_CONNECT_ADDRESS = f"127.0.0.1:{SERVER_PORT}"

# ---------------------------------------------------------------------------
# Treinamento Federado
# ---------------------------------------------------------------------------
NUM_CLIENTS = 3                # Numero de clientes federados
NUM_ROUNDS = 5                 # Rounds de comunicacao servidor-clientes
LOCAL_EPOCHS = 100             # Numero de estimadores/iteracoes por treino local
                               # (n_estimators para XGBoost/LightGBM, iterations para CatBoost)
LOG_EVERY = 10                 # Mostrar progresso a cada N epocas locais

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
N_SAMPLES = 50_000             # Amostras do Higgs a carregar (total)
TEST_SIZE = 0.2                # Fracao para teste
RANDOM_SEED = 42               # Seed para reproducibilidade

# ---------------------------------------------------------------------------
# Modelos - Hiperparametros base
# ---------------------------------------------------------------------------
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": LOCAL_EPOCHS,
    "random_state": RANDOM_SEED,
}

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": LOCAL_EPOCHS,
    "random_state": RANDOM_SEED,
}

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "depth": 6,
    "learning_rate": 0.1,
    "iterations": LOCAL_EPOCHS,
    "random_seed": RANDOM_SEED,
    "verbose": False,
}
