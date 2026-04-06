"""
Configuracoes para FL em SDN/GNS3.
"""

# Rede - ajustar IPs conforme topologia do GNS3
SERVER_ADDRESS = "0.0.0.0:8080"       # servidor escuta em todas interfaces
SERVER_CONNECT = "127.0.0.1:8080"     # clientes conectam aqui (trocar pelo IP real no GNS3)
NUM_ROUNDS = 20
NUM_CLIENTS = 3
LOCAL_EPOCHS = 20  # n_estimators para o XGBoost
RANDOM_STATE = 42
