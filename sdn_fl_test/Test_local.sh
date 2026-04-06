#!/bin/bash
# ============================================================
# test_local.sh - Teste rápido: servidor + 2 clientes no mesmo WSL
# ============================================================
# Este script roda tudo localmente para validar que o código funciona
# antes de partir para o teste distribuído com múltiplas VMs.
#
# Uso:
#   chmod +x test_local.sh
#   ./test_local.sh
# ============================================================

set -e

echo "============================================"
echo "  TESTE LOCAL - FL XGBoost (mesmo WSL)"
echo "============================================"

NUM_CLIENTS=2
ROUNDS=3
METHOD="cyclic"
SERVER_ADDR="127.0.0.1:8080"

# Inicia o servidor em background
echo ""
echo "[1/3] Iniciando servidor Flower..."
python server.py \
    --address "0.0.0.0:8080" \
    --rounds $ROUNDS \
    --method $METHOD \
    --min-clients $NUM_CLIENTS \
    --centralised-eval &
SERVER_PID=$!
echo "  PID do servidor: $SERVER_PID"

# Aguarda o servidor subir
echo "  Aguardando servidor inicializar (15s para baixar dataset)..."
sleep 15

# Inicia os clientes
echo ""
echo "[2/3] Iniciando $NUM_CLIENTS clientes..."

CLIENT_PIDS=()
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    echo "  Iniciando cliente $i..."
    python client.py \
        --server $SERVER_ADDR \
        --partition-id $i \
        --num-partitions $NUM_CLIENTS \
        --method $METHOD \
        --num-local-rounds 1 &
    CLIENT_PIDS+=($!)
    echo "  PID do cliente $i: ${CLIENT_PIDS[-1]}"
    sleep 2
done

# Aguarda todos finalizarem
echo ""
echo "[3/3] Aguardando treinamento federado..."
wait $SERVER_PID
echo ""
echo "============================================"
echo "  TESTE CONCLUÍDO COM SUCESSO!"
echo "============================================"