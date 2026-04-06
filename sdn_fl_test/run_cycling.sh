#!/bin/bash
# Roda XGBoost com estratégia CYCLING

DATASET="breast_cancer"
MODEL="xgboost"
STRATEGY="cycling"
ROUNDS=20
LOCAL_EPOCHS=10
PORT=8080
NUM_CLIENTS=3

echo "============================================================"
echo "  FEDERATED LEARNING - ESTRATEGIA: CYCLING (XGBOOST)"
echo "============================================================"

# Servidor em background
python server.py \
    --strategy "$STRATEGY" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --rounds "$ROUNDS" \
    --num-clients "$NUM_CLIENTS" \
    --address "0.0.0.0:$PORT" &
SERVER_PID=$!

sleep 5

# Iniciar clientes
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    python client.py \
        --client-id "$i" \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --num-clients "$NUM_CLIENTS" &
done

# Esperar servidor terminar
wait $SERVER_PID
echo ">>> CYCLING CONCLUIDO"