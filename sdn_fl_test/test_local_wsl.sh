#!/usr/bin/env bash
# ============================================================
#  test_local_wsl.sh
#  Testa o pipeline FL localmente no WSL.
#  Todos os logs aparecem em tempo real no mesmo terminal,
#  com prefixo colorido por processo.
#
#  Uso:
#    chmod +x test_local_wsl.sh
#    ./test_local_wsl.sh
#
#  Requisitos:
#    pip install flwr flwr-datasets xgboost datasets numpy
# ============================================================

SERVER_IP="127.0.0.1"
PORT=8080
ROUNDS=20
LOCAL_EPOCHS=20
NUM_CLIENTS=3
VENV_PYTHON="python3"   # troque pelo caminho do seu venv se necessário

# ── Cores ANSI ─────────────────────────────────────────────────────────────
RESET="\033[0m"
BOLD="\033[1m"

C_SERVER="\033[1;36m"   # ciano    — servidor
C_CLI0="\033[1;32m"     # verde    — cliente 0
C_CLI1="\033[1;33m"     # amarelo  — cliente 1
C_CLI2="\033[1;35m"     # magenta  — cliente 2
C_INFO="\033[1;37m"     # branco   — mensagens do script
C_ERR="\033[1;31m"      # vermelho — erros
C_OK="\033[1;92m"       # verde claro — sucesso

CLIENT_COLORS=("$C_CLI0" "$C_CLI1" "$C_CLI2")

# Arrays de PIDs para cleanup
declare -a ALL_PIDS=()
declare -a TAIL_PIDS=()

# ── Cleanup ────────────────────────────────────────────────────────────────
cleanup() {
    echo -e "\n${C_INFO}[SCRIPT] Encerrando todos os processos...${RESET}"
    for pid in "${ALL_PIDS[@]}" "${TAIL_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    rm -rf "$TMPDIR_FL" 2>/dev/null || true
    echo -e "${C_INFO}[SCRIPT] Pronto.${RESET}"
}
trap cleanup EXIT INT TERM

# ── Função: prefixar saída de um processo com cor ─────────────────────────
stream_log() {
    local fifo="$1"
    local prefix="$2"
    local color="$3"
    while IFS= read -r line; do
        echo -e "${color}${prefix}${RESET} ${line}"
    done < "$fifo" &
    TAIL_PIDS+=($!)
}

# ── Banner ─────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "══════════════════════════════════════════════════"
echo "   FL XGBoost — Logs em Tempo Real (WSL)"
echo "   Servidor : $SERVER_IP:$PORT"
echo "   Rounds   : $ROUNDS  |  Épocas locais: $LOCAL_EPOCHS"
echo "   Clientes : $NUM_CLIENTS"
echo "══════════════════════════════════════════════════"
echo -e "${RESET}"

# ── Legenda de cores ───────────────────────────────────────────────────────
echo -e "Legenda de cores:"
echo -e "  ${C_SERVER}[SERVER  ]${RESET} → servidor FL"
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    echo -e "  ${CLIENT_COLORS[$i]}[CLIENT $i]${RESET} → cliente $i"
done
echo ""

# ── 1. Verificar dependências ──────────────────────────────────────────────
echo -e "${C_INFO}[SCRIPT] Verificando dependências...${RESET}"
if ! $VENV_PYTHON -c "import flwr, xgboost, datasets, flwr_datasets, numpy" 2>/dev/null; then
    echo -e "${C_ERR}[ERRO] Dependências faltando. Execute:${RESET}"
    echo "  pip install flwr flwr-datasets xgboost datasets numpy"
    exit 1
fi
echo -e "${C_OK}[SCRIPT] Dependências OK!${RESET}\n"

# ── 2. Criar FIFOs (pipes nomeados) para streaming sem arquivos de log ─────
TMPDIR_FL=$(mktemp -d)
SERVER_FIFO="$TMPDIR_FL/server.fifo"
mkfifo "$SERVER_FIFO"

CLIENT_FIFOS=()
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    fifo="$TMPDIR_FL/client_$i.fifo"
    mkfifo "$fifo"
    CLIENT_FIFOS+=("$fifo")
done

# ── 3. Iniciar streamers ────────────────────────────────────────────────────
# Cada streamer lê seu FIFO e imprime no terminal com prefixo colorido
stream_log "$SERVER_FIFO" "[SERVER  ]" "$C_SERVER"
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    stream_log "${CLIENT_FIFOS[$i]}" "[CLIENT $i]" "${CLIENT_COLORS[$i]}"
done

# ── 4. Iniciar Servidor ─────────────────────────────────────────────────────
echo -e "${C_INFO}[SCRIPT] Iniciando servidor FL...${RESET}"
$VENV_PYTHON -u server.py \
    --host "$SERVER_IP" \
    --port "$PORT" \
    --rounds "$ROUNDS" \
    --min-clients "$NUM_CLIENTS" \
    > "$SERVER_FIFO" 2>&1 &
SERVER_PID=$!
ALL_PIDS+=($SERVER_PID)

echo -e "${C_INFO}[SCRIPT] Aguardando servidor subir (5s)...${RESET}"
sleep 5

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo -e "${C_ERR}[ERRO] Servidor falhou ao iniciar!${RESET}"
    exit 1
fi
echo -e "${C_OK}[SCRIPT] Servidor ativo (PID $SERVER_PID)!${RESET}\n"

# ── 5. Iniciar Clientes ─────────────────────────────────────────────────────
echo -e "${C_INFO}[SCRIPT] Iniciando $NUM_CLIENTS clientes...${RESET}\n"
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    $VENV_PYTHON -u client.py \
        --server-ip "$SERVER_IP" \
        --port "$PORT" \
        --partition-id "$i" \
        --num-partitions "$NUM_CLIENTS" \
        --local-epochs "$LOCAL_EPOCHS" \
        > "${CLIENT_FIFOS[$i]}" 2>&1 &
    CLIENT_PID=$!
    ALL_PIDS+=($CLIENT_PID)
    echo -e "${CLIENT_COLORS[$i]}[CLIENT $i]${RESET} Iniciado (PID $CLIENT_PID)"
    sleep 1
done

echo ""
echo -e "${C_INFO}[SCRIPT] Todos rodando. Pressione Ctrl+C para encerrar.${RESET}"
echo "──────────────────────────────────────────────────"
echo ""

# ── 6. Aguarda servidor terminar ────────────────────────────────────────────
wait "$SERVER_PID"
EXIT_CODE=$?

echo ""
echo "══════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${C_OK}  Treinamento concluído com sucesso!${RESET}"
else
    echo -e "${C_ERR}  Servidor encerrou com erro (código $EXIT_CODE)${RESET}"
fi
[ -f "final_model.json" ] && echo -e "${C_OK}  Modelo salvo em: final_model.json${RESET}"
echo "══════════════════════════════════════════════════"

# Aguarda clientes encerrarem
for pid in "${ALL_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done