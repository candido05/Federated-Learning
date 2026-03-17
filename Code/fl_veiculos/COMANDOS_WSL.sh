#!/bin/bash
# Script com todos os comandos para executar no WSL

echo "=========================================="
echo "FEDERATED LEARNING - COMANDOS PARA WSL"
echo "=========================================="
echo ""

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}1. NAVEGANDO PARA O DIRETÓRIO${NC}"
cd /mnt/c/Users/candi/OneDrive/Desktop/Federated-Learning/Code/tcc_code
echo -e "${GREEN}[OK] Diretório: $(pwd)${NC}"
echo ""

echo -e "${BLUE}2. ATIVANDO AMBIENTE VIRTUAL${NC}"
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}[OK] Ambiente virtual ativado${NC}"
else
    echo -e "${YELLOW}[AVISO] Ambiente virtual não encontrado em venv/${NC}"
    echo "Ajuste o caminho do ambiente virtual"
fi
echo ""

echo -e "${BLUE}3. VERIFICANDO DEPENDÊNCIAS${NC}"
python -c "import flwr, xgboost, lightgbm, catboost; print('[OK] Todas as dependências instaladas')" 2>/dev/null && echo -e "${GREEN}[OK] OK${NC}" || {
    echo -e "${YELLOW}[AVISO] Instalando dependências...${NC}"
    pip install -r requirements.txt
}
echo ""

echo "=========================================="
echo "COMANDOS DISPONÍVEIS:"
echo "=========================================="
echo ""
echo "TESTE RÁPIDO (3 clientes, 3 rodadas, poucos dados):"
echo -e "${YELLOW}PYTHONPATH=. python run_experiments.py --algorithm xgboost --num-clients 3 --num-rounds 3 --samples 2000${NC}"
echo ""
echo "EXPERIMENTO PADRÃO (XGBoost, ambas estratégias):"
echo -e "${YELLOW}PYTHONPATH=. python run_experiments.py --algorithm xgboost --strategy both${NC}"
echo ""
echo "TODOS OS ALGORITMOS (6 experimentos):"
echo -e "${YELLOW}PYTHONPATH=. python run_experiments.py --algorithm all --strategy both${NC}"
echo ""
echo "=========================================="
echo ""

# Perguntar qual executar
echo "Deseja executar algum experimento agora? (1/2/3/N)"
echo "1) Teste rápido"
echo "2) Experimento padrão"  
echo "3) Todos os algoritmos"
echo "N) Não, apenas preparar ambiente"
echo ""
read -p "Escolha: " choice

case $choice in
    1)
        echo -e "${BLUE}Executando teste rápido...${NC}"
        PYTHONPATH=. python run_experiments.py --algorithm xgboost --num-clients 3 --num-rounds 3 --samples 2000
        ;;
    2)
        echo -e "${BLUE}Executando experimento padrão...${NC}"
        PYTHONPATH=. python run_experiments.py --algorithm xgboost --strategy both
        ;;
    3)
        echo -e "${BLUE}Executando TODOS os algoritmos...${NC}"
        PYTHONPATH=. python run_experiments.py --algorithm all --strategy both
        ;;
    *)
        echo -e "${GREEN}[OK] Ambiente preparado! Use os comandos acima quando quiser executar.${NC}"
        ;;
esac
