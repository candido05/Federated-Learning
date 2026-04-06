#!/bin/bash
# ============================================================
# test_distributed.sh - Teste distribuído: rodar em VMs separadas
# ============================================================
# Este script é para referência. Você executa CADA comando
# em uma VM/terminal diferente.
#
# Pré-requisitos:
#   - Todas as VMs com Python + dependências instaladas
#   - Todas as VMs com o projeto copiado
#   - Conectividade TCP entre as VMs (testar com: nc -zv <IP> 8080)
# ============================================================

echo "============================================"
echo "  INSTRUÇÕES - Teste Distribuído"
echo "============================================"
echo ""
echo "PASSO 1: No terminal da VM SERVIDOR (ex: 10.0.0.10):"
echo "  python server.py --address 0.0.0.0:8080 --rounds 5 --method cyclic --min-clients 2"
echo ""
echo "PASSO 2: No terminal da VM CLIENTE 1 (ex: 10.0.0.11):"
echo "  python client.py --server 10.0.0.10:8080 --partition-id 0 --num-partitions 2"
echo ""
echo "PASSO 3: No terminal da VM CLIENTE 2 (ex: 10.0.0.12):"
echo "  python client.py --server 10.0.0.10:8080 --partition-id 1 --num-partitions 2"
echo ""
echo "IMPORTANTE: Ajuste os IPs para os da sua rede!"
echo "============================================"