#!/bin/bash
################################################################################
# Script de Limpeza de Cache Python
# Parte do TODO 1.6 - FASE 1 CRÍTICA
################################################################################

echo "=========================================="
echo "Limpando cache Python (.pyc e __pycache__)"
echo "=========================================="
echo ""

# Conta arquivos antes
PYCACHE_DIRS=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
PYC_FILES=$(find . -name "*.pyc" 2>/dev/null | wc -l)

echo "Encontrados:"
echo "  - $PYCACHE_DIRS diretórios __pycache__"
echo "  - $PYC_FILES arquivos .pyc"
echo ""

# Remove diretórios __pycache__
echo "Removendo diretórios __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove arquivos .pyc
echo "Removendo arquivos .pyc..."
find . -name "*.pyc" -delete 2>/dev/null

# Remove arquivos .pyo (Python Optimized)
echo "Removendo arquivos .pyo..."
find . -name "*.pyo" -delete 2>/dev/null

# Remove diretórios .pytest_cache se existirem
echo "Removendo .pytest_cache (se existir)..."
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

echo ""
echo "=========================================="
echo "✅ CACHE LIMPO COM SUCESSO!"
echo "=========================================="
echo ""
echo "Próximo passo:"
echo "  python main.py --algorithm all --strategy both"
echo ""
