# 🚀 Quick Start Guide

## Passo 1: Ativar ambiente virtual

```bash
cd Code/tcc_code

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

## Passo 2: Verificar instalação

```bash
python -c "import flwr, xgboost; print('✓ Dependências OK')"
```

Se houver erro, instale as dependências:

```bash
pip install -r requirements.txt
```

## Passo 3: Executar experimento teste

```bash
# Teste rápido (3 clientes, 3 rodadas, poucos dados)
python run_experiments.py \
    --algorithm xgboost \
    --strategy cyclic \
    --num-clients 3 \
    --num-rounds 3 \
    --samples 2000
```

## Passo 4: Executar experimento completo

```bash
# Experimento completo (configuração padrão do TCC)
python run_experiments.py \
    --algorithm xgboost \
    --strategy both \
    --num-clients 6 \
    --num-rounds 6 \
    --local-rounds 20 \
    --samples 8000
```

## Resultados

Após a execução:
- ✅ Métricas em tempo real no console
- ✅ Arquivo JSON: `federated_learning_results.json`

## Troubleshooting Rápido

### Erro de imports
```bash
pip install --upgrade -r requirements.txt
```

### Dataset não baixa
- Certifique-se de ter internet
- Aguarde (~1GB na primeira vez)

### Out of memory
```bash
# Reduza os parâmetros
python run_experiments.py --num-clients 3 --samples 1000
```

---

**✨ Para mais detalhes, veja [README.md](README.md)**
