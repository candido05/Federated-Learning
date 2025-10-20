# 🚀 Guia de Execução no WSL

## ✅ Pré-requisitos
- WSL instalado e configurado
- Ambiente virtual Python já criado
- Estar no diretório do projeto

## 📍 Passo 1: Navegar para o diretório (no WSL)

```bash
cd /mnt/c/Users/candi/OneDrive/Desktop/Federated-Learning/Code/tcc_code
```

## 🔧 Passo 2: Ativar ambiente virtual

```bash
# Se o ambiente virtual está em venv/
source venv/bin/activate

# OU se está em outro lugar (ajuste o caminho)
source /caminho/para/seu/venv/bin/activate
```

## 📦 Passo 3: Verificar/Instalar dependências

```bash
# Verificar se as dependências estão instaladas
pip list | grep -E "flwr|xgboost|lightgbm|catboost"

# Se necessário, instalar:
pip install -r requirements.txt
```

## ▶️ Passo 4: Executar experimentos

### Opção 1: Teste rápido (recomendado para primeira execução)

```bash
# Teste com XGBoost apenas (rápido, poucos dados)
PYTHONPATH=. python run_experiments.py \
    --algorithm xgboost \
    --strategy cyclic \
    --num-clients 3 \
    --num-rounds 3 \
    --samples 2000
```

### Opção 2: Experimento com um algoritmo

```bash
# XGBoost
PYTHONPATH=. python run_experiments.py --algorithm xgboost --strategy cyclic

# LightGBM
PYTHONPATH=. python run_experiments.py --algorithm lightgbm --strategy cyclic

# CatBoost
PYTHONPATH=. python run_experiments.py --algorithm catboost --strategy cyclic
```

### Opção 3: Experimento completo (ambas estratégias)

```bash
# Um algoritmo com ambas estratégias
PYTHONPATH=. python run_experiments.py --algorithm xgboost --strategy both
```

### Opção 4: TODOS os algoritmos

```bash
# Executar os 3 algoritmos com ambas estratégias (6 experimentos!)
PYTHONPATH=. python run_experiments.py --algorithm all --strategy both
```

### Opção 5: Configuração customizada

```bash
# Experimento customizado
PYTHONPATH=. python run_experiments.py \
    --algorithm xgboost \
    --strategy cyclic \
    --num-clients 10 \
    --num-rounds 10 \
    --local-rounds 30 \
    --samples 5000 \
    --seed 42
```

## 📊 Opções disponíveis

```
--algorithm {xgboost,lightgbm,catboost,all}
    Qual algoritmo executar (padrão: xgboost)

--strategy {cyclic,bagging,both}
    Estratégia de agregação (padrão: cyclic)

--num-clients NUM
    Número de clientes FL (padrão: 6)

--num-rounds NUM
    Rodadas do servidor (padrão: 6)

--local-rounds NUM
    Rodadas locais de boosting (padrão: 20)

--samples NUM
    Amostras por cliente (padrão: 8000)

--seed NUM
    Random seed (padrão: 42)
```

## 📈 Verificar resultados

```bash
# Ver arquivo de resultados gerado
cat federated_learning_results.json | jq .

# Ou sem jq:
cat federated_learning_results.json
```

## 🐛 Troubleshooting

### Erro: "ModuleNotFoundError"
```bash
# Certifique-se de usar PYTHONPATH=.
PYTHONPATH=. python run_experiments.py ...
```

### Erro: "ImportError: flwr"
```bash
# Instalar dependências
pip install -r requirements.txt
```

### Erro: "CUDA out of memory"
```bash
# Reduzir recursos
PYTHONPATH=. python run_experiments.py \
    --num-clients 3 \
    --samples 2000
```

### Dataset HIGGS demora muito para baixar
```bash
# Na primeira execução, o dataset (~1GB) será baixado
# Aguarde a conclusão, nas próximas será mais rápido
```

## 💡 Dicas

1. **Primeira execução**: Use teste rápido
   ```bash
   PYTHONPATH=. python run_experiments.py --algorithm xgboost --num-clients 3 --num-rounds 3 --samples 2000
   ```

2. **Monitorar execução**: Use `tee` para salvar logs
   ```bash
   PYTHONPATH=. python run_experiments.py --algorithm xgboost | tee experiment_log.txt
   ```

3. **Background**: Execute em background
   ```bash
   nohup PYTHONPATH=. python run_experiments.py --algorithm all --strategy both > output.log 2>&1 &
   ```

4. **Verificar progresso** (se em background):
   ```bash
   tail -f output.log
   ```

## 📝 Exemplo de execução completa

```bash
# 1. Ir para o diretório
cd /mnt/c/Users/candi/OneDrive/Desktop/Federated-Learning/Code/tcc_code

# 2. Ativar ambiente
source venv/bin/activate

# 3. Verificar instalação
python -c "import flwr, xgboost, lightgbm, catboost; print('✓ OK')"

# 4. Executar teste rápido
PYTHONPATH=. python run_experiments.py \
    --algorithm xgboost \
    --strategy cyclic \
    --num-clients 3 \
    --num-rounds 3 \
    --samples 2000

# 5. Se deu certo, executar experimento completo
PYTHONPATH=. python run_experiments.py --algorithm all --strategy both
```

## ✅ Comandos rápidos (copiar e colar)

### Setup inicial
```bash
cd /mnt/c/Users/candi/OneDrive/Desktop/Federated-Learning/Code/tcc_code
source venv/bin/activate
pip install -r requirements.txt
```

### Teste rápido
```bash
PYTHONPATH=. python run_experiments.py --algorithm xgboost --num-clients 3 --num-rounds 3 --samples 2000
```

### Experimento padrão
```bash
PYTHONPATH=. python run_experiments.py --algorithm xgboost --strategy both
```

### Experimento completo (todos algoritmos)
```bash
PYTHONPATH=. python run_experiments.py --algorithm all --strategy both
```

---

**Importante**: Sempre use `PYTHONPATH=.` antes do comando para garantir que os imports funcionem corretamente!
