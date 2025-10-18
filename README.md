# Federated Learning com Otimização SDN

## 📚 Sobre o Projeto

Este repositório contém o **Trabalho de Conclusão de Curso (TCC)** intitulado **"Otimização de Modelos de Aprendizado Federado com SDN (Software-Defined Networking)"**, desenvolvido como requisito para obtenção do grau de Bacharel em Ciência da Computação pela Universidade Federal da Paraíba (UFPB).

### Autor
**Cândido Leandro de Queiroga Bisneto**

### Orientador
**Prof. Fernando Menezes Matos**

---

## 🎯 Objetivos da Pesquisa

O TCC tem como objetivo principal investigar e avaliar o desempenho de **modelos baseados em árvores de decisão** (XGBoost, LightGBM e CatBoost) em ambientes de **Aprendizado Federado**, com foco em:

1. **Implementação de modelos tree-based em FL**: Adaptar XGBoost, LightGBM e CatBoost para funcionarem no framework Flower
2. **Otimização com SDN**: Integrar Software-Defined Networking para otimizar comunicação entre clientes e servidor
3. **Análise de cenários IID vs non-IID**: Comparar desempenho em distribuições de dados homogêneas e heterogêneas
4. **Avaliação de estratégias de agregação**: Testar diferentes algoritmos (FedAvg, FedProx, FedAdam, etc.)
5. **Métricas de desempenho**: Analisar acurácia, tempo de convergência, overhead de comunicação, latência e consumo de banda

---

## 📂 Estrutura do Repositório

```
Federated-Learning/
├── doc/                          # 📄 DOCUMENTAÇÃO DO TCC (PRINCIPAL)
│   ├── main.tex                  # Documento principal LaTeX
│   ├── chapters/                 # Capítulos da monografia
│   │   ├── 01_introducao.tex
│   │   ├── 02_revisao_literatura.tex
│   │   ├── 03_fundamentos_teoricos.tex
│   │   ├── 04_metodologia.tex
│   │   ├── 05_dataset.tex
│   │   ├── 06_config_experimental.tex
│   │   ├── 07_resultados.tex
│   │   ├── 08_discussao.tex
│   │   └── 09_conclusao.tex
│   ├── pre_textual/              # Elementos pré-textuais
│   ├── figures/                  # Imagens e logos
│   ├── tables/                   # Tabelas de resultados
│   ├── tcc.cls                   # Classe LaTeX customizada
│   └── Makefile                  # Compilação automática
│
├── Code/                         # 💻 CÓDIGO-FONTE
│   ├── ml_code/                  # ⭐ BASE PARA IMPLEMENTAÇÃO DO TCC
│   │   ├── models.py             # Wrapper para modelos ML (ESTENDER AQUI!)
│   │   ├── client.py             # Cliente Flower para ML tradicional
│   │   ├── server.py             # Servidor FL com estratégias
│   │   ├── data_loader.py        # Carregamento e particionamento de dados
│   │   ├── visualization.py      # Visualização de resultados
│   │   └── main.py               # Script principal de execução
│   │
│   └── nn_code/                  # Implementação com Redes Neurais (referência)
│       ├── models.py             # ResNet, EfficientNet, MobileNet
│       ├── client.py             # Cliente FL para PyTorch
│       ├── server.py             # Servidor com seleção baseada em VRAM
│       └── main.py               # Execução de experimentos
│
├── CLAUDE.md                     # Guia para Claude Code (instruções de desenvolvimento)
└── README.md                     # Este arquivo
```

---

## 🚀 Implementação Atual vs. Planejada

### ✅ Código Existente (Protótipos)

#### 1. `Code/ml_code/` - **Modelos Tradicionais de ML**
Implementação base com **Flower framework** contendo:
- **Modelos atuais**: Random Forest, SVM, Logistic Regression, KNN, Naive Bayes
- **Dataset**: MNIST (28×28 flattened para 784 features)
- **Estratégias FL**: FedAvg, FedProx, FedAdam, FedAdagrad, FedYogi, FedMedian
- **Funcionalidades**: Serialização pickle, particionamento IID, visualização de resultados

#### 2. `Code/nn_code/` - **Redes Neurais** (Referência)
Implementação com PyTorch:
- **Modelos**: ResNet, EfficientNetV2, MobileNetV3
- **Recurso único**: Seleção de clientes baseada em VRAM disponível
- **Uso**: Apenas como referência, não será usado no TCC

### 🔨 A Ser Implementado para o TCC

#### **Fase 1: Modelos Tree-Based** (PRIORITÁRIO)
Estender `Code/ml_code/models.py` para incluir:
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
```

**Tarefas**:
- [ ] Adicionar XGBoost ao `_create_model()` em `models.py`
- [ ] Adicionar LightGBM ao `_create_model()` em `models.py`
- [ ] Adicionar CatBoost ao `_create_model()` em `models.py`
- [ ] Testar serialização pickle para cada modelo
- [ ] Validar funcionamento em ambiente FL

#### **Fase 2: Dataset Tabular**
**Requisito**: Capítulo 5 do TCC deve ser completado primeiro

**Tarefas**:
- [ ] Selecionar dataset tabular adequado (UCI, Kaggle)
- [ ] Documentar metadados no Capítulo 5
- [ ] Implementar carregamento em `data_loader.py`
- [ ] Criar particionamento non-IID (Dirichlet)
- [ ] Validar distribuição IID vs non-IID

#### **Fase 3: Integração SDN**
**Objetivo**: Otimizar comunicação FL com políticas de QoS

**Tarefas**:
- [ ] Definir controlador SDN (OpenFlow)
- [ ] Implementar monitoramento de rede (latência, throughput)
- [ ] Criar políticas de priorização de tráfego FL
- [ ] Comparar FL com/sem SDN
- [ ] Coletar métricas de overhead de comunicação

#### **Fase 4: Experimentos e Análise**
**Tarefas**:
- [ ] Executar grid search de hiperparâmetros
- [ ] Rodar experimentos: {XGBoost, LightGBM, CatBoost} × {FedAvg, FedProx, FedAdam} × {IID, non-IID}
- [ ] Coletar métricas: accuracy, precision, recall, F1, AUC-ROC
- [ ] Análise estatística (Friedman + Nemenyi)
- [ ] Gerar gráficos de convergência
- [ ] Preencher Capítulo 7 (Resultados)

---

## 🛠️ Como Executar

### Pré-requisitos
```bash
# Python 3.9+
pip install flwr>=1.6.0
pip install xgboost>=2.0.0      # A INSTALAR
pip install lightgbm>=4.0.0     # A INSTALAR
pip install catboost>=1.2.0     # A INSTALAR
pip install scikit-learn>=1.3.0
pip install numpy pandas matplotlib seaborn
```

### Executar Protótipo Atual (ML Tradicional)
```bash
cd Code/ml_code
python main.py

# Menu interativo:
# 1. Todos os experimentos (todos modelos × todas estratégias)
# 2. Experimentos específicos (editar main.py linhas 159-161)
# 3. Experimento único
```

### Configuração de Experimentos
Editar `Code/ml_code/main.py`:
```python
NUM_PARTITIONS = 10   # Número de clientes FL
NUM_ROUNDS = 20       # Rodadas de treinamento
BATCH_SIZE = 2        # Tamanho do batch local

# Para TCC (após implementar tree-based models):
selected_models = ["xgboost", "lightgbm", "catboost"]
selected_strategies = ["FedAvg", "FedProx", "FedAdam"]
```

---

## 📖 Documentação do TCC

### Compilar LaTeX
```bash
cd doc
latexmk -pdf main.tex    # Compilação completa com auto-referências
make                     # Alternativa com Makefile
make watch               # Auto-recompilação ao salvar arquivos
```

### Status de Preenchimento
O documento possui **~180 placeholders** `<<...>>` a serem preenchidos. Progresso por capítulo:

| Capítulo | Status | Placeholders | Prioridade |
|----------|--------|--------------|------------|
| 1. Introdução | 📝 Rascunho | ~15 | Média |
| 2. Revisão de Literatura | 📝 Rascunho | ~20 | Alta |
| 3. Fundamentos Teóricos | 📝 Rascunho | ~25 | Alta |
| 4. Metodologia | 📝 Rascunho | ~25 | Alta |
| 5. Dataset | ⚠️ Pendente | ~50 | **CRÍTICA** |
| 6. Configuração Experimental | 📝 Rascunho | ~20 | Alta |
| 7. Resultados | ⚠️ Pendente | ~30 | **CRÍTICA** |
| 8. Discussão | 📝 Rascunho | ~10 | Média |
| 9. Conclusão | 📝 Rascunho | ~5 | Baixa |

**Legenda**: ⚠️ Pendente de dados experimentais | 📝 Estrutura criada

### Encontrar Placeholders
```bash
cd doc
grep -r "<<" chapters/          # Listar todos os placeholders
grep "<<" chapters/05_dataset.tex | wc -l  # Contar no Capítulo 5
```

---

## 🔬 Metodologia de Pesquisa

### Frameworks e Tecnologias
- **Federated Learning**: [Flower](https://flower.dev/) 1.6+
- **Modelos**: XGBoost, LightGBM, CatBoost
- **Linguagem**: Python 3.9+
- **SDN**: A definir (OpenFlow/Ryu/ONOS)
- **Comunicação**: gRPC (padrão do Flower)

### Cenários de Teste
1. **IID (Independent and Identically Distributed)**: Dados distribuídos uniformemente entre clientes
2. **Non-IID**: Distribuição heterogênea usando Dirichlet (α = 0.5)

### Métricas de Avaliação
- **Desempenho do Modelo**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Convergência**: Número de rodadas até atingir acurácia alvo
- **Comunicação**: Bytes transmitidos por rodada, overhead total
- **Rede**: Latência, throughput, packet loss (com SDN)

---

## 📅 Cronograma

| Fase | Descrição | Status |
|------|-----------|--------|
| ✅ 1 | Estrutura do TCC e revisão bibliográfica | Concluído |
| ✅ 2 | Protótipo com modelos tradicionais ML | Concluído |
| 🔄 3 | Implementação de XGBoost/LightGBM/CatBoost | Em andamento |
| ⏳ 4 | Seleção e preparação do dataset tabular | Pendente |
| ⏳ 5 | Integração com SDN | Pendente |
| ⏳ 6 | Execução de experimentos | Pendente |
| ⏳ 7 | Análise de resultados e escrita final | Pendente |
| ⏳ 8 | Defesa do TCC | Dezembro/2025 |

---

## 📝 Licença

Este projeto é desenvolvido para fins acadêmicos como parte do Trabalho de Conclusão de Curso do Centro de Informática da UFPB.

---

## 📧 Contato

Para dúvidas sobre o projeto:
- **Autor**: Cândido Leandro de Queiroga Bisneto
- **Instituição**: Centro de Informática - UFPB
- **Orientador**: Prof. Fernando Menezes Matos

---

## 🙏 Agradecimentos

- Centro de Informática da UFPB
- Flower Framework Community
- Comunidades de código aberto XGBoost, LightGBM e CatBoost

---

**Última atualização**: Outubro/2025
