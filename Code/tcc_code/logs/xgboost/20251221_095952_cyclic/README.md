# Experimento: XGBOOST - CYCLIC

## Configuração
- **Algoritmo**: xgboost
- **Estratégia**: cyclic
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 102213
- **Tempo Total**: 590.00s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.7761
- **AUC Final**: 0.8298

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.8716
- **Revocação Weighted**: 0.7761
- **F1-Score Weighted**: 0.8114

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.5035
- **Revocação Macro**: 0.6724
- **F1-Score Macro**: 0.5423
