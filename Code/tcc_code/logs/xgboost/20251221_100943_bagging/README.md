# Experimento: XGBOOST - BAGGING

## Configuração
- **Algoritmo**: xgboost
- **Estratégia**: bagging
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 102213
- **Tempo Total**: 663.90s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.7813
- **AUC Final**: 0.8296

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.8718
- **Revocação Weighted**: 0.7813
- **F1-Score Weighted**: 0.8151

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.5075
- **Revocação Macro**: 0.6700
- **F1-Score Macro**: 0.5461
