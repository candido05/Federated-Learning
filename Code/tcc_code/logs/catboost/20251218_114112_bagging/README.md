# Experimento: CATBOOST - BAGGING

## Configuração
- **Algoritmo**: catboost
- **Estratégia**: bagging
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 95.47s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9231
- **AUC Final**: 0.8279

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9202
- **Revocação Weighted**: 0.9231
- **F1-Score Weighted**: 0.9079

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.8954
- **Revocação Macro**: 0.5664
- **F1-Score Macro**: 0.6542
