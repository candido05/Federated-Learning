# Experimento: XGBOOST - BAGGING

## Configuração
- **Algoritmo**: xgboost
- **Estratégia**: bagging
- **Clientes**: 3
- **Rodadas Globais**: 10
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 54.35s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9249
- **AUC Final**: 0.8171

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9204
- **Revocação Weighted**: 0.9249
- **F1-Score Weighted**: 0.9116

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.8791
- **Revocação Macro**: 0.5854
- **F1-Score Macro**: 0.6706
