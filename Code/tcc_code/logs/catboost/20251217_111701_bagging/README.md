# Experimento: CATBOOST - BAGGING

## Configuração
- **Algoritmo**: catboost
- **Estratégia**: bagging
- **Clientes**: 3
- **Rodadas Globais**: 10
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 44.74s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9214
- **AUC Final**: 0.8275

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9181
- **Revocação Weighted**: 0.9214
- **F1-Score Weighted**: 0.9053

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.8913
- **Revocação Macro**: 0.5568
- **F1-Score Macro**: 0.6435
