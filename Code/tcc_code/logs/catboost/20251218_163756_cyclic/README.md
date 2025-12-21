# Experimento: CATBOOST - CYCLIC

## Configuração
- **Algoritmo**: catboost
- **Estratégia**: cyclic
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 59015.11s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9229
- **AUC Final**: 0.8275

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9199
- **Revocação Weighted**: 0.9229
- **F1-Score Weighted**: 0.9078

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.8937
- **Revocação Macro**: 0.5663
- **F1-Score Macro**: 0.6539
