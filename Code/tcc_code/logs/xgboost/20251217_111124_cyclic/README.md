# Experimento: XGBOOST - CYCLIC

## Configuração
- **Algoritmo**: xgboost
- **Estratégia**: cyclic
- **Clientes**: 3
- **Rodadas Globais**: 10
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 45.35s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9256
- **AUC Final**: 0.8186

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9208
- **Revocação Weighted**: 0.9256
- **F1-Score Weighted**: 0.9130

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.8753
- **Revocação Macro**: 0.5930
- **F1-Score Macro**: 0.6773
