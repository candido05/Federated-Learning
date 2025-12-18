# Experimento: XGBOOST - CYCLIC

## Configuração
- **Algoritmo**: xgboost
- **Estratégia**: cyclic
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 472.71s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9193
- **AUC Final**: 0.8056

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9105
- **Revocação Weighted**: 0.9193
- **F1-Score Weighted**: 0.9068

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.8225
- **Revocação Macro**: 0.5832
- **F1-Score Macro**: 0.6580
