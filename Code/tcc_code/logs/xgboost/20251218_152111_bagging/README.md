# Experimento: XGBOOST - BAGGING

## Configuração
- **Algoritmo**: xgboost
- **Estratégia**: bagging
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 483.07s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9193
- **AUC Final**: 0.8007

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9101
- **Revocação Weighted**: 0.9193
- **F1-Score Weighted**: 0.9073

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.8145
- **Revocação Macro**: 0.5876
- **F1-Score Macro**: 0.6604
