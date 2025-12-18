# Experimento: XGBOOST - BAGGING

## Configuração
- **Algoritmo**: xgboost
- **Estratégia**: bagging
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 520.20s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9176
- **AUC Final**: 0.8043

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9075
- **Revocação Weighted**: 0.9176
- **F1-Score Weighted**: 0.9054

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.8019
- **Revocação Macro**: 0.5827
- **F1-Score Macro**: 0.6534
