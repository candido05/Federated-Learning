# Experimento: CATBOOST - BAGGING

## Configuração
- **Algoritmo**: catboost
- **Estratégia**: bagging
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 138.70s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9228
- **AUC Final**: 0.8274

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9197
- **Revocação Weighted**: 0.9228
- **F1-Score Weighted**: 0.9076

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.8929
- **Revocação Macro**: 0.5658
- **F1-Score Macro**: 0.6532
