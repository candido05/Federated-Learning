# Experimento: CATBOOST - CYCLIC

## Configuração
- **Algoritmo**: catboost
- **Estratégia**: cyclic
- **Clientes**: 3
- **Rodadas Globais**: 10
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 69.51s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9219
- **AUC Final**: 0.8276

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9194
- **Revocação Weighted**: 0.9219
- **F1-Score Weighted**: 0.9059

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.8982
- **Revocação Macro**: 0.5582
- **F1-Score Macro**: 0.6458
