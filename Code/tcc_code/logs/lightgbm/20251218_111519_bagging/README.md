# Experimento: LIGHTGBM - BAGGING

## Configuração
- **Algoritmo**: lightgbm
- **Estratégia**: bagging
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 1441.66s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9153
- **AUC Final**: 0.8067

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9043
- **Revocação Weighted**: 0.9153
- **F1-Score Weighted**: 0.9036

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.7852
- **Revocação Macro**: 0.5829
- **F1-Score Macro**: 0.6501
