# Experimento: LIGHTGBM - BAGGING

## Configuração
- **Algoritmo**: lightgbm
- **Estratégia**: bagging
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 2957.09s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9157
- **AUC Final**: 0.8023

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9048
- **Revocação Weighted**: 0.9157
- **F1-Score Weighted**: 0.9041

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.7850
- **Revocação Macro**: 0.5840
- **F1-Score Macro**: 0.6509
