# Experimento: LIGHTGBM - BAGGING

## Configuração
- **Algoritmo**: lightgbm
- **Estratégia**: bagging
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 102213
- **Tempo Total**: 1806.23s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.7901
- **AUC Final**: 0.8260

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.8724
- **Revocação Weighted**: 0.7901
- **F1-Score Weighted**: 0.8209

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.5108
- **Revocação Macro**: 0.6720
- **F1-Score Macro**: 0.5520
