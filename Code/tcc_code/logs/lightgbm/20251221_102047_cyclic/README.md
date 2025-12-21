# Experimento: LIGHTGBM - CYCLIC

## Configuração
- **Algoritmo**: lightgbm
- **Estratégia**: cyclic
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 102213
- **Tempo Total**: 1876.62s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.7927
- **AUC Final**: 0.8261

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.8726
- **Revocação Weighted**: 0.7927
- **F1-Score Weighted**: 0.8229

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.5138
- **Revocação Macro**: 0.6693
- **F1-Score Macro**: 0.5540
