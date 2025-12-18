# Experimento: LIGHTGBM - CYCLIC

## Configuração
- **Algoritmo**: lightgbm
- **Estratégia**: cyclic
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 6615.64s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9147
- **AUC Final**: 0.8024

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9033
- **Revocação Weighted**: 0.9147
- **F1-Score Weighted**: 0.9032

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.7762
- **Revocação Macro**: 0.5841
- **F1-Score Macro**: 0.6490
