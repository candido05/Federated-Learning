# Experimento: XGBOOST - CYCLIC

## Configuração
- **Algoritmo**: xgboost
- **Estratégia**: cyclic
- **Clientes**: 3
- **Rodadas Globais**: 50
- **Rodadas Locais**: 50
- **Amostras/Cliente**: 38409
- **Tempo Total**: 615.55s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais

### Métricas Gerais
- **Acurácia Final**: 0.9179
- **AUC Final**: 0.8040

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: 0.9081
- **Revocação Weighted**: 0.9179
- **F1-Score Weighted**: 0.9059

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: 0.8059
- **Revocação Macro**: 0.5850
- **F1-Score Macro**: 0.6562
