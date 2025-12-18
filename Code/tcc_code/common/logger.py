"""
Sistema de logging para experimentos de Federated Learning
Gerencia logs de experimentos, métricas por rodada e salvamento de resultados
"""

import json
import time
import os
from typing import Dict, Optional


class ExperimentLogger:
    """Gerencia logging de experimentos de Federated Learning"""

    def __init__(self, algorithm_name: str, strategy_name: str, num_clients: int,
                 num_rounds: int, num_local_rounds: int, samples_per_client: int):
        self.algorithm_name = algorithm_name
        self.strategy_name = strategy_name
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.num_local_rounds = num_local_rounds
        self.samples_per_client = samples_per_client
        self.start_time = None
        self.metrics_history = {}
        self.round_logs = []

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/{algorithm_name}/{timestamp}_{strategy_name}"
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = f"{self.log_dir}/execution_log.txt"
        self.json_file = f"{self.log_dir}/metrics.json"

    def start_experiment(self):
        """Inicia experimento e marca tempo de início"""
        self.start_time = time.time()

        header = f"\n{'='*80}\n"
        header += f"INICIANDO EXPERIMENTO: {self.algorithm_name.upper()} - {self.strategy_name.upper()}\n"
        header += f"{'='*80}\n"
        header += f"Configuração:\n"
        header += f"  - Algoritmo: {self.algorithm_name}\n"
        header += f"  - Estratégia: {self.strategy_name}\n"
        header += f"  - Número de clientes: {self.num_clients}\n"
        header += f"  - Rodadas globais: {self.num_rounds}\n"
        header += f"  - Rodadas locais: {self.num_local_rounds}\n"
        header += f"  - Amostras por cliente: {self.samples_per_client}\n"
        header += f"{'='*80}\n\n"

        print(header)
        self._write_to_file(header)

    def log_round_metrics(self, round_num: int, metrics: Dict, source: str = "server"):
        """Loga métricas de uma rodada (3 classes)"""
        self.metrics_history[round_num] = metrics

        log_text = f"\n[{source.upper()}] Round {round_num} Métricas de Performance:\n"
        log_text += f"  Acurácia:           {metrics.get('accuracy', 0):.4f}\n"
        log_text += f"  Balanced Accuracy:  {metrics.get('balanced_accuracy', 0):.4f}  [PRIORIDADE]\n"
        log_text += f"  MCC:                {metrics.get('mcc', 0):.4f}\n"
        log_text += f"  AUC:                {metrics.get('auc', 0):.4f} (macro avg)\n"

        log_text += f"\n  Métricas Macro (média não-ponderada - PRIORIZAR!):\n"
        log_text += f"    Precisão:  {metrics.get('precision_macro', 0):.4f}\n"
        log_text += f"    Revocação: {metrics.get('recall_macro', 0):.4f}  [METRICA PRINCIPAL]\n"
        log_text += f"    F1-Score:  {metrics.get('f1_score_macro', 0):.4f}  [METRICA PRINCIPAL]\n"

        log_text += f"\n  Recall por Classe (detectar problemas em classes minoritárias):\n"
        log_text += f"    Classe 0 (minoritária): {metrics.get('recall_class_0', 0):.4f}\n"
        log_text += f"    Classe 1 (majoritária): {metrics.get('recall_class_1', 0):.4f}\n"
        log_text += f"    Classe 2 (minoritária): {metrics.get('recall_class_2', 0):.4f}\n"

        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            if isinstance(cm, list):
                log_text += f"\n  Matriz de Confusão (3 classes):\n"
                for i, row in enumerate(cm):
                    log_text += f"    Classe {i}: {row}\n"

        print(log_text)
        self._write_to_file(log_text)

        self.round_logs.append({
            "round": round_num,
            "source": source,
            "metrics": metrics,
            "timestamp": time.time() - self.start_time if self.start_time else 0
        })

    def log_aggregated_metrics(self, round_num: int, metrics: Dict):
        """Loga métricas agregadas de múltiplos clientes"""
        log_text = f"\n[AGREGADO] Round {round_num} - Métricas Agregadas:\n"
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                log_text += f"  {metric_name}: {value:.4f}\n"

        print(log_text)
        self._write_to_file(log_text)

    def end_experiment(self, final_history=None):
        """Finaliza experimento e salva métricas"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        summary = f"\n{'='*80}\n"
        summary += "RESUMO DAS MÉTRICAS POR RODADA:\n"
        summary += f"{'='*80}\n\n"

        for round_num in sorted(self.metrics_history.keys(), key=int):
            metrics = self.metrics_history[round_num]

            summary += f"Round {round_num}:\n"
            summary += f"  Acurácia:    {metrics.get('accuracy', 0):.4f}\n"
            summary += f"  AUC:         {metrics.get('auc', 0):.4f}\n"
            summary += f"  [Weighted] Precisão:  {metrics.get('precision_weighted', metrics.get('precision', 0)):.4f}\n"
            summary += f"  [Weighted] Revocação: {metrics.get('recall_weighted', metrics.get('recall', 0)):.4f}\n"
            summary += f"  [Weighted] F1-Score:  {metrics.get('f1_score_weighted', metrics.get('f1_score', 0)):.4f}\n"
            summary += f"  [Macro] Precisão:  {metrics.get('precision_macro', 0):.4f}\n"
            summary += f"  [Macro] Revocação: {metrics.get('recall_macro', 0):.4f}\n"
            summary += f"  [Macro] F1-Score:  {metrics.get('f1_score_macro', 0):.4f}\n"
            summary += "\n"

        print(summary)
        self._write_to_file(summary)

        footer = f"\n{'='*80}\n"
        footer += f"EXPERIMENTO CONCLUÍDO: {self.algorithm_name.upper()} - {self.strategy_name.upper()}\n"
        footer += f"Tempo total: {elapsed_time:.2f} segundos\n"
        footer += f"Total de rodadas executadas: {len(self.metrics_history)}\n"
        footer += f"{'='*80}\n\n"

        if final_history:
            footer += "HISTÓRICO FINAL:\n"
            footer += f"{final_history}\n"
            footer += f"{'='*80}\n\n"

        print(footer)
        self._write_to_file(footer)

        complete_data = {
            "experiment_info": {
                "algorithm": self.algorithm_name,
                "strategy": self.strategy_name,
                "num_clients": self.num_clients,
                "num_rounds": self.num_rounds,
                "num_local_rounds": self.num_local_rounds,
                "samples_per_client": self.samples_per_client,
                "total_time_seconds": elapsed_time,
                "log_directory": self.log_dir
            },
            "metrics_by_round": self.metrics_history,
            "detailed_logs": self.round_logs,
            "final_history": str(final_history) if final_history else None
        }

        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(complete_data, f, indent=2, ensure_ascii=False)

            self._create_readme(elapsed_time)

            print(f"\n{'='*80}")
            print(f"[LOGS] Logs salvos em: {self.log_dir}")
            print(f"   - execution_log.txt: Log completo da execução")
            print(f"   - metrics.json: Métricas estruturadas")
            print(f"   - README.md: Resumo do experimento")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Erro ao salvar arquivos: {e}")

        return self.metrics_history

    def _create_readme(self, elapsed_time: float):
        """Cria arquivo README do experimento"""
        readme_content = f"""# Experimento: {self.algorithm_name.upper()} - {self.strategy_name.upper()}

## Configuração
- **Algoritmo**: {self.algorithm_name}
- **Estratégia**: {self.strategy_name}
- **Clientes**: {self.num_clients}
- **Rodadas Globais**: {self.num_rounds}
- **Rodadas Locais**: {self.num_local_rounds}
- **Amostras/Cliente**: {self.samples_per_client}
- **Tempo Total**: {elapsed_time:.2f}s

## Arquivos
- `execution_log.txt`: Log completo da execução
- `metrics.json`: Métricas estruturadas em JSON
- `README.md`: Este arquivo

## Resultados Finais
"""
        if self.metrics_history:
            last_round = max(self.metrics_history.keys(), key=int)
            last_metrics = self.metrics_history[last_round]

            readme_content += f"""
### Métricas Gerais
- **Acurácia Final**: {last_metrics.get('accuracy', 0):.4f}
- **AUC Final**: {last_metrics.get('auc', 0):.4f}

### Métricas Weighted (Ponderadas por Suporte)
- **Precisão Weighted**: {last_metrics.get('precision_weighted', last_metrics.get('precision', 0)):.4f}
- **Revocação Weighted**: {last_metrics.get('recall_weighted', last_metrics.get('recall', 0)):.4f}
- **F1-Score Weighted**: {last_metrics.get('f1_score_weighted', last_metrics.get('f1_score', 0)):.4f}

### Métricas Macro (Não-Ponderadas)
- **Precisão Macro**: {last_metrics.get('precision_macro', 0):.4f}
- **Revocação Macro**: {last_metrics.get('recall_macro', 0):.4f}
- **F1-Score Macro**: {last_metrics.get('f1_score_macro', 0):.4f}
"""

        with open(f"{self.log_dir}/README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)

    def _write_to_file(self, text: str):
        """Escreve texto no arquivo de log"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            print(f"Aviso: Erro ao escrever no log: {e}")
