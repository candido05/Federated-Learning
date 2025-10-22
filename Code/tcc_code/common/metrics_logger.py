"""
Módulo comum para cálculo de métricas e logging
Compartilhado por todos os algoritmos (XGBoost, LightGBM, CatBoost)
"""

import json
import time
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from flwr.common.logger import log
from logging import INFO


def calculate_comprehensive_metrics(y_true, y_pred_proba, threshold=0.5) -> Dict:
    """
    Calcula métricas abrangentes para classificação binária

    Args:
        y_true: labels verdadeiros
        y_pred_proba: probabilidades preditas
        threshold: limiar para conversão de probabilidade em classe

    Returns:
        Dict com todas as métricas
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calcular métricas básicas
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = 0.5

    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    # Matriz de confusão
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Acurácia
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Especificidade (Taxa de Verdadeiros Negativos)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'specificity': float(specificity),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }

    return metrics


def print_metrics_summary(metrics: Dict, client_id: Optional[int] = None,
                         round_num: Optional[int] = None):
    """
    Imprime um resumo organizado das métricas

    Args:
        metrics: Dicionário com métricas calculadas
        client_id: ID do cliente (opcional)
        round_num: Número do round (opcional)
    """
    prefix = f"[Client {client_id}]" if client_id is not None else "[Server]"
    if round_num is not None:
        prefix += f" Round {round_num}"

    print(f"\n{prefix} Métricas de Performance:")
    print(f"  Acurácia:    {metrics['accuracy']:.4f}")
    print(f"  Precisão:    {metrics['precision']:.4f}")
    print(f"  Revocação:   {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  AUC:         {metrics['auc']:.4f}")

    if 'specificity' in metrics:
        print(f"  Especific.:  {metrics['specificity']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"  Matriz de Confusão:")
    print(f"    TN: {cm['tn']:4d} | FP: {cm['fp']:4d}")
    print(f"    FN: {cm['fn']:4d} | TP: {cm['tp']:4d}")


def save_metrics_to_file(metrics_history, filename="federated_learning_metrics.json"):
    """
    Salva histórico de métricas em arquivo JSON

    Args:
        metrics_history: Dicionário com histórico de métricas
        filename: Nome do arquivo para salvar
    """
    try:
        with open(filename, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f"\nMétricas salvas em: {filename}")
    except Exception as e:
        print(f"Erro ao salvar métricas: {e}")


def print_final_analysis(strategy_name: str, metrics_history=None):
    """
    Imprime análise final dos resultados

    Args:
        strategy_name: Nome da estratégia executada
        metrics_history: Histórico de métricas (opcional)
    """
    print(f"\n{'='*60}")
    print(f"ANÁLISE FINAL - {strategy_name.upper()}")
    print(f"{'='*60}")

    if metrics_history:
        print("Resumo das métricas por rodada:")
        for round_num, metrics in metrics_history.items():
            print(f"\nRodada {round_num}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")

    print(f"{'='*60}\n")


def evaluate_metrics_aggregation(eval_metrics):
    """
    Agrega métricas de avaliação de forma robusta

    Args:
        eval_metrics: Lista de tuplas (num_examples, metrics)

    Returns:
        Dicionário com métricas agregadas
    """
    if not eval_metrics:
        return {
            "auc": 0.5,
            "accuracy": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    total_num = sum([num for num, _ in eval_metrics])
    if total_num == 0:
        return {
            "auc": 0.5,
            "accuracy": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    # Agregação robusta com valores padrão
    metric_sums = {
        "auc": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
    }

    for num, metrics in eval_metrics:
        for metric_name in metric_sums.keys():
            metric_val = metrics.get(metric_name, 0.5 if metric_name in ['auc', 'accuracy'] else 0.0)
            metric_sums[metric_name] += metric_val * num

    # Calcular médias ponderadas
    metrics_aggregated = {}
    for metric_name, total_sum in metric_sums.items():
        metrics_aggregated[metric_name] = total_sum / total_num

    log(INFO, f"Métricas agregadas:")
    for metric_name, value in metrics_aggregated.items():
        log(INFO, f"  {metric_name}: {value:.4f}")

    return metrics_aggregated


class ExperimentLogger:
    """Gerencia logging de experimentos de Federated Learning"""

    def __init__(self, algorithm_name: str, strategy_name: str, num_clients: int,
                 num_rounds: int, num_local_rounds: int, samples_per_client: int):
        """
        Args:
            algorithm_name: Nome do algoritmo (xgboost, lightgbm, catboost)
            strategy_name: Nome da estratégia (cyclic, bagging)
            num_clients: Número de clientes
            num_rounds: Número de rodadas
            num_local_rounds: Número de rodadas locais
            samples_per_client: Amostras por cliente
        """
        self.algorithm_name = algorithm_name
        self.strategy_name = strategy_name
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.num_local_rounds = num_local_rounds
        self.samples_per_client = samples_per_client
        self.start_time = None
        self.metrics_history = {}
        self.round_logs = []

        # Criar estrutura de diretórios com data/hora
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/{algorithm_name}/{timestamp}_{strategy_name}"
        import os
        os.makedirs(self.log_dir, exist_ok=True)

        # Arquivos de log na pasta específica
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
        """
        Loga métricas de uma rodada

        Args:
            round_num: Número da rodada
            metrics: Dicionário com métricas
            source: Fonte das métricas ('server', 'client_X', 'aggregated')
        """
        self.metrics_history[round_num] = metrics

        # Formatar saída como no exemplo output.txt
        log_text = f"\n[{source.upper()}] Round {round_num} Métricas de Performance:\n"
        log_text += f"  Acurácia:    {metrics.get('accuracy', 0):.4f}\n"
        log_text += f"  Precisão:    {metrics.get('precision', 0):.4f}\n"
        log_text += f"  Revocação:   {metrics.get('recall', 0):.4f}\n"
        log_text += f"  F1-Score:    {metrics.get('f1_score', 0):.4f}\n"
        log_text += f"  AUC:         {metrics.get('auc', 0):.4f}\n"

        if 'specificity' in metrics:
            log_text += f"  Especific.:  {metrics['specificity']:.4f}\n"

        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            log_text += f"  Matriz de Confusão:\n"
            log_text += f"    TN: {cm['tn']:4d} | FP: {cm['fp']:4d}\n"
            log_text += f"    FN: {cm['fn']:4d} | TP: {cm['tp']:4d}\n"

        print(log_text)
        self._write_to_file(log_text)

        # Armazenar para histórico
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

        # Criar resumo das métricas por rodada
        summary = f"\n{'='*80}\n"
        summary += "RESUMO DAS MÉTRICAS POR RODADA:\n"
        summary += f"{'='*80}\n\n"

        for round_num in sorted(self.metrics_history.keys(), key=int):
            metrics = self.metrics_history[round_num]
            summary += f"Round {round_num}:\n"
            summary += f"  Acurácia:    {metrics.get('accuracy', 0):.4f}\n"
            summary += f"  Precisão:    {metrics.get('precision', 0):.4f}\n"
            summary += f"  Revocação:   {metrics.get('recall', 0):.4f}\n"
            summary += f"  F1-Score:    {metrics.get('f1_score', 0):.4f}\n"
            summary += f"  AUC:         {metrics.get('auc', 0):.4f}\n"
            if 'specificity' in metrics:
                summary += f"  Especific.:  {metrics['specificity']:.4f}\n"
            summary += "\n"

        print(summary)
        self._write_to_file(summary)

        footer = f"\n{'='*80}\n"
        footer += f"EXPERIMENTO CONCLUÍDO: {self.algorithm_name.upper()} - {self.strategy_name.upper()}\n"
        footer += f"Tempo total: {elapsed_time:.2f} segundos\n"
        footer += f"Total de rodadas executadas: {len(self.metrics_history)}\n"
        footer += f"{'='*80}\n\n"

        # Adicionar resumo do histórico
        if final_history:
            footer += "HISTÓRICO FINAL:\n"
            footer += f"{final_history}\n"
            footer += f"{'='*80}\n\n"

        print(footer)
        self._write_to_file(footer)

        # Salvar JSON completo
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

            # Criar arquivo README na pasta do experimento
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
- **Acurácia Final**: {last_metrics.get('accuracy', 0):.4f}
- **AUC Final**: {last_metrics.get('auc', 0):.4f}
- **F1-Score Final**: {last_metrics.get('f1_score', 0):.4f}
"""

            with open(f"{self.log_dir}/README.md", 'w', encoding='utf-8') as f:
                f.write(readme_content)

            print(f"\n{'='*80}")
            print(f"📁 Logs salvos em: {self.log_dir}")
            print(f"   - execution_log.txt: Log completo da execução")
            print(f"   - metrics.json: Métricas estruturadas")
            print(f"   - README.md: Resumo do experimento")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Erro ao salvar arquivos: {e}")

        return self.metrics_history

    def _write_to_file(self, text: str):
        """Escreve texto no arquivo de log"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            print(f"Aviso: Erro ao escrever no log: {e}")
