"""Módulo de logging estruturado para experimentos de Aprendizado Federado.

Este módulo fornece funcionalidades para logging de experimentos FL,
incluindo logs por rodada, por cliente e resumos globais.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import MetricsCalculator


class ExperimentLogger:
    """Logger estruturado para experimentos de Aprendizado Federado.

    Gerencia a estrutura de diretórios e logs para experimentos FL,
    organizando logs por modelo, estratégia e timestamp.
    """

    def __init__(
        self,
        model_name: str,
        strategy_name: str,
        config: Any,
        base_log_dir: str = "logs",
    ) -> None:
        """Inicializa o logger de experimento.

        Args:
            model_name: Nome do modelo (ex: 'xgboost', 'lightgbm', 'catboost').
            strategy_name: Nome da estratégia FL (ex: 'FedAvg', 'FedProx').
            config: Objeto de configuração do experimento (GlobalConfig).
            base_log_dir: Diretório base para logs (padrão: 'logs').
        """
        self.model_name = model_name
        self.strategy_name = strategy_name
        self.config = config
        self.base_log_dir = Path(base_log_dir)

        # Timestamp para este experimento
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Estrutura de diretórios
        self.experiment_dir: Optional[Path] = None
        self.rounds_dir: Optional[Path] = None
        self.clients_dir: Optional[Path] = None

        # Armazenamento em memória de logs
        self.rounds_logs: List[Dict[str, Any]] = []
        self.clients_logs: Dict[str, List[Dict[str, Any]]] = {}

        # Logger Python
        self.logger = logging.getLogger(f"FL.{model_name}.{strategy_name}")

    def setup_directories(self) -> None:
        """Cria a estrutura de diretórios para logs.

        Estrutura criada:
        logs/
        └── {model_name}/
            └── {strategy_name}/
                └── {timestamp}/
                    ├── rounds/
                    └── clients/
        """
        # Diretório do experimento
        self.experiment_dir = (
            self.base_log_dir / self.model_name / self.strategy_name / self.timestamp
        )

        # Subdiretórios
        self.rounds_dir = self.experiment_dir / "rounds"
        self.clients_dir = self.experiment_dir / "clients"

        # Cria todos os diretórios
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.rounds_dir.mkdir(exist_ok=True)
        self.clients_dir.mkdir(exist_ok=True)

        self.logger.info(f"Diretórios de log criados em: {self.experiment_dir}")

    def log_round(
        self,
        round_num: int,
        metrics: Dict[str, Any],
        clients_trained: Optional[List[str]] = None,
        training_time: Optional[float] = None,
    ) -> None:
        """Registra métricas e informações de uma rodada de treinamento.

        Args:
            round_num: Número da rodada de treinamento.
            metrics: Dicionário de métricas da rodada.
            clients_trained: Lista de IDs dos clientes que treinaram nesta rodada.
            training_time: Tempo de treinamento da rodada em segundos.
        """
        if self.rounds_dir is None:
            raise RuntimeError("Diretórios não inicializados. Chame setup_directories() primeiro.")

        # Prepara dados do log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "round": round_num,
            "metrics": MetricsCalculator.metrics_to_json(metrics),
        }

        if clients_trained is not None:
            log_entry["clients_trained"] = clients_trained
            log_entry["num_clients"] = len(clients_trained)

        if training_time is not None:
            log_entry["training_time_seconds"] = float(training_time)

        # Armazena em memória
        self.rounds_logs.append(log_entry)

        # Salva em arquivo JSON individual
        round_file = self.rounds_dir / f"round_{round_num:03d}.json"
        with open(round_file, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"Rodada {round_num} registrada: "
            f"Acurácia={metrics.get('accuracy', 0):.4f}, "
            f"F1={metrics.get('f1_score', 0):.4f}"
        )

    def log_client(
        self,
        client_id: str,
        metrics: Dict[str, Any],
        round_num: Optional[int] = None,
    ) -> None:
        """Registra métricas de um cliente individual.

        Args:
            client_id: Identificador do cliente.
            metrics: Dicionário de métricas do cliente.
            round_num: Número da rodada (opcional).
        """
        if self.clients_dir is None:
            raise RuntimeError("Diretórios não inicializados. Chame setup_directories() primeiro.")

        # Prepara dados do log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "metrics": MetricsCalculator.metrics_to_json(metrics),
        }

        if round_num is not None:
            log_entry["round"] = round_num

        # Armazena em memória
        if client_id not in self.clients_logs:
            self.clients_logs[client_id] = []
        self.clients_logs[client_id].append(log_entry)

        # Salva em arquivo JSON do cliente
        client_file = self.clients_dir / f"client_{client_id}.json"

        # Carrega logs existentes ou cria novo
        if client_file.exists():
            with open(client_file, "r", encoding="utf-8") as f:
                existing_logs = json.load(f)
        else:
            existing_logs = []

        existing_logs.append(log_entry)

        # Salva logs atualizados
        with open(client_file, "w", encoding="utf-8") as f:
            json.dump(existing_logs, f, indent=2, ensure_ascii=False)

        self.logger.debug(f"Cliente {client_id} registrado")

    def log_server_evaluation(
        self,
        round_num: int,
        metrics: Dict[str, Any],
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra avaliação do servidor (modelo global).

        Args:
            round_num: Número da rodada de avaliação.
            metrics: Métricas de avaliação do modelo global.
            additional_info: Informações adicionais opcionais.
        """
        if self.experiment_dir is None:
            raise RuntimeError("Diretórios não inicializados. Chame setup_directories() primeiro.")

        # Prepara dados do log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "round": round_num,
            "evaluation_type": "server",
            "metrics": MetricsCalculator.metrics_to_json(metrics),
        }

        if additional_info:
            log_entry["additional_info"] = MetricsCalculator.metrics_to_json(additional_info)

        # Salva em arquivo de avaliações do servidor
        eval_file = self.experiment_dir / "server_evaluations.json"

        # Carrega avaliações existentes ou cria nova lista
        if eval_file.exists():
            with open(eval_file, "r", encoding="utf-8") as f:
                evaluations = json.load(f)
        else:
            evaluations = []

        evaluations.append(log_entry)

        # Salva avaliações atualizadas
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(evaluations, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"Avaliação do servidor (Rodada {round_num}): "
            f"Acurácia={metrics.get('accuracy', 0):.4f}"
        )

    def save_config(self) -> None:
        """Salva a configuração do experimento em arquivo JSON."""
        if self.experiment_dir is None:
            raise RuntimeError("Diretórios não inicializados. Chame setup_directories() primeiro.")

        # Converte config para dicionário
        if hasattr(self.config, "__dict__"):
            config_dict = self.config.__dict__.copy()
        else:
            config_dict = dict(self.config)

        # Adiciona metadados
        config_data = {
            "model_name": self.model_name,
            "strategy_name": self.strategy_name,
            "timestamp": self.timestamp,
            "config": config_dict,
        }

        # Salva em arquivo
        config_file = self.experiment_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Configuração salva em: {config_file}")

    def save_summary(self, all_rounds_metrics: Optional[List[Dict[str, Any]]] = None) -> None:
        """Salva um resumo completo do experimento.

        Args:
            all_rounds_metrics: Lista opcional de métricas de todas as rodadas.
                               Se None, usa self.rounds_logs.
        """
        if self.experiment_dir is None:
            raise RuntimeError("Diretórios não inicializados. Chame setup_directories() primeiro.")

        metrics_to_summarize = all_rounds_metrics if all_rounds_metrics else self.rounds_logs

        # Calcula estatísticas agregadas
        if metrics_to_summarize:
            # Extrai apenas as métricas (remove metadados como timestamp)
            metrics_only = [entry.get("metrics", {}) for entry in metrics_to_summarize]
            aggregated = MetricsCalculator.calculate_aggregated_metrics(metrics_only)
        else:
            aggregated = {}

        # Prepara resumo
        summary = {
            "experiment_info": {
                "model_name": self.model_name,
                "strategy_name": self.strategy_name,
                "timestamp": self.timestamp,
                "total_rounds": len(self.rounds_logs),
                "total_clients": len(self.clients_logs),
            },
            "aggregated_metrics": aggregated,
            "final_round_metrics": (
                self.rounds_logs[-1].get("metrics", {}) if self.rounds_logs else {}
            ),
            "rounds_summary": [
                {
                    "round": entry["round"],
                    "accuracy": entry.get("metrics", {}).get("accuracy", 0),
                    "f1_score": entry.get("metrics", {}).get("f1_score", 0),
                    "training_time": entry.get("training_time_seconds", 0),
                }
                for entry in self.rounds_logs
            ],
        }

        # Salva resumo
        summary_file = self.experiment_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Resumo do experimento salvo em: {summary_file}")

        # Imprime resumo no console
        print("\n" + "=" * 80)
        print("RESUMO DO EXPERIMENTO")
        print("=" * 80)
        print(f"Modelo: {self.model_name}")
        print(f"Estratégia: {self.strategy_name}")
        print(f"Total de Rodadas: {len(self.rounds_logs)}")
        print(f"Total de Clientes: {len(self.clients_logs)}")

        if aggregated:
            print("\n" + "-" * 80)
            print("MÉTRICAS AGREGADAS (Média ± Desvio Padrão):")
            print("-" * 80)
            for metric_name, stats in aggregated.items():
                print(
                    f"{metric_name.upper():12s}: "
                    f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                    f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})"
                )

        print("=" * 80 + "\n")

    def get_experiment_path(self) -> Optional[Path]:
        """Retorna o caminho do diretório do experimento.

        Returns:
            Path para o diretório do experimento ou None se não inicializado.
        """
        return self.experiment_dir
