"""
Registro completo de experimentos de Federated Learning.

Salva JSON estruturado, CSVs e relatorio textual.
As dataclasses de registro (ClientRoundRecord, ServerRoundRecord, SimulationRecord)
e a classe SimulationRecorder estao em config.py.
"""

import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import SimulationRecord


class ExperimentLogger:
    """
    Logger principal que agrega todos os registros de simulacao
    e salva o experimento completo em disco.
    """

    def __init__(self, output_dir: str = "experiments"):
        self.experiment_start: Optional[float] = None
        self.experiment_end: Optional[float] = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = timestamp
        self.output_dir = os.path.join(output_dir, timestamp)

        self.config: Dict[str, Any] = {}
        self.dataset_info: Dict[str, Any] = {}
        self.hyperparameters: Dict[str, Any] = {}

        self.simulations: List[SimulationRecord] = []
        self.final_results: List[Dict] = []

    def start_experiment(self):
        self.experiment_start = time.time()

    def end_experiment(self):
        self.experiment_end = time.time()

    def set_config(
        self,
        num_clients: int,
        num_rounds: int,
        local_epochs: int,
        fpr_target: float,
        random_state: int,
        optuna_trials: int,
        sample_fraction: float,
        max_learning_rate: float,
    ):
        self.config = {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "fpr_target": fpr_target,
            "random_state": random_state,
            "optuna_trials": optuna_trials,
            "sample_fraction": sample_fraction,
            "max_learning_rate": max_learning_rate,
        }

    def set_dataset_info(
        self,
        total_samples: int,
        num_features: int,
        train_size: int,
        val_size: int,
        test_size: int,
        fraud_rate_train: float,
        fraud_rate_val: float,
        fraud_rate_test: float,
        scale_pos_weight: float,
        client_distributions: List[Dict[str, Any]],
    ):
        self.dataset_info = {
            "total_samples": total_samples,
            "num_features": num_features,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "fraud_rate_train": fraud_rate_train,
            "fraud_rate_val": fraud_rate_val,
            "fraud_rate_test": fraud_rate_test,
            "scale_pos_weight": scale_pos_weight,
            "client_distributions": client_distributions,
        }

    def set_hyperparameters(self, best_params: Dict):
        self.hyperparameters = best_params

    def add_simulation(self, record: SimulationRecord):
        self.simulations.append(record)

    def set_final_results(self, df_results):
        """Aceita DataFrame pandas e converte para lista de dicts."""
        self.final_results = df_results.to_dict(orient="records")

    def save(self):
        """Salva tudo em disco."""
        os.makedirs(self.output_dir, exist_ok=True)

        total_time = (
            (self.experiment_end - self.experiment_start)
            if self.experiment_start and self.experiment_end
            else 0.0
        )

        experiment_data = {
            "experiment_id": self.experiment_id,
            "total_time_sec": total_time,
            "environment": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "timestamp_start": (
                    datetime.fromtimestamp(self.experiment_start).isoformat()
                    if self.experiment_start else None
                ),
                "timestamp_end": (
                    datetime.fromtimestamp(self.experiment_end).isoformat()
                    if self.experiment_end else None
                ),
            },
            "config": self.config,
            "dataset": self.dataset_info,
            "hyperparameters": self.hyperparameters,
            "simulations": [self._simulation_to_dict(s) for s in self.simulations],
            "final_results": self.final_results,
        }

        # 1. JSON completo
        json_path = os.path.join(self.output_dir, "experiment.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                experiment_data, f, indent=2, ensure_ascii=False,
                default=self._json_serializer,
            )

        # 2. CSV resumo das simulacoes
        csv_path = os.path.join(self.output_dir, "simulations_summary.csv")
        self._write_simulations_csv(csv_path)

        # 3. CSVs por simulacao (rodadas do servidor + rodadas dos clientes)
        for sim in self.simulations:
            prefix = f"{sim.model_type}_{sim.strategy_type}"

            rounds_csv = os.path.join(self.output_dir, f"{prefix}_rounds.csv")
            self._write_rounds_csv(sim, rounds_csv)

            clients_csv = os.path.join(self.output_dir, f"{prefix}_client_rounds.csv")
            self._write_client_rounds_csv(sim, clients_csv)

        # 4. Relatorio legivel
        summary_path = os.path.join(self.output_dir, "summary.txt")
        self._write_summary(summary_path, experiment_data)

        print(f"\n[ExperimentLogger] Resultados salvos em: {self.output_dir}/")
        print(f"  - experiment.json (dados completos)")
        print(f"  - simulations_summary.csv")
        print(f"  - <modelo>_<estrategia>_rounds.csv (metricas por rodada)")
        print(f"  - <modelo>_<estrategia>_client_rounds.csv (metricas por cliente)")
        print(f"  - summary.txt (relatorio legivel)")

    @staticmethod
    def _json_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Objeto do tipo {type(obj)} nao e serializavel em JSON")

    @staticmethod
    def _simulation_to_dict(sim: SimulationRecord) -> Dict:
        return {
            "model_type": sim.model_type,
            "strategy_type": sim.strategy_type,
            "duration_sec": sim.duration_sec,
            "total_bytes_transferred": sim.total_bytes_transferred,
            "num_rounds_completed": len(sim.server_rounds),
            "server_rounds": [asdict(r) for r in sim.server_rounds],
            "client_rounds": [asdict(r) for r in sim.client_rounds],
            "final_metrics_standard": sim.final_metrics_standard,
            "final_metrics_fair": sim.final_metrics_fair,
        }

    def _write_simulations_csv(self, path: str):
        rows = []
        for sim in self.simulations:
            last_round = sim.server_rounds[-1] if sim.server_rounds else None
            rows.append({
                "model_type": sim.model_type,
                "strategy_type": sim.strategy_type,
                "duration_sec": f"{sim.duration_sec:.2f}",
                "total_bytes_transferred": sim.total_bytes_transferred,
                "num_rounds": len(sim.server_rounds),
                "final_tpr_standard": (
                    f"{last_round.tpr_standard:.4f}" if last_round else ""
                ),
                "final_fairness_standard": (
                    f"{last_round.fairness_standard:.4f}" if last_round else ""
                ),
                "final_tpr_fair": (
                    f"{last_round.tpr_fair:.4f}" if last_round else ""
                ),
                "final_fairness_fair": (
                    f"{last_round.fairness_fair:.4f}" if last_round else ""
                ),
            })

        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    @staticmethod
    def _write_rounds_csv(sim: SimulationRecord, path: str):
        if not sim.server_rounds:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=asdict(sim.server_rounds[0]).keys(),
            )
            writer.writeheader()
            for r in sim.server_rounds:
                writer.writerow(asdict(r))

    @staticmethod
    def _write_client_rounds_csv(sim: SimulationRecord, path: str):
        if not sim.client_rounds:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=asdict(sim.client_rounds[0]).keys(),
            )
            writer.writeheader()
            for r in sim.client_rounds:
                writer.writerow(asdict(r))

    @staticmethod
    def _write_summary(path: str, data: Dict):
        lines = []
        lines.append("=" * 80)
        lines.append("RELATORIO DO EXPERIMENTO DE FEDERATED LEARNING")
        lines.append("=" * 80)
        lines.append(f"ID do Experimento: {data['experiment_id']}")
        lines.append(f"Tempo Total: {data['total_time_sec']:.2f} segundos")
        lines.append(f"Plataforma: {data['environment']['platform']}")
        lines.append(f"Python: {data['environment']['python_version']}")
        lines.append(f"Inicio: {data['environment']['timestamp_start']}")
        lines.append(f"Fim: {data['environment']['timestamp_end']}")
        lines.append("")

        # Configuracao
        lines.append("-" * 40)
        lines.append("CONFIGURACAO")
        lines.append("-" * 40)
        for k, v in data["config"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

        # Dataset
        lines.append("-" * 40)
        lines.append("DATASET")
        lines.append("-" * 40)
        ds = data["dataset"]
        for k, v in ds.items():
            if k == "client_distributions":
                lines.append(f"  Distribuicao por cliente:")
                for cd in v:
                    lines.append(
                        f"    Cliente {cd['client_id']}: "
                        f"{cd['total_samples']} amostras, "
                        f"{cd['fraud_samples']} fraudes "
                        f"({cd['fraud_rate']:.4f})"
                    )
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

        # Hiperparametros
        lines.append("-" * 40)
        lines.append("HIPERPARAMETROS (Agregados)")
        lines.append("-" * 40)
        for algo, params in data["hyperparameters"].items():
            lines.append(f"  {algo}:")
            if isinstance(params, dict):
                for pk, pv in params.items():
                    lines.append(f"    {pk}: {pv}")
            else:
                lines.append(f"    {params}")
        lines.append("")

        # Simulacoes
        lines.append("-" * 40)
        lines.append("SIMULACOES")
        lines.append("-" * 40)
        for sim in data["simulations"]:
            label = f"{sim['model_type'].upper()} {sim['strategy_type'].capitalize()}"
            lines.append(f"\n  {label}")
            lines.append(f"    Duracao: {sim['duration_sec']:.2f}s")
            lines.append(f"    Bytes transferidos: {sim['total_bytes_transferred']:,}")
            lines.append(f"    Rodadas completadas: {sim['num_rounds_completed']}")

            if sim["final_metrics_standard"]:
                m = sim["final_metrics_standard"]
                lines.append(
                    f"    [Teste Standard] TPR={m.get('tpr_at_5fpr', 0):.4f}, "
                    f"AUC={m.get('roc_auc', 0):.4f}, "
                    f"Fairness={m.get('fairness_ratio', 0):.4f}"
                )
            if sim["final_metrics_fair"]:
                m = sim["final_metrics_fair"]
                lines.append(
                    f"    [Teste Fair]     TPR={m.get('tpr_at_5fpr', 0):.4f}, "
                    f"AUC={m.get('roc_auc', 0):.4f}, "
                    f"Fairness={m.get('fairness_ratio', 0):.4f}"
                )

            # Convergencia por rodada
            if sim["server_rounds"]:
                lines.append(f"    Convergencia (TPR Standard por rodada):")
                for sr in sim["server_rounds"]:
                    client_info = ""
                    if sr.get("trained_client") is not None:
                        client_info = f" [Cliente {sr['trained_client']}]"
                    lines.append(
                        f"      R{sr['round_num']:02d}: "
                        f"TPR={sr['tpr_standard']:.4f}, "
                        f"Fairness={sr['fairness_standard']:.4f}, "
                        f"Tempo={sr['round_time_sec']:.1f}s"
                        f"{client_info}"
                    )
        lines.append("")

        # Resultados finais
        lines.append("-" * 40)
        lines.append("RESULTADOS FINAIS")
        lines.append("-" * 40)
        for result in data["final_results"]:
            tpr = result.get('tpr_at_5fpr', 0)
            auc = result.get('roc_auc', 0)
            fr = result.get('fairness_ratio', 0)
            tpr_ok = "[OK]" if tpr > 0.52 else "[ X]"
            fr_ok = "[OK]" if 0.9 <= fr <= 1.1 else "[ X]"
            lines.append(
                f"  {tpr_ok} {fr_ok} {result.get('model', '?')}: "
                f"TPR={tpr:.4f}, AUC={auc:.4f}, Fairness={fr:.4f}"
            )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
