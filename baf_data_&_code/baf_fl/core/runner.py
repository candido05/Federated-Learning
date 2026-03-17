"""
Orquestracao do treinamento federado: executa simulacoes e avaliacao final.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import flwr as fl
from flwr.simulation import start_simulation

from ..config import CONFIG, SimulationRecorder
from ..data.metrics import evaluate_model_standard, evaluate_model_fair
from .client import TreeModelClient
from ..strategies import BaggingStrategy, CyclingStrategy
from ..reporting.experiment_logger import ExperimentLogger

# Suprimir logs INFO do Flower
logging.getLogger("flwr").setLevel(logging.WARNING)

# Suprimir warnings do Flower (client_fn deprecated signature)
import warnings
warnings.filterwarnings("ignore", message=".*DEPRECATED FEATURE.*")
warnings.filterwarnings("ignore", message=".*client_fn.*")

# Variaveis de ambiente para os workers do Ray
_RAY_ENV_VARS = {
    "RAY_DEDUP_LOGS": "0",
    "RAY_ENABLE_METRICS_AGENT": "0",
    "RAY_METRICS_AGENT_ENABLED": "0",
    "RAY_BACKEND_LOG_LEVEL": "fatal",
    "RAY_verbose_spill_logs": "0",
    "PYTHONWARNINGS": "ignore::DeprecationWarning",
}

_RAY_INIT_ARGS = {
    "include_dashboard": False,
    "logging_level": logging.ERROR,
    "configure_logging": False,
    "log_to_driver": True,
    "runtime_env": {"env_vars": _RAY_ENV_VARS},
}


def _make_client_fn(client_data_dict, val_data, model_type, model_params, scale_pos_weight):
    """
    Cria client_fn como closure, capturando os dados diretamente.

    Necessario porque Ray spawna workers em processos separados,
    onde variaveis globais/singletons nao estao disponiveis.

    Nota: start_simulation (API legada) usa internamente a assinatura
    antiga (cid: str). A assinatura Context so funciona com `flwr run`.
    """

    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        return TreeModelClient(
            client_id=client_id,
            train_data=client_data_dict[client_id],
            val_data=val_data,
            model_type=model_type,
            model_params=model_params,
            scale_pos_weight=scale_pos_weight,
        ).to_client()

    return client_fn


def run_federated_training(
    client_data: List[Dict],
    X_val: np.ndarray, y_val: np.ndarray, age_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, age_test: np.ndarray,
    best_params: Dict,
    scale_pos_weight: float,
    experiment_logger: Optional[ExperimentLogger] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Executa treinamento federado para todos os modelos e estrategias.

    Retorna tupla (DataFrame com resultados finais, dict de estrategias treinadas).
    """
    # Preparar dicts para closure do client_fn
    client_data_dict = {i: data for i, data in enumerate(client_data)}
    val_data = {'X': X_val, 'y': y_val, 'age': age_val}

    strategies_trained = {}

    for model_type in ['xgboost', 'lightgbm', 'catboost']:
        print("\n" + "#" * 70)
        print(f"# MODELO: {model_type.upper()}")
        print("#" * 70)

        # Criar client_fn com dados capturados na closure
        current_client_fn = _make_client_fn(
            client_data_dict, val_data, model_type, best_params, scale_pos_weight,
        )

        # BAGGING
        print("\n" + "-" * 60)
        print(f"ESTRATEGIA: BAGGING ({model_type.upper()})")
        print("-" * 60)

        bagging_recorder = SimulationRecorder(model_type, "bagging")
        bagging_strategy = BaggingStrategy(
            num_clients=CONFIG.num_clients,
            val_data=val_data,
            recorder=bagging_recorder,
        )

        bagging_recorder.start()
        start_simulation(
            client_fn=current_client_fn,
            num_clients=CONFIG.num_clients,
            config=fl.server.ServerConfig(num_rounds=CONFIG.num_rounds),
            strategy=bagging_strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
            ray_init_args=_RAY_INIT_ARGS,
        )
        bagging_recorder.stop()

        strategies_trained[f"{model_type}_bagging"] = bagging_strategy

        # CYCLING
        print("\n" + "-" * 60)
        print(f"ESTRATEGIA: CYCLING ({model_type.upper()})")
        print("-" * 60)

        cycling_recorder = SimulationRecorder(model_type, "cycling")
        cycling_strategy = CyclingStrategy(
            num_clients=CONFIG.num_clients,
            val_data=val_data,
            recorder=cycling_recorder,
        )

        cycling_recorder.start()
        start_simulation(
            client_fn=current_client_fn,
            num_clients=CONFIG.num_clients,
            config=fl.server.ServerConfig(num_rounds=CONFIG.num_rounds),
            strategy=cycling_strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
            ray_init_args=_RAY_INIT_ARGS,
        )
        cycling_recorder.stop()

        strategies_trained[f"{model_type}_cycling"] = cycling_strategy

        if experiment_logger:
            experiment_logger.add_simulation(bagging_recorder.record)
            experiment_logger.add_simulation(cycling_recorder.record)

    # Avaliacao final no conjunto de teste
    df_results = _evaluate_final(
        strategies_trained, X_test, y_test, age_test,
        experiment_logger=experiment_logger,
    )
    return df_results, strategies_trained


def _evaluate_final(
    strategies_trained: Dict,
    X_test: np.ndarray, y_test: np.ndarray, age_test: np.ndarray,
    experiment_logger: Optional[ExperimentLogger] = None,
) -> pd.DataFrame:
    """Avalia todos os modelos treinados no conjunto de teste."""
    print("\n" + "=" * 70)
    print("AVALIACAO FINAL (TESTE - MES 7)")
    print("=" * 70)

    all_results = []

    for key, strategy in strategies_trained.items():
        model_type, strategy_type = key.split('_')

        if strategy_type == 'bagging':
            if not strategy.client_models:
                print(f"  AVISO: {key} sem modelos!")
                continue
            y_prob = strategy.predict_ensemble(X_test)
        else:
            if strategy.current_model is None:
                print(f"  AVISO: {key} sem modelo!")
                continue
            y_prob = strategy.predict(X_test)

        metrics_std = evaluate_model_standard(y_test, y_prob, age_test)
        metrics_fair = evaluate_model_fair(y_test, y_prob, age_test)

        # Associar metricas finais de teste ao SimulationRecord
        if experiment_logger:
            for sim in experiment_logger.simulations:
                if sim.model_type == model_type and sim.strategy_type == strategy_type:
                    sim.final_metrics_standard = metrics_std
                    sim.final_metrics_fair = metrics_fair
                    break

        all_results.append({
            'model': f"{model_type.upper()} {strategy_type.capitalize()}",
            'threshold': 'single',
            'tpr_at_5fpr': metrics_std['tpr_at_5fpr'],
            'roc_auc': metrics_std['roc_auc'],
            'fairness_ratio': metrics_std['fairness_ratio'],
        })

        all_results.append({
            'model': f"{model_type.upper()} {strategy_type.capitalize()} (Fair)",
            'threshold': 'per_group',
            'tpr_at_5fpr': metrics_fair['tpr_at_5fpr'],
            'roc_auc': metrics_fair['roc_auc'],
            'fairness_ratio': metrics_fair['fairness_ratio'],
        })

    return pd.DataFrame(all_results)
