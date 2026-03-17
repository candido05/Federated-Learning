"""
Cliente Flower com conexao gRPC explicita.

Uso:
    python client.py --client-id 0 --model xgboost
    python client.py --client-id 1 --model lightgbm
"""

import argparse
import logging
import os
import pickle
import sys
import time
import warnings

# Suprimir warnings ANTES de importar flower/grpc
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*DEPRECATED.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*feature names.*")

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import flwr as fl
from flwr.common import (
    Code, Status, FitIns, FitRes, EvaluateIns, EvaluateRes,
    Parameters,
)

logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)

from config import (
    CLIENT_CONNECT_ADDRESS, NUM_CLIENTS, LOCAL_EPOCHS, LOG_EVERY,
    N_SAMPLES, TEST_SIZE, RANDOM_SEED,
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS,
)


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def load_higgs_client_data(client_id: int):
    """Carrega o Higgs e retorna a particao IID deste cliente."""
    print(f"[Cliente {client_id}] Carregando dataset Higgs ({N_SAMPLES} amostras)...")
    higgs = fetch_openml(name="higgs", version=2, as_frame=False, parser="auto")
    X, y = higgs.data[:N_SAMPLES], higgs.target[:N_SAMPLES].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )

    indices = np.array_split(np.arange(len(y_train)), NUM_CLIENTS)
    my_idx = indices[client_id]

    X_client = X_train[my_idx]
    y_client = y_train[my_idx]

    print(f"[Cliente {client_id}] Particao carregada:")
    print(f"    Treino: {len(my_idx)} amostras")
    print(f"    Teste:  {len(y_test)} amostras")
    print(f"    Classe 0: {(y_client == 0).sum()} ({(y_client == 0).mean()*100:.1f}%)")
    print(f"    Classe 1: {(y_client == 1).sum()} ({(y_client == 1).mean()*100:.1f}%)")

    return X_client, y_client, X_test, y_test


# ---------------------------------------------------------------------------
# Callbacks de progresso por epoca local
# ---------------------------------------------------------------------------

class XGBoostProgressCallback(xgb.callback.TrainingCallback):
    """Imprime progresso a cada LOG_EVERY boosting rounds."""

    def __init__(self, client_id: int, server_round: int, total_rounds: int):
        self.client_id = client_id
        self.server_round = server_round
        self.total_rounds = total_rounds

    def after_iteration(self, model, epoch, evals_log):
        current = epoch + 1
        if current % LOG_EVERY == 0 or current == self.total_rounds:
            loss_info = ""
            if evals_log:
                for ds_name, metrics_dict in evals_log.items():
                    for metric_name, values in metrics_dict.items():
                        val = values[-1] if isinstance(values[-1], float) else values[-1][0]
                        loss_info = f" | {ds_name}_{metric_name}={val:.4f}"
            print(f"    [Cliente {self.client_id}] Round {self.server_round} | "
                  f"Epoca local {current}/{self.total_rounds}{loss_info}")
            sys.stdout.flush()
        return False


def lgb_progress_callback(client_id: int, server_round: int, total_rounds: int):
    """Factory de callback para LightGBM."""
    start_iter = [None]  # mutable para capturar na closure
    def callback(env):
        if start_iter[0] is None:
            start_iter[0] = env.iteration
        local_iter = env.iteration - start_iter[0] + 1
        if local_iter % LOG_EVERY == 0 or local_iter == total_rounds:
            loss_info = ""
            if env.evaluation_result_list:
                ds, metric, val, _ = env.evaluation_result_list[0]
                loss_info = f" | {ds}_{metric}={val:.4f}"
            print(f"    [Cliente {client_id}] Round {server_round} | "
                  f"Epoca local {local_iter}/{total_rounds}{loss_info}")
            sys.stdout.flush()
    callback.order = 10
    return callback


# ---------------------------------------------------------------------------
# Treinamento de modelos
# ---------------------------------------------------------------------------

def train_model(model_type: str, X_train, y_train, client_id: int,
                server_round: int, warm_start_model=None):
    """Treina modelo com progresso por epoca."""
    total = LOCAL_EPOCHS

    if model_type == "xgboost":
        params = {**XGBOOST_PARAMS}
        cb = XGBoostProgressCallback(client_id, server_round, total)
        model = xgb.XGBClassifier(**params, callbacks=[cb])
        eval_set = [(X_train, y_train)]
        if warm_start_model is not None:
            model.fit(X_train, y_train, xgb_model=warm_start_model.get_booster(),
                      eval_set=eval_set, verbose=False)
        else:
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        # Limpar callbacks para que pickle nao tente serializa-los
        model.set_params(callbacks=None)
        if hasattr(model, '_callbacks'):
            model._callbacks = None

    elif model_type == "lightgbm":
        params = {**LIGHTGBM_PARAMS}
        cb = lgb_progress_callback(client_id, server_round, total)
        model = lgb.LGBMClassifier(**params)
        if warm_start_model is not None:
            model.fit(X_train, y_train, init_model=warm_start_model,
                      eval_set=[(X_train, y_train)], callbacks=[cb])
        else:
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train)], callbacks=[cb])

    elif model_type == "catboost":
        params = {**CATBOOST_PARAMS}
        params["verbose"] = 0  # Suprimir output nativo do CatBoost
        model = CatBoostClassifier(**params)

        print(f"    [Cliente {client_id}] Round {server_round} | "
              f"Treinando CatBoost ({total} iteracoes)...")
        sys.stdout.flush()

        if warm_start_model is not None:
            model.fit(X_train, y_train, init_model=warm_start_model, verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)

        print(f"    [Cliente {client_id}] Round {server_round} | "
              f"CatBoost {total}/{total} iteracoes concluidas")
        sys.stdout.flush()
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")

    return model


# ---------------------------------------------------------------------------
# Cliente Flower (usando Client base com bytes puros)
# ---------------------------------------------------------------------------

class SimpleClient(fl.client.Client):
    """
    Cliente FL que envia/recebe modelos como bytes puros via Parameters.tensors.
    Usa fl.client.Client (nao NumPyClient) para evitar conversoes numpy
    que corrompem a serializacao pickle.
    """

    def __init__(self, client_id: int, model_type: str,
                 X_train, y_train, X_test, y_test):
        self.client_id = client_id
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None

    def fit(self, ins: FitIns) -> FitRes:
        """Treina modelo localmente."""
        config = ins.config
        server_round = int(config.get("server_round", 0))
        use_warm_start = bool(config.get("warm_start", False))

        print(f"\n{'─'*60}")
        print(f"  [Cliente {self.client_id}] INICIO Round {server_round}")
        print(f"  [Cliente {self.client_id}] Modelo: {self.model_type} | "
              f"Epocas locais: {LOCAL_EPOCHS} | Warm start: {use_warm_start}")
        print(f"  [Cliente {self.client_id}] Amostras treino: {len(self.X_train)}")
        print(f"{'─'*60}")
        sys.stdout.flush()

        # Receber modelo para warm start
        warm_model = None
        if use_warm_start and ins.parameters.tensors:
            try:
                warm_model = pickle.loads(ins.parameters.tensors[0])
                print(f"    [Cliente {self.client_id}] Modelo global recebido para warm start")
                sys.stdout.flush()
            except Exception as e:
                print(f"    [Cliente {self.client_id}] Falha no warm start: {e}")

        # Treinar
        t0 = time.time()
        self.model = train_model(
            self.model_type, self.X_train, self.y_train,
            self.client_id, server_round, warm_model,
        )
        elapsed = time.time() - t0

        # Metricas pos-treino
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_prob)

        # Serializar modelo como bytes puros
        model_bytes = pickle.dumps(self.model)
        model_size_kb = len(model_bytes) / 1024

        print(f"\n  [Cliente {self.client_id}] FIM Round {server_round} | Tempo: {elapsed:.1f}s")
        print(f"  [Cliente {self.client_id}] Tamanho modelo: {model_size_kb:.1f} KB")
        print(f"  [Cliente {self.client_id}] Metricas no teste:")
        print(f"    Accuracy  = {acc:.4f}")
        print(f"    Precision = {prec:.4f}")
        print(f"    Recall    = {rec:.4f}")
        print(f"    F1-Score  = {f1:.4f}")
        print(f"    AUC-ROC   = {auc:.4f}")
        sys.stdout.flush()

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensors=[model_bytes], tensor_type="pickle"),
            num_examples=len(self.X_train),
            metrics={
                "client_id": self.client_id,
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "auc": float(auc),
                "training_time": float(elapsed),
                "model_size_kb": float(model_size_kb),
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Avalia modelo no conjunto de teste local."""
        if self.model is None:
            return EvaluateRes(
                status=Status(code=Code.OK, message="No model"),
                loss=1.0, num_examples=0, metrics={},
            )

        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_prob)

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(1 - acc),
            num_examples=len(self.X_test),
            metrics={"accuracy": acc, "auc": auc},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cliente FL com conexao gRPC explicita")
    parser.add_argument("--client-id", type=int, required=True,
                        help="ID do cliente (0, 1, 2, ...)")
    parser.add_argument("--model", type=str, required=True,
                        choices=["xgboost", "lightgbm", "catboost"],
                        help="Modelo a treinar")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CLIENTE {args.client_id} - {args.model.upper()}")
    print(f"  Servidor: {CLIENT_CONNECT_ADDRESS}")
    print(f"  Epocas locais: {LOCAL_EPOCHS} | Log a cada: {LOG_EVERY}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test = load_higgs_client_data(args.client_id)

    client = SimpleClient(
        client_id=args.client_id,
        model_type=args.model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    print(f"\n[Cliente {args.client_id}] Conectando ao servidor: {CLIENT_CONNECT_ADDRESS}")

    fl.client.start_client(
        server_address=CLIENT_CONNECT_ADDRESS,
        client=client,
    )

    print(f"\n[Cliente {args.client_id}] Treinamento federado concluido!")


if __name__ == "__main__":
    main()
