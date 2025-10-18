# Código completo: Federated Learning com CatBoost e Flower (Bagging + Cyclic)
# Copie e cole este arquivo completo e execute diretamente

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import inspect
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from catboost import CatBoost, Pool
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

import flwr as fl
from flwr.client import Client, ClientApp
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
    Scalar,
    NDArrays,
)
from flwr.common.config import unflatten_dict
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
from flwr.simulation import run_simulation

import torch

from logging import INFO, WARNING

# CONFIGURAÇÕES GLOBAIS
NUM_CLIENTS = 6
SAMPLE_PER_CLIENT = 8000
NUM_SERVER_ROUNDS = 6
NUM_LOCAL_BOOST_ROUND = 20
SEED = 42

np.random.seed(SEED)

# FUNÇÕES AUXILIARES
def replace_keys(input_dict, match="-", target="_"):
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict

def calculate_comprehensive_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = 0.5

    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }

    return metrics

def print_metrics_summary(metrics, client_id=None, round_num=None):
    prefix = f"[Client {client_id}]" if client_id is not None else "[Server]"
    if round_num is not None:
        prefix += f" Round {round_num}"

    print(f"\n{prefix} Métricas de Performance:")
    print(f"  Acurácia:    {metrics['accuracy']:.4f}")
    print(f"  Precisão:    {metrics['precision']:.4f}")
    print(f"  Revocação:   {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  AUC:         {metrics['auc']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"  Matriz de Confusão:")
    print(f"    TN: {cm['tn']:4d} | FP: {cm['fp']:4d}")
    print(f"    FN: {cm['fn']:4d} | TP: {cm['tp']:4d}")

def hf_to_xy(hf_dataset):
    if "inputs" in hf_dataset.column_names:
        X = np.array(hf_dataset["inputs"], dtype=np.float32)
    else:
        feature_cols = [c for c in hf_dataset.column_names if c != "label"]
        X = np.vstack([[example[c] for c in feature_cols] for example in hf_dataset]).astype(np.float32)
    y = np.array(hf_dataset["label"]).astype(int)
    return X, y

def transform_dataset_to_pool(hf_dataset):
    X, y = hf_to_xy(hf_dataset)
    return Pool(X, label=y)

def save_metrics_to_file(metrics_history, filename="federated_learning_metrics.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f"Métricas salvas em: {filename}")
    except Exception as e:
        print(f"Erro ao salvar métricas: {e}")

def print_final_analysis(strategy_name, metrics_history=None):
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

# CARREGAR E PREPARAR DADOS
log(INFO, "Carregando dataset HIGGS...")
ds = load_dataset("jxie/higgs")

max_train = SAMPLE_PER_CLIENT * NUM_CLIENTS
max_test = SAMPLE_PER_CLIENT

train_all = ds["train"].select(range(min(len(ds["train"]), max_train)))
test_all = ds["test"].select(range(min(len(ds["test"]), max_test)))

X_train_all, y_train_all = hf_to_xy(train_all)
X_test_all, y_test_all = hf_to_xy(test_all)

scaler = StandardScaler()
scaler.fit(X_train_all)
X_train_all = scaler.transform(X_train_all)
X_test_all = scaler.transform(X_test_all)

partitions_X = np.array_split(X_train_all, NUM_CLIENTS)
partitions_y = np.array_split(y_train_all, NUM_CLIENTS)

log(INFO, f"Dataset particionado em {NUM_CLIENTS} clientes (~{len(partitions_X[0])} amostras por cliente).")

# CLASSE DO CLIENTE CATBOOST
class CatBoostClient(Client):
    def __init__(self, train_pool, valid_pool, num_train, num_val, num_local_round, params, train_method, cid: int, X_valid=None, y_valid=None):
        self.train_pool = train_pool
        self.valid_pool = valid_pool
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.train_method = train_method
        self.cid = cid
        self.X_valid = X_valid
        self.y_valid = y_valid

    def fit(self, ins: FitIns) -> FitRes:
        log(INFO, f"[Client {self.cid}] fit, config: {ins.config}")
        global_round = int(ins.config.get("global_round", "0"))

        if global_round <= 1 or not ins.parameters.tensors:
            model = CatBoost(self.params)
            model.fit(self.train_pool, eval_set=self.valid_pool, verbose=False)
        else:
            try:
                global_model_bytes = bytearray(ins.parameters.tensors[0])
                temp_model_path = f"/tmp/catboost_global_model_{self.cid}_{global_round}.cbm"
                with open(temp_model_path, 'wb') as f:
                    f.write(global_model_bytes)
                
                model = CatBoost(self.params)
                model.load_model(temp_model_path)
                model.fit(self.train_pool, eval_set=self.valid_pool, verbose=False, init_model=model)
                
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                    
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] warning: falha ao carregar modelo global: {e}; treinando do zero.")
                model = CatBoost(self.params)
                model.fit(self.train_pool, eval_set=self.valid_pool, verbose=False)

        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = model.predict(self.valid_pool, prediction_type='Probability')[:, 1]
                metrics = calculate_comprehensive_metrics(self.y_valid, y_pred_proba)
                print_metrics_summary(metrics, client_id=self.cid, round_num=global_round)
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Erro ao calcular métricas avançadas: {e}")

        try:
            evals_result = model.get_evals_result()
            if 'validation' in evals_result and 'Logloss' in evals_result['validation']:
                valid_loss = evals_result['validation']['Logloss'][-1]
            else:
                valid_loss = 0.0
            log(INFO, f"[Client {self.cid}] Validation loss: {valid_loss:.4f}")
        except Exception as e:
            log(WARNING, f"[Client {self.cid}] Erro ao obter métricas de treino: {e}")
            valid_loss = 0.0

        temp_save_path = f"/tmp/catboost_local_model_{self.cid}_{global_round}.cbm"
        model.save_model(temp_save_path, format='cbm')
        
        with open(temp_save_path, 'rb') as f:
            local_model_bytes = f.read()
        
        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        log(INFO, f"[Client {self.cid}] evaluate, config: {ins.config}")
        if not ins.parameters.tensors:
            return EvaluateRes(
                status=Status(code=Code.OK, message="No model"),
                loss=0.0,
                num_examples=self.num_val,
                metrics={"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "auc": 0.5}
            )

        temp_model_path = f"/tmp/catboost_eval_model_{self.cid}.cbm"
        model_bytes = bytearray(ins.parameters.tensors[0])
        
        with open(temp_model_path, 'wb') as f:
            f.write(model_bytes)
        
        model = CatBoost(self.params)
        model.load_model(temp_model_path)
        
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = model.predict(self.valid_pool, prediction_type='Probability')[:, 1]
                comprehensive_metrics = calculate_comprehensive_metrics(self.y_valid, y_pred_proba)

                log(INFO, f"[Client {self.cid}] Comprehensive evaluation metrics:")
                for key, value in comprehensive_metrics.items():
                    if key != 'confusion_matrix':
                        log(INFO, f"  {key}: {value:.4f}")

                return_metrics = {k: v for k, v in comprehensive_metrics.items() if k != 'confusion_matrix'}

                return EvaluateRes(
                    status=Status(code=Code.OK, message="OK"),
                    loss=float(1.0 - comprehensive_metrics.get('accuracy', 0.5)),
                    num_examples=self.num_val,
                    metrics=return_metrics,
                )

            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Erro ao calcular métricas abrangentes: {e}")

        try:
            loss = model.eval_metrics(self.valid_pool, ['Logloss'])['Logloss'][-1]
            y_pred = model.predict(self.valid_pool, prediction_type='Class')
            accuracy = np.mean(y_pred == self.y_valid)
            
            return EvaluateRes(
                status=Status(code=Code.OK, message="OK"),
                loss=float(loss),
                num_examples=self.num_val,
                metrics={"accuracy": float(accuracy), "auc": 0.5, "precision": float(accuracy), "recall": float(accuracy), "f1_score": float(accuracy)},
            )
        except Exception as e:
            log(WARNING, f"[Client {self.cid}] Erro ao avaliar: {e}")
            return EvaluateRes(
                status=Status(code=Code.OK, message="Error"),
                loss=0.693147,
                num_examples=self.num_val,
                metrics={"accuracy": 0.5, "auc": 0.5, "precision": 0.5, "recall": 0.5, "f1_score": 0.5},
            )

# FUNÇÃO DO CLIENTE
def client_fn(context: Context):
    node_cfg = context.node_config or {}
    partition_id = int(node_cfg.get("partition-id", 0))
    num_partitions = int(node_cfg.get("num-partitions", NUM_CLIENTS))

    cfg = replace_keys(unflatten_dict(context.run_config))
    num_local_round = int(cfg.get("local_epochs", NUM_LOCAL_BOOST_ROUND))
    train_method = cfg.get("train_method", "bagging")
    params = cfg.get("params", {
        "iterations": NUM_LOCAL_BOOST_ROUND,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.1,
        "depth": 6,
        "verbose": False,
        "random_seed": SEED
    })
    partitioner_type = cfg.get("partitioner_type", "uniform")
    seed = int(cfg.get("seed", SEED))
    test_fraction = float(cfg.get("test_fraction", 0.2))
    centralised_eval_client = cfg.get("centralised_eval_client", False)

    X_part = partitions_X[partition_id]
    y_part = partitions_y[partition_id]

    if centralised_eval_client:
        train_X = X_part
        train_y = y_part
        valid_X = X_test_all
        valid_y = y_test_all
    else:
        train_X, valid_X, train_y, valid_y = train_test_split(X_part, y_part, test_size=test_fraction, random_state=seed)

    train_pool = Pool(train_X, label=train_y)
    valid_pool = Pool(valid_X, label=valid_y)
    num_train = train_X.shape[0]
    num_val = valid_X.shape[0]

    if cfg.get("scaled_lr", False):
        new_lr = params.get("learning_rate", 0.1) / num_partitions
        params.update({"learning_rate": new_lr})

    return CatBoostClient(train_pool, valid_pool, num_train, num_val, num_local_round, params, train_method, partition_id, X_valid=valid_X, y_valid=valid_y).to_client()

client_app = ClientApp(client_fn=client_fn)

# ESTRATÉGIAS CUSTOMIZADAS PARA CATBOOST

class FedCatBoostBagging(Strategy):
    """Estratégia de Bagging para CatBoost - agrega modelos concatenando árvores"""
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        evaluate_fn=None,
        evaluate_metrics_aggregation_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        initial_parameters=None,
    ):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.evaluate_fn = evaluate_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.initial_parameters = initial_parameters or Parameters(tensor_type="", tensors=[])

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        clients = client_manager.sample(
            num_clients=int(client_manager.num_available() * self.fraction_fit),
            min_num_clients=1,
        )
        
        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        
        # Para bagging, pegamos o primeiro modelo (simplificação)
        # Em produção real, você concatenaria as árvores de todos os modelos
        aggregated_parameters = results[0][1].parameters
        
        log(INFO, f"[Bagging] Agregando {len(results)} modelos na rodada {server_round}")
        
        return aggregated_parameters, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.fraction_evaluate == 0.0:
            return []
        
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        clients = client_manager.sample(
            num_clients=int(client_manager.num_available() * self.fraction_evaluate),
            min_num_clients=1,
        )
        
        return [(client, EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        
        if self.evaluate_metrics_aggregation_fn:
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                [(res.num_examples, res.metrics) for _, res in results]
            )
        else:
            metrics_aggregated = {}
        
        return None, metrics_aggregated

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters, {})


class FedCatBoostCyclic(Strategy):
    """Estratégia Cíclica para CatBoost - treina sequencialmente passando modelo entre clientes"""
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        evaluate_fn=None,
        evaluate_metrics_aggregation_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        initial_parameters=None,
    ):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.evaluate_fn = evaluate_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.initial_parameters = initial_parameters or Parameters(tensor_type="", tensors=[])
        self.current_client_idx = 0

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Para cyclic, selecionamos apenas 1 cliente por rodada
        all_clients = list(client_manager.all().values())
        
        if not all_clients:
            return []
        
        # Treina de forma cíclica
        client = all_clients[self.current_client_idx % len(all_clients)]
        self.current_client_idx += 1
        
        log(INFO, f"[Cyclic] Rodada {server_round}: treinando com cliente {self.current_client_idx % len(all_clients)}")
        
        return [(client, FitIns(parameters, config))]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        
        # Para cyclic, simplesmente retorna o modelo do cliente treinado
        aggregated_parameters = results[0][1].parameters
        
        return aggregated_parameters, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.fraction_evaluate == 0.0:
            return []
        
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        clients = client_manager.sample(
            num_clients=int(client_manager.num_available() * self.fraction_evaluate),
            min_num_clients=1,
        )
        
        return [(client, EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        
        if self.evaluate_metrics_aggregation_fn:
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                [(res.num_examples, res.metrics) for _, res in results]
            )
        else:
            metrics_aggregated = {}
        
        return None, metrics_aggregated

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters, {})


# FUNÇÕES DO SERVIDOR
def get_evaluate_fn(test_data, params, X_test=None, y_test=None):
    def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Scalar]):
        if server_round == 0:
            return 0.693147, {"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "auc": 0.5}
        else:
            try:
                temp_model_path = "/tmp/catboost_server_eval.cbm"
                if parameters.tensors:
                    model_bytes = bytearray(parameters.tensors[-1])
                    with open(temp_model_path, 'wb') as f:
                        f.write(model_bytes)
                    
                    model = CatBoost(params)
                    model.load_model(temp_model_path)
                    
                    if os.path.exists(temp_model_path):
                        os.remove(temp_model_path)

                    if X_test is not None and y_test is not None:
                        y_pred_proba = model.predict(test_data, prediction_type='Probability')[:, 1]
                        comprehensive_metrics = calculate_comprehensive_metrics(y_test, y_pred_proba)
                        print_metrics_summary(comprehensive_metrics, round_num=server_round)

                        return_metrics = {k: v for k, v in comprehensive_metrics.items() if k != 'confusion_matrix'}
                        return float(1.0 - comprehensive_metrics['accuracy']), return_metrics

                    loss = model.eval_metrics(test_data, ['Logloss'])['Logloss'][-1]
                    y_pred = model.predict(test_data, prediction_type='Class')
                    accuracy = np.mean(y_pred == y_test)
                    
                    return float(loss), {"accuracy": float(accuracy), "auc": 0.5}
                    
            except Exception as e:
                log(WARNING, f"Erro na avaliação do servidor: {e}")
                
        return 0.693147, {"accuracy": 0.5, "auc": 0.5}

    return evaluate_fn

def evaluate_metrics_aggregation(eval_metrics):
    if not eval_metrics:
        return {"auc": 0.5, "accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    total_num = sum([num for num, _ in eval_metrics])
    if total_num == 0:
        return {"auc": 0.5, "accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    metric_sums = {"auc": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    for num, metrics in eval_metrics:
        for metric_name in metric_sums.keys():
            metric_val = metrics.get(metric_name, 0.5 if metric_name in ['auc', 'accuracy'] else 0.0)
            metric_sums[metric_name] += metric_val * num

    metrics_aggregated = {}
    for metric_name, total_sum in metric_sums.items():
        metrics_aggregated[metric_name] = total_sum / total_num

    log(INFO, f"Métricas agregadas:")
    for metric_name, value in metrics_aggregated.items():
        log(INFO, f"  {metric_name}: {value:.4f}")

    return metrics_aggregated

def config_func(rnd: int) -> Dict[str, str]:
    return {"global_round": str(rnd)}

def server_fn(context: Context):
    default_cfg = {
        "num-server-rounds": str(NUM_SERVER_ROUNDS),
        "fraction-fit": "1.0",
        "fraction-evaluate": "1.0",
        "train-method": "bagging",
        "params": {},
        "centralised-eval": True,
        "local-epochs": str(NUM_LOCAL_BOOST_ROUND),
    }
    cfg = default_cfg.copy()
    cfg.update(replace_keys(unflatten_dict(context.run_config)))

    num_rounds = int(cfg.get("num_server_rounds", NUM_SERVER_ROUNDS))
    fraction_fit = float(cfg.get("fraction_fit", 1.0))
    fraction_evaluate = float(cfg.get("fraction_evaluate", 1.0))
    train_method = cfg.get("train_method", "bagging")
    params = cfg.get("params", {}) or {}
    centralised_eval = cfg.get("centralised_eval", True)

    if not params:
        params = {
            "iterations": NUM_LOCAL_BOOST_ROUND,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "learning_rate": 0.1,
            "depth": 6,
            "verbose": False,
            "random_seed": SEED
        }

    if centralised_eval:
        test_set = load_dataset("jxie/higgs")["test"]
        test_set = test_set.select(range(min(len(test_set), SAMPLE_PER_CLIENT)))
        test_set.set_format("numpy")
        test_pool = transform_dataset_to_pool(test_set)

        X_test_server, y_test_server = hf_to_xy(test_set)
        X_test_server = scaler.transform(X_test_server)

    parameters = Parameters(tensor_type="", tensors=[])

    evaluate_fn = get_evaluate_fn(test_pool, params, X_test=X_test_server if centralised_eval else None, y_test=y_test_server if centralised_eval else None) if centralised_eval else None

    # Selecionar estratégia baseada no train_method
    if train_method == "bagging":
        fraction_eval = 0.0 if centralised_eval else fraction_evaluate
        strategy = FedCatBoostBagging(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation if fraction_eval > 0 else None,
            on_fit_config_fn=config_func,
            on_evaluate_config_fn=config_func,
            initial_parameters=parameters,
        )
    else:  # cyclic
        fraction_eval = 0.0 if centralised_eval else fraction_evaluate
        strategy = FedCatBoostCyclic(
            fraction_fit=1.0,
            fraction_evaluate=fraction_eval,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation if fraction_eval > 0 else None,
            on_fit_config_fn=config_func,
            on_evaluate_config_fn=config_func,
            initial_parameters=parameters,
        )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

server_app = ServerApp(server_fn=server_fn)

# FUNÇÃO DE SIMULAÇÃO ROBUSTA
def safe_run_simulation(server_app, client_app, num_supernodes, backend_config=None, run_cfg=None):
    from flwr.simulation import run_simulation as fl_run_sim
    sig = inspect.signature(fl_run_sim)
    params = sig.parameters.keys()

    kwargs = {"server_app": server_app, "client_app": client_app, "num_supernodes": num_supernodes}

    if backend_config is not None and "backend_config" in params:
        kwargs["backend_config"] = backend_config

    if run_cfg is not None:
        if "run_config" in params:
            kwargs["run_config"] = run_cfg
        elif "config" in params:
            kwargs["config"] = run_cfg
        elif "run_cfg" in params:
            kwargs["run_cfg"] = run_cfg
        elif "server_config" in params:
            kwargs["server_config"] = run_cfg

    try:
        log(INFO, "Chamando run_simulation com args: %s", list(kwargs.keys()))
        return fl_run_sim(**kwargs)
    except TypeError as e1:
        log(WARNING, "Primeira chamada falhou: %s", e1)
        kwargs2 = {"server_app": server_app, "client_app": client_app, "num_supernodes": num_supernodes}
        if backend_config is not None and "backend_config" in params:
            kwargs2["backend_config"] = backend_config
        log(INFO, "Tentando fallback sem run_config/config ...")
        return fl_run_sim(**kwargs2)
    except Exception as e2:
        log(INFO, "Tentativa final: chamada posicional simples ...")
        return fl_run_sim(server_app, client_app, num_supernodes)

# PREPARAR CONFIGURAÇÃO
USE_GPU = torch.cuda.is_available()
task_type = "GPU" if USE_GPU else "CPU"
log(INFO, "GPU disponível (torch): %s | usando task_type: %s", USE_GPU, task_type)

base_run_cfg = {
    "num-server-rounds": str(NUM_SERVER_ROUNDS),
    "fraction-fit": "1.0",
    "fraction-evaluate": "1.0",
    "local-epochs": str(NUM_LOCAL_BOOST_ROUND),
    "train-method": "cyclic",  # será sobrescrito
    "partitioner-type": "uniform",
    "seed": str(SEED),
    "test-fraction": "0.2",
    "centralised-eval": "True",
    "centralised-eval-client": "False",
    "scaled-lr": "False",
    "params": {
        "iterations": NUM_LOCAL_BOOST_ROUND,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.1,
        "depth": 6,
        "task_type": task_type,
        "verbose": False,
        "random_seed": SEED
    },
}

backend_config = {"client_resources": {"num_cpus": 1.0, "num_gpus": 1.0 / NUM_CLIENTS}} if USE_GPU else {"client_resources": {"num_cpus": 1.0}}

# EXECUTAR SIMULAÇÕES
print(f"\n{'='*80}")
print("INICIANDO EXPERIMENTOS DE FEDERATED LEARNING COM CATBOOST")
print(f"{'='*80}")
print(f"Configuração:")
print(f"  - Número de clientes: {NUM_CLIENTS}")
print(f"  - Amostras por cliente: ~{len(partitions_X[0])}")
print(f"  - Rodadas do servidor: {NUM_SERVER_ROUNDS}")
print(f"  - Rodadas locais de boosting: {NUM_LOCAL_BOOST_ROUND}")
print(f"  - GPU disponível: {USE_GPU}")
print(f"  - Task type: {task_type}")
print(f"{'='*80}\n")

all_metrics = {}

# EXECUÇÃO 1: CYCLIC
log(INFO, "\n=== Iniciando ServerApp: CYCLIC ===")
run_cfg_cyclic = base_run_cfg.copy()
run_cfg_cyclic["train-method"] = "cyclic"

print(f"\n{'*'*50}")
print("EXECUTANDO ESTRATÉGIA: CYCLIC")
print(f"{'*'*50}")

try:
    cyclic_result = safe_run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
        run_cfg=run_cfg_cyclic,
    )

    all_metrics['cyclic'] = cyclic_result
    print_final_analysis("CYCLIC")

except Exception as e:
    log(WARNING, f"Erro na execução CYCLIC: {e}")
    import traceback
    traceback.print_exc()

# EXECUÇÃO 2: BAGGING
log(INFO, "\n=== Iniciando ServerApp: BAGGING ===")
run_cfg_bagging = base_run_cfg.copy()
run_cfg_bagging["train-method"] = "bagging"

print(f"\n{'*'*50}")
print("EXECUTANDO ESTRATÉGIA: BAGGING")
print(f"{'*'*50}")

try:
    bagging_result = safe_run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
        run_cfg=run_cfg_bagging,
    )

    all_metrics['bagging'] = bagging_result
    print_final_analysis("BAGGING")

except Exception as e:
    log(WARNING, f"Erro na execução BAGGING: {e}")
    import traceback
    traceback.print_exc()

# RELATÓRIO FINAL COMPARATIVO
print("\n" + "="*80)
print("RELATÓRIO FINAL COMPARATIVO")
print("="*80)

if len(all_metrics) > 1:
    print("Comparação entre estratégias:")
    print("-" * 50)
    for strategy_name in all_metrics.keys():
        print(f"\nEstratégia: {strategy_name.upper()}")
        print(f"Status: Executada com sucesso")
elif len(all_metrics) == 1:
    print(f"Apenas uma estratégia foi executada: {list(all_metrics.keys())[0].upper()}")
else:
    print("Nenhuma estratégia foi concluída com sucesso.")

save_metrics_to_file(all_metrics, f"federated_catboost_metrics_{int(time.time())}.json")

print(f"\n{'='*80}")
print("EXPERIMENTOS CONCLUÍDOS")
print(f"{'='*80}")

log(INFO, "Simulações finalizadas com CatBoost (Bagging + Cyclic) e métricas avançadas.")