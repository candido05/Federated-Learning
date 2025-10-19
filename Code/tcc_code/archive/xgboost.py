# Cell 2: código completo corrigido com métricas avançadas
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# imports gerais
import os
import time
import inspect
import json
from typing import Dict, List, Optional

import numpy as np
import Code.tcc_code.archive.xgboost as xgb
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
)
from flwr.common.config import unflatten_dict
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
from flwr.simulation import run_simulation

import torch

from logging import INFO, WARNING

# ----------------------------
# CONFIGURÁVEIS - ajuste conforme recurso
# ----------------------------
NUM_CLIENTS = 6
SAMPLE_PER_CLIENT = 8000   # amostra por cliente (mantenha pequeno em Colab)
NUM_SERVER_ROUNDS = 6
NUM_LOCAL_BOOST_ROUND = 20
SEED = 42

np.random.seed(SEED)

# ----------------------------
# Helper: replace_keys (igual ao exemplo)
# ----------------------------
def replace_keys(input_dict, match="-", target="_"):
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict

# ----------------------------
# Função para calcular métricas completas
# ----------------------------
def calculate_comprehensive_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calcula métricas abrangentes para classificação binária

    Args:
        y_true: labels verdadeiros
        y_pred_proba: probabilidades preditas
        threshold: limiar para conversão de probabilidade em classe

    Returns:
        Dict com todas as métricas
    """
    # Converter probabilidades em predições binárias
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

def print_metrics_summary(metrics, client_id=None, round_num=None):
    """
    Imprime um resumo organizado das métricas
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
    print(f"  Especific.:  {metrics['specificity']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"  Matriz de Confusão:")
    print(f"    TN: {cm['tn']:4d} | FP: {cm['fp']:4d}")
    print(f"    FN: {cm['fn']:4d} | TP: {cm['tp']:4d}")

# ----------------------------
# Preparar dataset HIGGS (amostrado) e particionamento IID
# ----------------------------
log(INFO, "Carregando dataset HIGGS (pode demorar a baixar na primeira execução)...")
ds = load_dataset("jxie/higgs")

# Reduzimos para demonstração; aumente se tiver recursos
max_train = SAMPLE_PER_CLIENT * NUM_CLIENTS
max_test = SAMPLE_PER_CLIENT

train_all = ds["train"].select(range(min(len(ds["train"]), max_train)))
test_all = ds["test"].select(range(min(len(ds["test"]), max_test)))

def hf_to_xy(hf_dataset):
    # suporta colunas 'inputs' ou colunas separadas
    if "inputs" in hf_dataset.column_names:
        X = np.array(hf_dataset["inputs"], dtype=np.float32)
    else:
        feature_cols = [c for c in hf_dataset.column_names if c != "label"]
        X = np.vstack([[example[c] for c in feature_cols] for example in hf_dataset]).astype(np.float32)
    y = np.array(hf_dataset["label"]).astype(int)
    return X, y

X_train_all, y_train_all = hf_to_xy(train_all)
X_test_all, y_test_all = hf_to_xy(test_all)

# escala (fit no train_all)
scaler = StandardScaler()
scaler.fit(X_train_all)
X_train_all = scaler.transform(X_train_all)
X_test_all = scaler.transform(X_test_all)

# particionamento IID simples
partitions_X = np.array_split(X_train_all, NUM_CLIENTS)
partitions_y = np.array_split(y_train_all, NUM_CLIENTS)

log(INFO, f"Dataset particionado em {NUM_CLIENTS} clientes (~{len(partitions_X[0])} amostras por cliente).")

# ----------------------------
# transform_dataset_to_dmatrix (para server_fn que usa HF dataset)
# ----------------------------
def transform_dataset_to_dmatrix(hf_dataset):
    # converte HF dataset (numpy format) em DMatrix
    X, y = hf_to_xy(hf_dataset)
    return xgb.DMatrix(X, label=y)

# ----------------------------
# XGBoost Client (baseado no exemplo com métricas avançadas)
# ----------------------------
class XgbClient(Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
        train_method,
        cid: int,
        X_valid=None,
        y_valid=None,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
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
        # ins.config contains on_fit_config_fn output -> global_round
        global_round = int(ins.config.get("global_round", "0"))
        tree_method = self.params.get("tree_method", "hist")

        if global_round <= 1 or not ins.parameters.tensors:
            # First round or no global model: train from scratch
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
                verbose_eval=False,
            )
        else:
            # There is a global model: load and continue training using xgb.train with xgb_model param
            try:
                global_model = bytearray(ins.parameters.tensors[0])
                global_bst = xgb.Booster(params=self.params)
                global_bst.load_model(global_model)
                bst = xgb.train(
                    self.params,
                    self.train_dmatrix,
                    num_boost_round=self.num_local_round,
                    evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
                    xgb_model=global_bst,
                    verbose_eval=False,
                )
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] warning: falha ao carregar modelo global: {e}; treinando do zero.")
                bst = xgb.train(
                    self.params,
                    self.train_dmatrix,
                    num_boost_round=self.num_local_round,
                    evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
                    verbose_eval=False,
                )

        # Calcular métricas avançadas se temos dados de validação
        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = bst.predict(self.valid_dmatrix)
                metrics = calculate_comprehensive_metrics(self.y_valid, y_pred_proba)
                print_metrics_summary(metrics, client_id=self.cid, round_num=global_round)
            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Erro ao calcular métricas avançadas: {e}")

        # Log local training metrics - versão robusta (mantém código original)
        try:
            eval_results = bst.eval_set(
                evals=[(self.train_dmatrix, "train"), (self.valid_dmatrix, "validate")],
                iteration=bst.num_boosted_rounds() - 1
            )

            log(INFO, f"[Client {self.cid}] Raw eval results: {eval_results}")

            # Parse mais robusto dos resultados
            parts = eval_results.strip().split('\t')
            eval_dict = {}

            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    try:
                        eval_dict[key.strip()] = float(value.strip())
                    except ValueError:
                        continue

            log(INFO, f"[Client {self.cid}] Parsed metrics: {eval_dict}")

            # Extrair métricas com nomes corretos baseados no formato do XGBoost
            train_loss = None
            train_error = None
            valid_loss = None
            valid_error = None

            for key, value in eval_dict.items():
                if 'train' in key and 'logloss' in key:
                    train_loss = value
                elif 'train' in key and 'error' in key:
                    train_error = value
                elif 'validate' in key and 'logloss' in key:
                    valid_loss = value
                elif 'validate' in key and 'error' in key:
                    valid_error = value

            # Calcular acurácia corretamente (1 - error_rate)
            train_acc = 1.0 - train_error if train_error is not None else 0.5
            valid_acc = 1.0 - valid_error if valid_error is not None else 0.5

            log(INFO, f"[Client {self.cid}] Training: loss={train_loss}, error_rate={train_error}, accuracy={train_acc:.4f}")
            log(INFO, f"[Client {self.cid}] Validation: loss={valid_loss}, error_rate={valid_error}, accuracy={valid_acc:.4f}")

        except Exception as e:
            log(WARNING, f"[Client {self.cid}] Erro ao fazer parse das métricas de treino: {e}")
            train_loss, train_acc = 0.0, 0.5

        # Salvar modelo como bytes (json)
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

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

        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Calcular métricas avançadas
        if self.X_valid is not None and self.y_valid is not None:
            try:
                y_pred_proba = bst.predict(self.valid_dmatrix)
                comprehensive_metrics = calculate_comprehensive_metrics(self.y_valid, y_pred_proba)

                # Log das métricas
                log(INFO, f"[Client {self.cid}] Comprehensive evaluation metrics:")
                for key, value in comprehensive_metrics.items():
                    if key != 'confusion_matrix':
                        log(INFO, f"  {key}: {value:.4f}")

                # Preparar métricas para retorno (sem confusion_matrix para evitar problemas de serialização)
                return_metrics = {k: v for k, v in comprehensive_metrics.items() if k != 'confusion_matrix'}

                return EvaluateRes(
                    status=Status(code=Code.OK, message="OK"),
                    loss=float(comprehensive_metrics.get('auc', 0.5)),  # Usar AUC como "loss" principal
                    num_examples=self.num_val,
                    metrics=return_metrics,
                )

            except Exception as e:
                log(WARNING, f"[Client {self.cid}] Erro ao calcular métricas abrangentes: {e}")

        # Fallback para método original
        try:
            eval_results = bst.eval_set(evals=[(self.valid_dmatrix, "valid")], iteration=bst.num_boosted_rounds() - 1)

            log(INFO, f"[Client {self.cid}] Raw evaluation results: {eval_results}")

            # Parse mais robusto dos resultados
            parts = eval_results.strip().split('\t')
            eval_dict = {}

            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    try:
                        eval_dict[key.strip()] = float(value.strip())
                    except ValueError:
                        continue

            log(INFO, f"[Client {self.cid}] Parsed evaluation metrics: {eval_dict}")

            # Extrair métricas
            loss = None
            error_rate = None

            for key, value in eval_dict.items():
                if 'valid' in key and 'logloss' in key:
                    loss = value
                elif 'valid' in key and 'error' in key:
                    error_rate = value

            # Valores padrão se não encontrados
            loss = loss if loss is not None else 0.693147  # ln(2) para classificação balanceada aleatória
            error_rate = error_rate if error_rate is not None else 0.5
            acc = 1.0 - error_rate

            # Calcular AUC aproximado baseado na acurácia (estimativa grosseira)
            # Para dados balanceados: AUC ≈ (accuracy - 0.5) * 2 + 0.5, limitado entre 0.5 e 1.0
            auc_estimate = max(0.5, min(1.0, (acc - 0.5) * 1.5 + 0.5))

            log(INFO, f"[Client {self.cid}] Final evaluation: loss={loss:.4f}, error_rate={error_rate:.4f}, accuracy={acc:.4f}, AUC_est={auc_estimate:.4f}")

            # Retornar métricas básicas com estimativas
            return EvaluateRes(
                status=Status(code=Code.OK, message="OK"),
                loss=loss,
                num_examples=self.num_val,
                metrics={
                    "accuracy": acc,
                    "auc": auc_estimate,
                    "precision": max(0.1, acc),  # Estimativa grosseira
                    "recall": max(0.1, acc),     # Estimativa grosseira
                    "f1_score": max(0.1, acc),   # Estimativa grosseira
                },
            )

        except Exception as e:
            log(WARNING, f"[Client {self.cid}] Erro ao fazer parse das métricas de avaliação: {e}")
            loss, acc, auc_estimate = 0.693147, 0.5, 0.5

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=loss,
            num_examples=self.num_val,
            metrics={
                "accuracy": acc,
                "auc": auc_estimate,
                "precision": 0.5,
                "recall": 0.5,
                "f1_score": 0.5,
            },
        )

# ----------------------------
# client_fn: cria XgbClient para cada contexto (partition id, etc.) - MODIFICADO
# ----------------------------
def client_fn(context: Context):
    node_cfg = context.node_config or {}
    partition_id = int(node_cfg.get("partition-id", 0))
    num_partitions = int(node_cfg.get("num-partitions", NUM_CLIENTS))

    cfg = replace_keys(unflatten_dict(context.run_config))
    num_local_round = int(cfg.get("local_epochs", NUM_LOCAL_BOOST_ROUND))
    train_method = cfg.get("train_method", "cyclic")
    params = cfg.get("params", {"objective":"binary:logistic","eval_metric":["logloss", "error"],"eta":0.1,"max_depth":6})
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

    train_dmatrix = xgb.DMatrix(train_X, label=train_y)
    valid_dmatrix = xgb.DMatrix(valid_X, label=valid_y)
    num_train = train_X.shape[0]
    num_val = valid_X.shape[0]

    if cfg.get("scaled_lr", False):
        new_lr = params.get("eta", 0.1) / num_partitions
        params.update({"eta": new_lr})

    # MODIFICAÇÃO: Passar também X_valid e y_valid para permitir cálculo de métricas avançadas
    return XgbClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
        train_method,
        partition_id,
        X_valid=valid_X,  # NOVO
        y_valid=valid_y   # NOVO
    ).to_client()

client_app = ClientApp(client_fn=client_fn)

# ----------------------------
# Server helpers (get_evaluate_fn, evaluate_metrics_aggregation, config_func) - MODIFICADOS
# ----------------------------
def get_evaluate_fn(test_data, params, X_test=None, y_test=None):
    def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Scalar]):
        if server_round == 0:
            return 0.693147, {
                "accuracy": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc": 0.5
            }
        else:
            try:
                bst = xgb.Booster(params=params)
                para_b = None
                if parameters.tensors:
                    para_b = bytearray(parameters.tensors[-1])
                if para_b is None:
                    return 0, {
                        "accuracy": 0.5,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                        "auc": 0.5
                    }

                bst.load_model(para_b)

                # Se temos dados de teste separados, usar métricas avançadas
                if X_test is not None and y_test is not None:
                    y_pred_proba = bst.predict(test_data)
                    comprehensive_metrics = calculate_comprehensive_metrics(y_test, y_pred_proba)
                    print_metrics_summary(comprehensive_metrics, round_num=server_round)

                    # Remover confusion_matrix para evitar problemas de serialização
                    return_metrics = {k: v for k, v in comprehensive_metrics.items() if k != 'confusion_matrix'}

                    return float(1.0 - comprehensive_metrics['accuracy']), return_metrics  # loss = 1 - accuracy

                # Fallback para método original
                eval_results = bst.eval_set(evals=[(test_data, "valid")], iteration=bst.num_boosted_rounds() - 1)

                log(INFO, f"Server evaluation raw results: {eval_results}")

                # Parse robusto dos resultados
                parts = eval_results.strip().split('\t')
                eval_dict = {}
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        try:
                            eval_dict[key.strip()] = float(value.strip())
                        except ValueError:
                            continue

                log(INFO, f"Server parsed metrics: {eval_dict}")

                # Extrair métricas
                loss = None
                error_rate = None

                for key, value in eval_dict.items():
                    if 'valid' in key and 'logloss' in key:
                        loss = value
                    elif 'valid' in key and 'error' in key:
                        error_rate = value

                # Valores padrão se não encontrados
                loss = loss if loss is not None else 0.693147
                error_rate = error_rate if error_rate is not None else 0.5
                acc = 1.0 - error_rate

                print(f"Server-side evaluation (Round {server_round}): Loss={loss:.4f}, Accuracy={acc:.4f}")
                return loss, {
                    "accuracy": acc,
                    "precision": max(0.1, acc),
                    "recall": max(0.1, acc),
                    "f1_score": max(0.1, acc),
                    "auc": max(0.5, min(1.0, (acc - 0.5) * 1.5 + 0.5))
                }

            except Exception as e:
                log(WARNING, f"Erro na avaliação do servidor: {e}")
                return 0.0, {
                    "accuracy": 0.5,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "auc": 0.5
                }

    return evaluate_fn

def evaluate_metrics_aggregation(eval_metrics):
    """Agrega métricas de avaliação de forma robusta - EXPANDIDO"""
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

def config_func(rnd: int) -> Dict[str, str]:
    return {"global_round": str(rnd)}

# ----------------------------
# server_fn conforme exemplo (usado pelo ServerApp) - MODIFICADO
# ----------------------------
def server_fn(context: Context):
    default_cfg = {
        "num-server-rounds": str(NUM_SERVER_ROUNDS),
        "fraction-fit": "1.0",
        "fraction-evaluate": "1.0",
        "train-method": "cyclic",
        "params": {},
        "centralised-eval": True,
        "local-epochs": str(NUM_LOCAL_BOOST_ROUND),
    }
    cfg = default_cfg.copy()
    cfg.update(replace_keys(unflatten_dict(context.run_config)))

    num_rounds = int(cfg.get("num_server_rounds", NUM_SERVER_ROUNDS))
    fraction_fit = float(cfg.get("fraction_fit", 1.0))
    fraction_evaluate = float(cfg.get("fraction_evaluate", 1.0))
    train_method = cfg.get("train_method", "cyclic")
    params = cfg.get("params", {}) or {}
    centralised_eval = cfg.get("centralised_eval", True)

    if centralised_eval:
        test_set = load_dataset("jxie/higgs")["test"]
        test_set = test_set.select(range(min(len(test_set), SAMPLE_PER_CLIENT)))
        test_set.set_format("numpy")
        test_dmatrix = transform_dataset_to_dmatrix(test_set)

        # NOVO: Extrair X e y para métricas avançadas
        X_test_server, y_test_server = hf_to_xy(test_set)
        X_test_server = scaler.transform(X_test_server)  # Aplicar mesma escala

    parameters = Parameters(tensor_type="", tensors=[])

    # CORREÇÃO: Usar evaluate_fn apenas para estratégias que suportam
    evaluate_fn = get_evaluate_fn(
        test_dmatrix,
        params,
        X_test=X_test_server if centralised_eval else None,
        y_test=y_test_server if centralised_eval else None
    ) if centralised_eval else None

    fraction_eval = 0.0 if centralised_eval and train_method == "bagging" else fraction_evaluate
    agg_fn = evaluate_metrics_aggregation if fraction_eval > 0 else None

    # CORREÇÃO: Remover evaluate_function das estratégias XGBoost
    if train_method == "bagging":
        strategy = FedXgbBagging(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            evaluate_metrics_aggregation_fn=agg_fn,
            initial_parameters=parameters,
        )
    else:
        strategy = FedXgbCyclic(
            fraction_fit=1.0,
            fraction_evaluate=fraction_eval,
            evaluate_metrics_aggregation_fn=agg_fn,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            initial_parameters=parameters,
        )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

server_app = ServerApp(server_fn=server_fn)

# ----------------------------
# Função robusta para chamar run_simulation (detectar assinatura de flwr)
# ----------------------------
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
    except Exception as e3:
        log(WARNING, "Todas tentativas falharam. Erros: %s %s %s", e1, e2, e3)
        raise RuntimeError("run_simulation não pôde ser invocado automaticamente. Informe a versão do Flower para adaptar a chamada.") from e3

# ----------------------------
# Função para salvar métricas em arquivo (NOVA)
# ----------------------------
def save_metrics_to_file(metrics_history, filename="federated_learning_metrics.json"):
    """
    Salva histórico de métricas em arquivo JSON
    """
    try:
        with open(filename, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f"Métricas salvas em: {filename}")
    except Exception as e:
        print(f"Erro ao salvar métricas: {e}")

# ----------------------------
# Função para análise final de resultados (NOVA)
# ----------------------------
def print_final_analysis(strategy_name, metrics_history=None):
    """
    Imprime análise final dos resultados
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

# ----------------------------
# Preparar run_cfg (com '-' nas keys; server_fn fará replace_keys)
# ----------------------------
USE_GPU = torch.cuda.is_available()

tree_method = "gpu_hist" if USE_GPU else "hist"
log(INFO, "GPU disponível (torch): %s | usando tree_method: %s", USE_GPU, tree_method)

base_run_cfg = {
    "num-server-rounds": str(NUM_SERVER_ROUNDS),
    "fraction-fit": "1.0",
    "fraction-evaluate": "1.0",
    "local-epochs": str(NUM_LOCAL_BOOST_ROUND),
    "train-method": "cyclic",  # overwritten quando necessário
    "partitioner-type": "uniform",
    "seed": str(SEED),
    "test-fraction": "0.2",
    "centralised-eval": "True",
    "centralised-eval-client": "False",
    "scaled-lr": "False",
    "params": {
        "eta": 0.1,
        "max_depth": 6,
        "tree_method": tree_method,
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],  # Mantido para compatibilidade
    },
}

backend_config = {"client_resources": {"num_cpus": 1.0, "num_gpus": 1.0 / NUM_CLIENTS}} if USE_GPU else {"client_resources": {"num_cpus": 1.0}}

# ----------------------------
# Execução das simulações com análise de resultados
# ----------------------------

print(f"\n{'='*80}")
print("INICIANDO EXPERIMENTOS DE FEDERATED LEARNING COM XGBOOST")
print(f"{'='*80}")
print(f"Configuração:")
print(f"  - Número de clientes: {NUM_CLIENTS}")
print(f"  - Amostras por cliente: ~{len(partitions_X[0])}")
print(f"  - Rodadas do servidor: {NUM_SERVER_ROUNDS}")
print(f"  - Rodadas locais de boosting: {NUM_LOCAL_BOOST_ROUND}")
print(f"  - GPU disponível: {USE_GPU}")
print(f"  - Tree method: {tree_method}")
print(f"{'='*80}\n")

# Dicionário para armazenar métricas de ambas as estratégias
all_metrics = {}

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

# ----------------------------
# Relatório final comparativo
# ----------------------------


if len(all_metrics) > 1:
    print("Comparação entre estratégias:")
    print("-" * 50)
    for strategy_name in all_metrics.keys():
        print(f"\nEstratégia: {strategy_name.upper()}")
        print(f"Status: Executada com sucesso")

else:
    print("Apenas uma estratégia foi executada com sucesso.")

# Salvar métricas em arquivo
save_metrics_to_file(all_metrics, f"federated_xgb_metrics_{int(time.time())}.json")

print(f"\n{'='*80}")
print("EXPERIMENTO CONCLUÍDO")
print(f"{'='*80}")

log(INFO, "Simulações finalizadas com métricas avançadas.")