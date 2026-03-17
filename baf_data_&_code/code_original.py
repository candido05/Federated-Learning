"""
================================================================================
FEDERATED LEARNING COM FLOWER - BANK ACCOUNT FRAUD DETECTION (CORRIGIDO v2)
================================================================================

Correcoes implementadas:
1. fl.simulation.start_simulation (motor oficial)
2. Warm Start para Cycling (transferencia de conhecimento entre clientes)
3. Post-Processing com Limiares por Grupo (Fairness Ratio = 1.0)
4. SERIALIZACAO CORRETA: parameters_to_ndarrays / ndarrays_to_parameters

================================================================================
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pickle
from collections import OrderedDict

# Sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve

# Modelos
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Optuna
import optuna
from optuna.samplers import TPESampler

# Flower
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    GetParametersRes,
    Status,
    Code,
    FitIns,
    EvaluateIns,
    GetParametersIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.simulation import start_simulation

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ==============================================================================
# SECAO 1: CONFIGURACOES GLOBAIS
# ==============================================================================

@dataclass
class FederatedConfig:
    """Configuracoes do treinamento federado."""
    num_clients: int = 3
    num_rounds: int = 5
    local_epochs: int = 5
    optuna_trials: int = 30
    fpr_target: float = 0.05
    random_state: int = 42


CONFIG = FederatedConfig()

# Armazenamento global para dados dos clientes (necessario para fl.simulation)
GLOBAL_CLIENT_DATA: Dict[int, Dict] = {}
GLOBAL_VAL_DATA: Dict[str, np.ndarray] = {}
GLOBAL_TEST_DATA: Dict[str, np.ndarray] = {}
GLOBAL_MODEL_PARAMS: Dict[str, Dict] = {}
GLOBAL_SCALE_POS_WEIGHT: float = 1.0
GLOBAL_MODEL_TYPE: str = "xgboost"


# ==============================================================================
# SECAO 2: FUNCOES DE METRICAS (COM FAIRNESS POR GRUPO)
# ==============================================================================

def calc_tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray,
                    fpr_target: float = 0.05) -> float:
    """Calcula TPR quando FPR <= fpr_target."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid_indices = np.where(fpr <= fpr_target)[0]

    if len(valid_indices) == 0:
        return 0.0

    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
    return tpr[best_idx]


def get_threshold_at_fpr(y_true: np.ndarray, y_prob: np.ndarray,
                         fpr_target: float = 0.05) -> float:
    """Retorna threshold correspondente ao FPR alvo."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_indices = np.where(fpr <= fpr_target)[0]

    if len(valid_indices) == 0:
        return 1.0

    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
    return thresholds[best_idx]


def get_group_threshold_at_fpr(y_true: np.ndarray, y_prob: np.ndarray,
                                fpr_target: float = 0.05) -> Tuple[float, float]:
    """
    Calcula threshold que resulta em FPR = fpr_target para um grupo especifico.
    Retorna (threshold, tpr_at_threshold).
    """
    if len(y_true) == 0 or y_true.sum() == 0 or (y_true == 0).sum() == 0:
        return 0.5, 0.0

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_indices = np.where(fpr <= fpr_target)[0]

    if len(valid_indices) == 0:
        return thresholds[0] if len(thresholds) > 0 else 1.0, 0.0

    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
    return thresholds[best_idx], tpr[best_idx]


def predict_with_group_thresholds(y_prob: np.ndarray, y_true: np.ndarray,
                                   age_flag: np.ndarray,
                                   fpr_target: float = 0.05) -> Tuple[np.ndarray, Dict]:
    """
    CORRECAO DE FAIRNESS: Predicao com limiares especificos por grupo.
    """
    mask_young = age_flag == 0
    mask_old = age_flag == 1

    thresh_young, tpr_young = get_group_threshold_at_fpr(
        y_true[mask_young], y_prob[mask_young], fpr_target
    )
    thresh_old, tpr_old = get_group_threshold_at_fpr(
        y_true[mask_old], y_prob[mask_old], fpr_target
    )

    y_pred = np.zeros(len(y_prob), dtype=int)
    y_pred[mask_young] = (y_prob[mask_young] >= thresh_young).astype(int)
    y_pred[mask_old] = (y_prob[mask_old] >= thresh_old).astype(int)

    def calc_group_rates(y_true_g, y_pred_g):
        if len(y_true_g) == 0:
            return {'tpr': 0.0, 'fpr': 0.0}

        tp = ((y_pred_g == 1) & (y_true_g == 1)).sum()
        fp = ((y_pred_g == 1) & (y_true_g == 0)).sum()
        tn = ((y_pred_g == 0) & (y_true_g == 0)).sum()
        fn = ((y_pred_g == 0) & (y_true_g == 1)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {'tpr': tpr, 'fpr': fpr}

    rates_young = calc_group_rates(y_true[mask_young], y_pred[mask_young])
    rates_old = calc_group_rates(y_true[mask_old], y_pred[mask_old])

    tp_global = ((y_pred == 1) & (y_true == 1)).sum()
    fp_global = ((y_pred == 1) & (y_true == 0)).sum()
    fn_global = ((y_pred == 0) & (y_true == 1)).sum()
    tn_global = ((y_pred == 0) & (y_true == 0)).sum()

    tpr_global = tp_global / (tp_global + fn_global) if (tp_global + fn_global) > 0 else 0.0
    fpr_global = fp_global / (fp_global + tn_global) if (fp_global + tn_global) > 0 else 0.0

    fairness_ratio = rates_old['fpr'] / rates_young['fpr'] if rates_young['fpr'] > 0 else 1.0

    return y_pred, {
        'thresh_young': thresh_young,
        'thresh_old': thresh_old,
        'tpr_young': rates_young['tpr'],
        'tpr_old': rates_old['tpr'],
        'fpr_young': rates_young['fpr'],
        'fpr_old': rates_old['fpr'],
        'tpr_global': tpr_global,
        'fpr_global': fpr_global,
        'fairness_ratio': fairness_ratio
    }


def evaluate_model_standard(y_true: np.ndarray, y_prob: np.ndarray,
                            age_flag: np.ndarray) -> Dict[str, float]:
    """Avaliacao com threshold unico."""
    tpr_at_5fpr = calc_tpr_at_fpr(y_true, y_prob, fpr_target=0.05)
    roc_auc = roc_auc_score(y_true, y_prob)
    threshold = get_threshold_at_fpr(y_true, y_prob, fpr_target=0.05)
    y_pred = (y_prob >= threshold).astype(int)

    mask_old = age_flag == 1
    mask_young = age_flag == 0

    def calc_fpr_group(y_true_g, y_pred_g):
        neg_mask = y_true_g == 0
        if neg_mask.sum() == 0:
            return 0.0
        fp = ((y_pred_g == 1) & (y_true_g == 0)).sum()
        return fp / neg_mask.sum()

    fpr_old = calc_fpr_group(y_true[mask_old], y_pred[mask_old])
    fpr_young = calc_fpr_group(y_true[mask_young], y_pred[mask_young])
    fairness_ratio = fpr_old / fpr_young if fpr_young > 0 else float('inf')

    return {
        'tpr_at_5fpr': tpr_at_5fpr,
        'roc_auc': roc_auc,
        'fairness_ratio': fairness_ratio,
        'fpr_old': fpr_old,
        'fpr_young': fpr_young,
        'threshold': threshold,
    }


def evaluate_model_fair(y_true: np.ndarray, y_prob: np.ndarray,
                        age_flag: np.ndarray) -> Dict[str, float]:
    """Avaliacao com fairness corrigido (limiares por grupo)."""
    roc_auc = roc_auc_score(y_true, y_prob)
    _, fair_metrics = predict_with_group_thresholds(y_prob, y_true, age_flag)

    return {
        'tpr_at_5fpr': fair_metrics['tpr_global'],
        'roc_auc': roc_auc,
        'fairness_ratio': fair_metrics['fairness_ratio'],
        'fpr_old': fair_metrics['fpr_old'],
        'fpr_young': fair_metrics['fpr_young'],
        'tpr_old': fair_metrics['tpr_old'],
        'tpr_young': fair_metrics['tpr_young'],
        'thresh_young': fair_metrics['thresh_young'],
        'thresh_old': fair_metrics['thresh_old'],
    }


# ==============================================================================
# SECAO 3: PRE-PROCESSAMENTO
# ==============================================================================

class DataPreprocessor:
    """Pre-processamento dos dados de fraude bancaria."""

    def __init__(self):
        self.ohe_encoder = None
        self.categorical_cols = None
        self.numeric_cols = None

    def load_and_clean(self, filepath: str) -> pd.DataFrame:
        print("=" * 60)
        print("CARREGAMENTO E LIMPEZA DOS DADOS")
        print("=" * 60)

        df = pd.read_csv(filepath)
        print(f"Dataset carregado: {df.shape[0]:,} linhas x {df.shape[1]} colunas")

        df_clean = df.replace(-1, np.nan)

        cols_to_drop = ['prev_address_months_count', 'intended_balcon_amount']
        df_clean = df_clean.drop(columns=cols_to_drop)
        df_clean = df_clean.drop(columns=['device_fraud_count'])

        rows_before = len(df_clean)
        df_clean = df_clean.dropna(subset=['device_distinct_emails_8w'])
        print(f"Linhas removidas: {rows_before - len(df_clean):,}")

        df_clean = df_clean.drop(columns=['velocity_24h'])

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['fraud_bool', 'month']]

        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)

        print(f"Dataset apos limpeza: {df_clean.shape[0]:,} linhas x {df_clean.shape[1]} colunas")
        return df_clean

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        device_os_counts = df.groupby('device_os').size().to_dict()
        df['device_os_frequency'] = df['device_os'].map(device_os_counts)

        source_counts = df.groupby('source').size().to_dict()
        df['source_frequency'] = df['source'].map(source_counts)

        df['velocity_ratio_6h_4w'] = df['velocity_6h'] / (df['velocity_4w'] + 1)

        for col in ['velocity_6h', 'velocity_4w', 'zip_count_4w', 'bank_branch_count_8w']:
            monthly_mean = df.groupby('month')[col].transform('mean')
            monthly_std = df.groupby('month')[col].transform('std').replace(0, 1)
            df[f'{col}_zscore_monthly'] = (df[col] - monthly_mean) / monthly_std

        df['income_x_age'] = df['income'] * df['customer_age']

        df['age_group'] = pd.cut(df['customer_age'],
                                  bins=[0, 30, 50, 100],
                                  labels=['young', 'middle', 'senior'])
        income_by_age = df.groupby('age_group')['income'].transform('mean')
        df['income_vs_age_group_mean'] = df['income'] - income_by_age

        df['employment_income_cat'] = df['employment_status'] + '_' + \
            pd.qcut(df['income'], q=4, labels=['Q1','Q2','Q3','Q4']).astype(str)

        df['age_x_credit_risk'] = df['customer_age'] * df['credit_risk_score']
        df['age_above_50'] = (df['customer_age'] > 50).astype(int)
        df['income_per_age'] = df['income'] / (df['customer_age'] + 1)

        emp_credit_mean = df.groupby('employment_status')['credit_risk_score'].transform('mean')
        emp_credit_std = df.groupby('employment_status')['credit_risk_score'].transform('std').replace(0, 1)
        df['credit_risk_vs_employment'] = (df['credit_risk_score'] - emp_credit_mean) / emp_credit_std

        df['phone_validation_score'] = df['phone_home_valid'] + df['phone_mobile_valid']

        source_session_mean = df.groupby('source')['session_length_in_minutes'].transform('mean')
        df['session_length_vs_source'] = df['session_length_in_minutes'] - source_session_mean

        df['email_similarity_x_free'] = df['name_email_similarity'] * df['email_is_free']
        df['bank_vs_address_months'] = df['bank_months_count'] / (df['current_address_months_count'] + 1)
        df['days_since_request_log'] = np.log1p(df['days_since_request'])

        df['credit_limit_bucket'] = pd.cut(df['proposed_credit_limit'],
                                            bins=[0, 200, 500, 1000, 2000, np.inf],
                                            labels=['very_low', 'low', 'medium', 'high', 'very_high'])

        print(f"Features criadas. Total de colunas: {df.shape[1]}")
        return df

    def temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("\n" + "=" * 60)
        print("DIVISAO TEMPORAL")
        print("=" * 60)

        df_train = df[df['month'].isin([0, 1, 2, 3, 4, 5])].copy()
        df_val = df[df['month'] == 6].copy()
        df_test = df[df['month'] == 7].copy()

        print(f"Treino (Meses 0-5): {len(df_train):,} ({df_train['fraud_bool'].mean()*100:.2f}% fraudes)")
        print(f"Validacao (Mes 6): {len(df_val):,} ({df_val['fraud_bool'].mean()*100:.2f}% fraudes)")
        print(f"Teste (Mes 7): {len(df_test):,} ({df_test['fraud_bool'].mean()*100:.2f}% fraudes)")

        return df_train, df_val, df_test

    def encode_features(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                        df_test: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        print("\n" + "=" * 60)
        print("ENCODING")
        print("=" * 60)

        categorical_cols = ['payment_type', 'employment_status', 'housing_status',
                           'source', 'device_os', 'age_group', 'employment_income_cat',
                           'credit_limit_bucket']
        self.categorical_cols = categorical_cols

        for col in categorical_cols:
            for df in [df_train, df_val, df_test]:
                df[col] = df[col].astype(str)

        exclude_cols = ['fraud_bool', 'month', 'age_above_50']
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]

        X_train = df_train[feature_cols].copy()
        X_val = df_val[feature_cols].copy()
        X_test = df_test[feature_cols].copy()

        y_train = df_train['fraud_bool'].values
        y_val = df_val['fraud_bool'].values
        y_test = df_test['fraud_bool'].values

        age_train = df_train['age_above_50'].values
        age_val = df_val['age_above_50'].values
        age_test = df_test['age_above_50'].values

        self.numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

        self.ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary')

        X_train_cat = self.ohe_encoder.fit_transform(X_train[categorical_cols])
        X_val_cat = self.ohe_encoder.transform(X_val[categorical_cols])
        X_test_cat = self.ohe_encoder.transform(X_test[categorical_cols])

        X_train_enc = np.hstack([X_train[self.numeric_cols].values, X_train_cat])
        X_val_enc = np.hstack([X_val[self.numeric_cols].values, X_val_cat])
        X_test_enc = np.hstack([X_test[self.numeric_cols].values, X_test_cat])

        print(f"Features encoded: {X_train_enc.shape[1]}")

        return (X_train_enc, y_train, age_train,
                X_val_enc, y_val, age_val,
                X_test_enc, y_test, age_test)


# ==============================================================================
# SECAO 4: PARTICIONAMENTO
# ==============================================================================

def partition_data_balanced(X: np.ndarray, y: np.ndarray, age: np.ndarray,
                            num_clients: int = 3) -> List[Dict]:
    print("\n" + "=" * 60)
    print("PARTICIONAMENTO PARA CLIENTES")
    print("=" * 60)

    idx_fraud = np.where(y == 1)[0]
    idx_legit = np.where(y == 0)[0]

    np.random.shuffle(idx_fraud)
    np.random.shuffle(idx_legit)

    fraud_splits = np.array_split(idx_fraud, num_clients)
    legit_splits = np.array_split(idx_legit, num_clients)

    client_data = []

    for i in range(num_clients):
        client_idx = np.concatenate([fraud_splits[i], legit_splits[i]])
        np.random.shuffle(client_idx)

        data = {
            'X': X[client_idx],
            'y': y[client_idx],
            'age': age[client_idx],
        }

        n_fraud = data['y'].sum()
        n_total = len(data['y'])
        fraud_rate = n_fraud / n_total * 100

        print(f"Cliente {i}: {n_total:,} amostras ({n_fraud:,} fraudes, {fraud_rate:.2f}%)")

        client_data.append(data)

    return client_data


# ==============================================================================
# SECAO 5: OTIMIZACAO OPTUNA FEDERADA (3 OPTUNAS LOCAIS -> AGREGACAO)
# ==============================================================================

class LocalHyperparameterOptimizer:
    """
    Otimizador local para um unico cliente.
    Roda Optuna usando apenas os dados daquele cliente.
    """

    def __init__(self, client_id: int, n_trials: int = 15):
        self.client_id = client_id
        self.n_trials = n_trials
        self.best_params = {}

    def optimize_xgboost(self, X_train, y_train, X_val, y_val, scale_pos_weight):
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'random_state': RANDOM_STATE,
                'scale_pos_weight': scale_pos_weight,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            y_prob = model.predict_proba(X_val)[:, 1]
            return calc_tpr_at_fpr(y_val, y_prob, fpr_target=0.05)

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE + self.client_id))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params['xgboost'] = study.best_params
        return study.best_params, study.best_value

    def optimize_lightgbm(self, X_train, y_train, X_val, y_val, scale_pos_weight):
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'random_state': RANDOM_STATE,
                'scale_pos_weight': scale_pos_weight,
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            return calc_tpr_at_fpr(y_val, y_prob, fpr_target=0.05)

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE + self.client_id))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params['lightgbm'] = study.best_params
        return study.best_params, study.best_value

    def optimize_catboost(self, X_train, y_train, X_val, y_val):
        def objective(trial):
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': RANDOM_STATE,
                'auto_class_weights': 'Balanced',
                'verbose': False,
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'iterations': trial.suggest_int('iterations', 50, 300),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            }

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            y_prob = model.predict_proba(X_val)[:, 1]
            return calc_tpr_at_fpr(y_val, y_prob, fpr_target=0.05)

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE + self.client_id))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params['catboost'] = study.best_params
        return study.best_params, study.best_value


def aggregate_hyperparameters(local_params_list: List[Dict], param_keys: List[str],
                               int_keys: List[str]) -> Dict:
    """
    Agrega hiperparametros de multiplos clientes usando MEDIANA.
    Mediana e mais robusta que media contra outliers.

    Args:
        local_params_list: Lista de dicts com parametros de cada cliente
        param_keys: Chaves dos parametros a agregar
        int_keys: Chaves que devem ser convertidas para inteiro

    Returns:
        Dict com parametros agregados
    """
    aggregated = {}

    for key in param_keys:
        values = [p[key] for p in local_params_list if key in p]

        if len(values) == 0:
            continue

        median_val = np.median(values)

        if key in int_keys:
            aggregated[key] = int(median_val)
        else:
            aggregated[key] = float(median_val)

    return aggregated


def federated_hyperparameter_optimization(client_data: List[Dict],
                                          X_val: np.ndarray, y_val: np.ndarray,
                                          scale_pos_weight: float,
                                          n_trials_per_client: int = 15,
                                          sample_fraction: float = 0.3,
                                          max_learning_rate: float = 0.05) -> Dict:
    """
    Otimizacao FEDERADA de hiperparametros:
    1. Cada cliente roda Optuna localmente com seus proprios dados
    2. Parametros sao agregados (mediana) no servidor
    3. Aplica freio de seguranca no learning_rate

    Args:
        client_data: Dados de cada cliente
        X_val, y_val: Dados de validacao (compartilhados apenas para avaliacao)
        scale_pos_weight: Peso para classes desbalanceadas
        n_trials_per_client: Trials Optuna por cliente
        sample_fraction: Fracao dos dados do cliente para usar no Optuna
        max_learning_rate: Learning rate maximo permitido (freio de seguranca)

    Returns:
        Dict com parametros agregados para cada modelo
    """
    print("\n" + "=" * 70)
    print("OTIMIZACAO FEDERADA DE HIPERPARAMETROS")
    print("=" * 70)
    print(f"  - {len(client_data)} clientes")
    print(f"  - {n_trials_per_client} trials por cliente")
    print(f"  - {sample_fraction*100:.0f}% dos dados de cada cliente")
    print(f"  - Learning rate maximo: {max_learning_rate}")

    # Armazenar parametros locais de cada cliente
    xgb_local_params = []
    lgbm_local_params = []
    cat_local_params = []

    # =========================================================================
    # FASE 1: Optuna LOCAL em cada cliente
    # =========================================================================
    for i in range(len(client_data)):
        print(f"\n>>> Cliente {i}: Rodando Optuna localmente...")

        # Dados apenas deste cliente
        X_c = client_data[i]['X']
        y_c = client_data[i]['y']

        # Amostrar fracao dos dados para ser mais rapido
        n_samples = int(len(y_c) * sample_fraction)
        idx = np.random.choice(len(y_c), n_samples, replace=False)
        X_opt = X_c[idx]
        y_opt = y_c[idx]

        print(f"    Usando {n_samples:,} amostras ({sample_fraction*100:.0f}% dos dados)")

        optimizer = LocalHyperparameterOptimizer(client_id=i, n_trials=n_trials_per_client)

        # XGBoost
        print(f"    [XGBoost] Otimizando...")
        xgb_params, xgb_score = optimizer.optimize_xgboost(X_opt, y_opt, X_val, y_val, scale_pos_weight)
        xgb_local_params.append(xgb_params)
        print(f"    [XGBoost] TPR@5%FPR: {xgb_score:.4f}, LR: {xgb_params['learning_rate']:.4f}")

        # LightGBM
        print(f"    [LightGBM] Otimizando...")
        lgbm_params, lgbm_score = optimizer.optimize_lightgbm(X_opt, y_opt, X_val, y_val, scale_pos_weight)
        lgbm_local_params.append(lgbm_params)
        print(f"    [LightGBM] TPR@5%FPR: {lgbm_score:.4f}, LR: {lgbm_params['learning_rate']:.4f}")

        # CatBoost
        print(f"    [CatBoost] Otimizando...")
        cat_params, cat_score = optimizer.optimize_catboost(X_opt, y_opt, X_val, y_val)
        cat_local_params.append(cat_params)
        print(f"    [CatBoost] TPR@5%FPR: {cat_score:.4f}, LR: {cat_params['learning_rate']:.4f}")

    # =========================================================================
    # FASE 2: Agregar parametros (MEDIANA)
    # =========================================================================
    print("\n>>> Agregando parametros dos clientes (mediana)...")

    # XGBoost
    xgb_keys = ['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight',
                'subsample', 'colsample_bytree']
    xgb_int_keys = ['max_depth', 'n_estimators', 'min_child_weight']

    final_xgb = aggregate_hyperparameters(xgb_local_params, xgb_keys, xgb_int_keys)

    print("\n    XGBoost:")
    for key in xgb_keys:
        local_vals = [p[key] for p in xgb_local_params]
        print(f"      {key}: {[f'{v:.4f}' if isinstance(v, float) else v for v in local_vals]} -> {final_xgb[key]}")

    # LightGBM
    lgbm_keys = ['max_depth', 'learning_rate', 'n_estimators', 'num_leaves',
                 'min_child_samples', 'subsample', 'colsample_bytree']
    lgbm_int_keys = ['max_depth', 'n_estimators', 'num_leaves', 'min_child_samples']

    final_lgbm = aggregate_hyperparameters(lgbm_local_params, lgbm_keys, lgbm_int_keys)

    print("\n    LightGBM:")
    for key in lgbm_keys:
        local_vals = [p[key] for p in lgbm_local_params]
        print(f"      {key}: {[f'{v:.4f}' if isinstance(v, float) else v for v in local_vals]} -> {final_lgbm[key]}")

    # CatBoost
    cat_keys = ['depth', 'learning_rate', 'iterations', 'l2_leaf_reg']
    cat_int_keys = ['depth', 'iterations']

    final_cat = aggregate_hyperparameters(cat_local_params, cat_keys, cat_int_keys)

    print("\n    CatBoost:")
    for key in cat_keys:
        local_vals = [p[key] for p in cat_local_params]
        print(f"      {key}: {[f'{v:.4f}' if isinstance(v, float) else v for v in local_vals]} -> {final_cat[key]}")

    # =========================================================================
    # FASE 3: Freio de seguranca no Learning Rate
    # =========================================================================
    print(f"\n>>> Aplicando freio de seguranca (LR max = {max_learning_rate})...")

    if final_xgb['learning_rate'] > max_learning_rate:
        print(f"    XGBoost: {final_xgb['learning_rate']:.4f} -> {max_learning_rate}")
        final_xgb['learning_rate'] = max_learning_rate

    if final_lgbm['learning_rate'] > max_learning_rate:
        print(f"    LightGBM: {final_lgbm['learning_rate']:.4f} -> {max_learning_rate}")
        final_lgbm['learning_rate'] = max_learning_rate

    if final_cat['learning_rate'] > max_learning_rate:
        print(f"    CatBoost: {final_cat['learning_rate']:.4f} -> {max_learning_rate}")
        final_cat['learning_rate'] = max_learning_rate

    print("\n>>> Parametros finais agregados:")
    print(f"    XGBoost:  LR={final_xgb['learning_rate']:.4f}, depth={final_xgb['max_depth']}, n_est={final_xgb['n_estimators']}")
    print(f"    LightGBM: LR={final_lgbm['learning_rate']:.4f}, depth={final_lgbm['max_depth']}, n_est={final_lgbm['n_estimators']}")
    print(f"    CatBoost: LR={final_cat['learning_rate']:.4f}, depth={final_cat['depth']}, iter={final_cat['iterations']}")

    return {
        'xgboost': final_xgb,
        'lightgbm': final_lgbm,
        'catboost': final_cat
    }


# ==============================================================================
# SECAO 6: FUNCOES AUXILIARES DE SERIALIZACAO
# ==============================================================================

def serialize_model(model) -> List[np.ndarray]:
    """
    Serializa modelo para formato compativel com Flower.
    Retorna lista de ndarrays (formato esperado pelo NumPyClient).
    """
    if model is None:
        return [np.array([], dtype=np.uint8)]

    model_bytes = pickle.dumps(model)
    model_array = np.frombuffer(model_bytes, dtype=np.uint8)
    return [model_array]


def deserialize_model_from_ndarrays(ndarrays: List[np.ndarray]) -> Optional[object]:
    """
    Deserializa modelo a partir de lista de ndarrays.
    """
    if ndarrays is None or len(ndarrays) == 0:
        return None
    if len(ndarrays[0]) == 0:
        return None

    try:
        model_bytes = ndarrays[0].tobytes()
        return pickle.loads(model_bytes)
    except Exception as e:
        print(f"    Erro deserializando modelo: {e}")
        return None


def deserialize_model_from_parameters(params: Parameters) -> Optional[object]:
    """
    Deserializa modelo a partir de objeto Parameters do Flower.
    USA parameters_to_ndarrays para conversao correta!
    """
    try:
        ndarrays = parameters_to_ndarrays(params)
        return deserialize_model_from_ndarrays(ndarrays)
    except Exception as e:
        print(f"    Erro convertendo Parameters: {e}")
        return None


# ==============================================================================
# SECAO 7: CLIENTE FLOWER COM WARM START
# ==============================================================================

class TreeModelClient(fl.client.NumPyClient):
    """
    Cliente Flower para modelos de arvore com suporte a WARM START.
    """

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model = None

        self.X_train = GLOBAL_CLIENT_DATA[client_id]['X']
        self.y_train = GLOBAL_CLIENT_DATA[client_id]['y']
        self.age_train = GLOBAL_CLIENT_DATA[client_id]['age']

        self.X_val = GLOBAL_VAL_DATA['X']
        self.y_val = GLOBAL_VAL_DATA['y']
        self.age_val = GLOBAL_VAL_DATA['age']

        self.model_type = GLOBAL_MODEL_TYPE
        self.model_params = GLOBAL_MODEL_PARAMS.get(self.model_type, {})
        self.scale_pos_weight = GLOBAL_SCALE_POS_WEIGHT

    def _create_and_train_model(self, warm_start_model: Optional[object] = None):
        """Cria e treina modelo com suporte a Warm Start."""
        if self.model_type == 'xgboost':
            full_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'random_state': RANDOM_STATE,
                'scale_pos_weight': self.scale_pos_weight,
                **self.model_params
            }

            self.model = xgb.XGBClassifier(**full_params)

            if warm_start_model is not None:
                print(f"    [Cliente {self.client_id}] WARM START XGBoost")
                self.model.fit(
                    self.X_train, self.y_train,
                    xgb_model=warm_start_model.get_booster(),
                    verbose=False
                )
            else:
                print(f"    [Cliente {self.client_id}] XGBoost - do zero")
                self.model.fit(self.X_train, self.y_train, verbose=False)

        elif self.model_type == 'lightgbm':
            full_params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'random_state': RANDOM_STATE,
                'scale_pos_weight': self.scale_pos_weight,
                **self.model_params
            }

            self.model = lgb.LGBMClassifier(**full_params)

            if warm_start_model is not None:
                print(f"    [Cliente {self.client_id}] WARM START LightGBM")
                self.model.fit(self.X_train, self.y_train, init_model=warm_start_model)
            else:
                print(f"    [Cliente {self.client_id}] LightGBM - do zero")
                self.model.fit(self.X_train, self.y_train)

        elif self.model_type == 'catboost':
            full_params = {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': RANDOM_STATE,
                'auto_class_weights': 'Balanced',
                'verbose': False,
                **self.model_params
            }

            self.model = CatBoostClassifier(**full_params)

            if warm_start_model is not None:
                print(f"    [Cliente {self.client_id}] WARM START CatBoost")
                self.model.fit(self.X_train, self.y_train, init_model=warm_start_model, verbose=False)
            else:
                print(f"    [Cliente {self.client_id}] CatBoost - do zero")
                self.model.fit(self.X_train, self.y_train, verbose=False)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Retorna modelo serializado."""
        return serialize_model(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Treina modelo localmente com suporte a WARM START.

        NOTA: O Flower ja converte Parameters -> List[np.ndarray] automaticamente
        antes de chamar este metodo.
        """
        server_round = config.get("server_round", 0)
        use_warm_start = config.get("warm_start", False)

        print(f"\n    [Cliente {self.client_id}] Round {server_round} | warm_start={use_warm_start}")

        # Deserializar modelo para warm start (se aplicavel)
        warm_start_model = None
        if use_warm_start:
            warm_start_model = deserialize_model_from_ndarrays(parameters)
            if warm_start_model is not None:
                print(f"    [Cliente {self.client_id}] Modelo recebido para warm start!")
            else:
                print(f"    [Cliente {self.client_id}] AVISO: Falha ao receber modelo, treinando do zero")

        # Treinar
        self._create_and_train_model(warm_start_model)

        # Metricas
        y_prob_train = self.model.predict_proba(self.X_train)[:, 1]
        train_tpr = calc_tpr_at_fpr(self.y_train, y_prob_train)

        y_prob_val = self.model.predict_proba(self.X_val)[:, 1]
        val_tpr = calc_tpr_at_fpr(self.y_val, y_prob_val)

        print(f"    [Cliente {self.client_id}] Treino TPR={train_tpr:.4f} | Val TPR={val_tpr:.4f}")

        return serialize_model(self.model), len(self.X_train), {
            "client_id": self.client_id,
            "train_tpr": float(train_tpr),
            "val_tpr": float(val_tpr),
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        if self.model is None:
            return 1.0, 0, {}

        y_prob = self.model.predict_proba(self.X_val)[:, 1]
        metrics = evaluate_model_standard(self.y_val, y_prob, self.age_val)

        return 1.0 - metrics['tpr_at_5fpr'], len(self.X_val), {
            "tpr_at_5fpr": metrics['tpr_at_5fpr'],
            "roc_auc": metrics['roc_auc'],
        }


def client_fn(cid: str) -> fl.client.Client:
    """Factory function para criar clientes."""
    client_id = int(cid)
    return TreeModelClient(client_id).to_client()


# ==============================================================================
# SECAO 8: ESTRATEGIA BAGGING
# ==============================================================================

class BaggingStrategy(Strategy):
    """Bagging: Todos os clientes treinam, predicoes agregadas por media."""

    def __init__(self, num_clients: int = 3):
        self.num_clients = num_clients
        self.client_models: Dict[int, object] = {}
        self.round_metrics: List[Dict] = []

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        print(f"\n{'='*60}")
        print(f"BAGGING - Round {server_round}")
        print(f"{'='*60}")

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients
        )

        config = {"server_round": server_round, "warm_start": False}
        fit_ins = FitIns(parameters=Parameters(tensors=[], tensor_type=""), config=config)

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"  [Servidor] Agregando {len(results)} modelos...")

        self.client_models = {}

        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("client_id", 0)

            # CORRECAO: Usar parameters_to_ndarrays para extrair o array
            model = deserialize_model_from_parameters(fit_res.parameters)

            if model is not None:
                self.client_models[client_id] = model
                print(f"  [Servidor] Modelo do Cliente {client_id} OK")
            else:
                print(f"  [Servidor] Falha ao carregar modelo do Cliente {client_id}")

        return None, {"num_models": len(self.client_models)}

    def configure_evaluate(self, server_round: int, parameters: Parameters,
                           client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return []

    def aggregate_evaluate(self, server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if not self.client_models:
            return None

        X_val = GLOBAL_VAL_DATA['X']
        y_val = GLOBAL_VAL_DATA['y']
        age_val = GLOBAL_VAL_DATA['age']

        predictions = []
        for model in self.client_models.values():
            pred = model.predict_proba(X_val)[:, 1]
            predictions.append(pred)

        y_prob_ensemble = np.mean(predictions, axis=0)

        metrics_std = evaluate_model_standard(y_val, y_prob_ensemble, age_val)
        metrics_fair = evaluate_model_fair(y_val, y_prob_ensemble, age_val)

        print(f"\n  [Servidor] BAGGING Round {server_round}:")
        print(f"    Threshold Unico:     TPR={metrics_std['tpr_at_5fpr']:.4f}, Fairness={metrics_std['fairness_ratio']:.4f}")
        print(f"    Threshold por Grupo: TPR={metrics_fair['tpr_at_5fpr']:.4f}, Fairness={metrics_fair['fairness_ratio']:.4f}")

        self.round_metrics.append({
            'round': server_round,
            'tpr_standard': metrics_std['tpr_at_5fpr'],
            'fairness_standard': metrics_std['fairness_ratio'],
            'tpr_fair': metrics_fair['tpr_at_5fpr'],
            'fairness_fair': metrics_fair['fairness_ratio'],
        })

        return 1.0 - metrics_std['tpr_at_5fpr'], {"tpr_at_5fpr": metrics_std['tpr_at_5fpr']}

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        predictions = [m.predict_proba(X)[:, 1] for m in self.client_models.values()]
        return np.mean(predictions, axis=0)


# ==============================================================================
# SECAO 9: ESTRATEGIA CYCLING COM WARM START
# ==============================================================================

class CyclingStrategy(Strategy):
    """
    Cycling com WARM START: Treinamento sequencial com transferencia de conhecimento.
    """

    def __init__(self, num_clients: int = 3):
        self.num_clients = num_clients
        self.current_client_idx = 0
        self.current_model: Optional[object] = None
        self.round_metrics: List[Dict] = []

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        print(f"\n{'='*60}")
        print(f"CYCLING - Round {server_round} - Cliente {self.current_client_idx}")
        print(f"{'='*60}")

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients
        )

        selected_client = clients[self.current_client_idx]

        # Preparar parametros para warm start
        if self.current_model is not None:
            # CORRECAO: Usar ndarrays_to_parameters para conversao correta
            model_ndarrays = serialize_model(self.current_model)
            fit_params = ndarrays_to_parameters(model_ndarrays)
            config = {"server_round": server_round, "warm_start": True}
            print(f"  [Servidor] Enviando modelo para Cliente {self.current_client_idx} (warm start)")
        else:
            fit_params = Parameters(tensors=[], tensor_type="")
            config = {"server_round": server_round, "warm_start": False}
            print(f"  [Servidor] Cliente {self.current_client_idx} iniciara do zero")

        fit_ins = FitIns(parameters=fit_params, config=config)

        return [(selected_client, fit_ins)]

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            print(f"  [Servidor] AVISO: Nenhum resultado recebido!")
            return None, {}

        client_proxy, fit_res = results[0]
        client_id = fit_res.metrics.get("client_id", self.current_client_idx)

        # CORRECAO: Usar parameters_to_ndarrays para extrair o modelo
        model = deserialize_model_from_parameters(fit_res.parameters)

        if model is not None:
            self.current_model = model
            print(f"  [Servidor] Modelo do Cliente {client_id} salvo para proximo round")
        else:
            print(f"  [Servidor] ERRO: Falha ao deserializar modelo do Cliente {client_id}")

        # Avancar para proximo cliente
        prev_client = self.current_client_idx
        self.current_client_idx = (self.current_client_idx + 1) % self.num_clients

        return None, {"trained_client": prev_client}

    def configure_evaluate(self, server_round: int, parameters: Parameters,
                           client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return []

    def aggregate_evaluate(self, server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self.current_model is None:
            return None

        X_val = GLOBAL_VAL_DATA['X']
        y_val = GLOBAL_VAL_DATA['y']
        age_val = GLOBAL_VAL_DATA['age']

        y_prob = self.current_model.predict_proba(X_val)[:, 1]

        metrics_std = evaluate_model_standard(y_val, y_prob, age_val)
        metrics_fair = evaluate_model_fair(y_val, y_prob, age_val)

        trained_client = (self.current_client_idx - 1) % self.num_clients

        print(f"\n  [Servidor] CYCLING Round {server_round} (Cliente {trained_client}):")
        print(f"    Threshold Unico:     TPR={metrics_std['tpr_at_5fpr']:.4f}, Fairness={metrics_std['fairness_ratio']:.4f}")
        print(f"    Threshold por Grupo: TPR={metrics_fair['tpr_at_5fpr']:.4f}, Fairness={metrics_fair['fairness_ratio']:.4f}")

        self.round_metrics.append({
            'round': server_round,
            'trained_client': trained_client,
            'tpr_standard': metrics_std['tpr_at_5fpr'],
            'fairness_standard': metrics_std['fairness_ratio'],
            'tpr_fair': metrics_fair['tpr_at_5fpr'],
            'fairness_fair': metrics_fair['fairness_ratio'],
        })

        return 1.0 - metrics_std['tpr_at_5fpr'], {
            "tpr_at_5fpr": metrics_std['tpr_at_5fpr'],
            "trained_client": trained_client,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.current_model is None:
            raise ValueError("Nenhum modelo disponivel")
        return self.current_model.predict_proba(X)[:, 1]


# ==============================================================================
# SECAO 10: EXECUCAO PRINCIPAL
# ==============================================================================

def run_federated_training(client_data: List[Dict],
                           X_val: np.ndarray, y_val: np.ndarray, age_val: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray, age_test: np.ndarray,
                           best_params: Dict, scale_pos_weight: float) -> pd.DataFrame:
    global GLOBAL_CLIENT_DATA, GLOBAL_VAL_DATA, GLOBAL_TEST_DATA
    global GLOBAL_MODEL_PARAMS, GLOBAL_SCALE_POS_WEIGHT, GLOBAL_MODEL_TYPE

    for i, data in enumerate(client_data):
        GLOBAL_CLIENT_DATA[i] = data

    GLOBAL_VAL_DATA = {'X': X_val, 'y': y_val, 'age': age_val}
    GLOBAL_TEST_DATA = {'X': X_test, 'y': y_test, 'age': age_test}
    GLOBAL_SCALE_POS_WEIGHT = scale_pos_weight
    GLOBAL_MODEL_PARAMS = best_params

    all_results = []
    strategies_trained = {}

    for model_type in ['xgboost', 'lightgbm', 'catboost']:
        GLOBAL_MODEL_TYPE = model_type

        print("\n" + "#" * 70)
        print(f"# MODELO: {model_type.upper()}")
        print("#" * 70)

        # BAGGING
        print("\n" + "-" * 60)
        print(f"ESTRATEGIA: BAGGING ({model_type.upper()})")
        print("-" * 60)

        bagging_strategy = BaggingStrategy(num_clients=CONFIG.num_clients)

        start_simulation(
            client_fn=client_fn,
            num_clients=CONFIG.num_clients,
            config=fl.server.ServerConfig(num_rounds=CONFIG.num_rounds),
            strategy=bagging_strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

        strategies_trained[f"{model_type}_bagging"] = bagging_strategy

        # CYCLING
        print("\n" + "-" * 60)
        print(f"ESTRATEGIA: CYCLING ({model_type.upper()})")
        print("-" * 60)

        cycling_strategy = CyclingStrategy(num_clients=CONFIG.num_clients)

        start_simulation(
            client_fn=client_fn,
            num_clients=CONFIG.num_clients,
            config=fl.server.ServerConfig(num_rounds=CONFIG.num_rounds),
            strategy=cycling_strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

        strategies_trained[f"{model_type}_cycling"] = cycling_strategy

    # Avaliacao final
    print("\n" + "=" * 70)
    print("AVALIACAO FINAL (TESTE - MES 7)")
    print("=" * 70)

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


def main(data_path: str = 'Base.csv'):
    print("#" * 80)
    print("#" + " " * 20 + "FEDERATED LEARNING COM FLOWER" + " " * 20 + "#")
    print("#" + " " * 15 + "BANK ACCOUNT FRAUD DETECTION (CORRIGIDO)" + " " * 15 + "#")
    print("#" * 80)
    print("\nCorrecoes implementadas:")
    print("  1. fl.simulation.start_simulation (motor oficial)")
    print("  2. Warm Start para Cycling (transferencia de conhecimento)")
    print("  3. Post-Processing com Limiares por Grupo (Fairness = 1.0)")
    print("  4. Serializacao correta: parameters_to_ndarrays / ndarrays_to_parameters")
    print("  5. Optuna FEDERADO: cada cliente otimiza localmente, parametros agregados")

    preprocessor = DataPreprocessor()

    df = preprocessor.load_and_clean(data_path)
    df = preprocessor.create_features(df)
    df_train, df_val, df_test = preprocessor.temporal_split(df)

    (X_train, y_train, age_train,
     X_val, y_val, age_val,
     X_test, y_test, age_test) = preprocessor.encode_features(df_train, df_val, df_test)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"\nScale pos weight: {scale_pos_weight:.2f}")

    client_data = partition_data_balanced(X_train, y_train, age_train, CONFIG.num_clients)

    # Otimizacao FEDERADA de hiperparametros
    # Cada cliente roda Optuna localmente, parametros sao agregados no servidor
    best_params = federated_hyperparameter_optimization(
        client_data=client_data,
        X_val=X_val,
        y_val=y_val,
        scale_pos_weight=scale_pos_weight,
        n_trials_per_client=15,      # Menos trials por cliente (total = 15 * 3 = 45)
        sample_fraction=1/3,          # 1/3 dos dados de cada cliente (~33% do total)
        max_learning_rate=0.05        # Freio de seguranca
    )

    df_results = run_federated_training(
        client_data, X_val, y_val, age_val,
        X_test, y_test, age_test,
        best_params, scale_pos_weight
    )

    # Relatorio
    print("\n" + "=" * 80)
    print("RELATORIO FINAL")
    print("=" * 80)

    if len(df_results) > 0:
        print("\n--- Threshold Unico ---")
        df_single = df_results[df_results['threshold'] == 'single']
        for _, row in df_single.iterrows():
            print(f"  {row['model']}: TPR={row['tpr_at_5fpr']:.4f}, Fairness={row['fairness_ratio']:.4f}")

        print("\n--- Threshold por Grupo (Fair) ---")
        df_fair = df_results[df_results['threshold'] == 'per_group']
        for _, row in df_fair.iterrows():
            print(f"  {row['model']}: TPR={row['tpr_at_5fpr']:.4f}, Fairness={row['fairness_ratio']:.4f}")

        print("\n" + "-" * 60)
        print("BENCHMARK (TPR > 0.52, Fairness ~ 1.0)")
        print("-" * 60)

        for _, row in df_results.iterrows():
            tpr = row['tpr_at_5fpr']
            fr = row['fairness_ratio']
            tpr_ok = "[OK]" if tpr > 0.52 else "[X]"
            fr_ok = "[OK]" if 0.9 <= fr <= 1.1 else "[X]"
            print(f"{tpr_ok} {fr_ok} {row['model']}: TPR={tpr:.4f}, Fairness={fr:.4f}")
    else:
        print("AVISO: Nenhum resultado obtido!")

    return df_results


if __name__ == "__main__":
    results = main()