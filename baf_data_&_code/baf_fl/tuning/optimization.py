"""
Otimizacao federada de hiperparametros com Optuna.

Cada cliente roda Optuna localmente, parametros sao agregados via mediana.
"""

import numpy as np
from typing import Dict, List

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import optuna
from optuna.samplers import TPESampler

from ..config import RANDOM_STATE
from ..data.metrics import calc_tpr_at_fpr


class LocalHyperparameterOptimizer:
    """
    Otimizador local para um unico cliente.
    Roda Optuna usando apenas os dados daquele cliente.
    """

    def __init__(self, client_id: int, n_trials: int = 15):
        self.client_id = client_id
        self.n_trials = n_trials
        self.best_params: Dict[str, Dict] = {}

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

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=RANDOM_STATE + self.client_id),
        )
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

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=RANDOM_STATE + self.client_id),
        )
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

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=RANDOM_STATE + self.client_id),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params['catboost'] = study.best_params
        return study.best_params, study.best_value


class HyperparameterAggregator:
    """Agrega hiperparametros de multiplos clientes usando mediana."""

    # Definicoes de chaves por algoritmo
    PARAM_KEYS = {
        'xgboost': {
            'all': ['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight',
                    'subsample', 'colsample_bytree'],
            'int': ['max_depth', 'n_estimators', 'min_child_weight'],
        },
        'lightgbm': {
            'all': ['max_depth', 'learning_rate', 'n_estimators', 'num_leaves',
                    'min_child_samples', 'subsample', 'colsample_bytree'],
            'int': ['max_depth', 'n_estimators', 'num_leaves', 'min_child_samples'],
        },
        'catboost': {
            'all': ['depth', 'learning_rate', 'iterations', 'l2_leaf_reg'],
            'int': ['depth', 'iterations'],
        },
    }

    @staticmethod
    def aggregate(local_params_list: List[Dict], param_keys: List[str],
                  int_keys: List[str]) -> Dict:
        """
        Agrega hiperparametros usando MEDIANA.
        Mediana e mais robusta que media contra outliers.
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

    @classmethod
    def aggregate_for_algorithm(cls, algo: str, local_params_list: List[Dict]) -> Dict:
        """Agrega parametros para um algoritmo especifico."""
        keys = cls.PARAM_KEYS[algo]
        return cls.aggregate(local_params_list, keys['all'], keys['int'])


def federated_hyperparameter_optimization(
    client_data: List[Dict],
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    n_trials_per_client: int = 15,
    sample_fraction: float = 0.3,
    max_learning_rate: float = 0.05,
) -> Dict:
    """
    Otimizacao FEDERADA de hiperparametros:
    1. Cada cliente roda Optuna localmente com seus proprios dados
    2. Parametros sao agregados (mediana) no servidor
    3. Aplica freio de seguranca no learning_rate
    """
    print("\n" + "=" * 70)
    print("OTIMIZACAO FEDERADA DE HIPERPARAMETROS")
    print("=" * 70)
    print(f"  - {len(client_data)} clientes")
    print(f"  - {n_trials_per_client} trials por cliente")
    print(f"  - {sample_fraction*100:.0f}% dos dados de cada cliente")
    print(f"  - Learning rate maximo: {max_learning_rate}")

    xgb_local_params = []
    lgbm_local_params = []
    cat_local_params = []

    # FASE 1: Optuna LOCAL em cada cliente
    for i in range(len(client_data)):
        print(f"\n>>> Cliente {i}: Rodando Optuna localmente...")

        X_c = client_data[i]['X']
        y_c = client_data[i]['y']

        n_samples = int(len(y_c) * sample_fraction)
        idx = np.random.choice(len(y_c), n_samples, replace=False)
        X_opt = X_c[idx]
        y_opt = y_c[idx]

        print(f"    Usando {n_samples:,} amostras ({sample_fraction*100:.0f}% dos dados)")

        optimizer = LocalHyperparameterOptimizer(client_id=i, n_trials=n_trials_per_client)

        print(f"    [XGBoost] Otimizando...")
        xgb_params, xgb_score = optimizer.optimize_xgboost(X_opt, y_opt, X_val, y_val, scale_pos_weight)
        xgb_local_params.append(xgb_params)
        print(f"    [XGBoost] TPR@5%FPR: {xgb_score:.4f}, LR: {xgb_params['learning_rate']:.4f}")

        print(f"    [LightGBM] Otimizando...")
        lgbm_params, lgbm_score = optimizer.optimize_lightgbm(X_opt, y_opt, X_val, y_val, scale_pos_weight)
        lgbm_local_params.append(lgbm_params)
        print(f"    [LightGBM] TPR@5%FPR: {lgbm_score:.4f}, LR: {lgbm_params['learning_rate']:.4f}")

        print(f"    [CatBoost] Otimizando...")
        cat_params, cat_score = optimizer.optimize_catboost(X_opt, y_opt, X_val, y_val)
        cat_local_params.append(cat_params)
        print(f"    [CatBoost] TPR@5%FPR: {cat_score:.4f}, LR: {cat_params['learning_rate']:.4f}")

    # FASE 2: Agregar parametros (MEDIANA)
    print("\n>>> Agregando parametros dos clientes (mediana)...")

    aggregator = HyperparameterAggregator()

    final_xgb = aggregator.aggregate_for_algorithm('xgboost', xgb_local_params)
    final_lgbm = aggregator.aggregate_for_algorithm('lightgbm', lgbm_local_params)
    final_cat = aggregator.aggregate_for_algorithm('catboost', cat_local_params)

    for algo, local_list, final in [
        ('XGBoost', xgb_local_params, final_xgb),
        ('LightGBM', lgbm_local_params, final_lgbm),
        ('CatBoost', cat_local_params, final_cat),
    ]:
        keys = aggregator.PARAM_KEYS[algo.lower()]['all']
        print(f"\n    {algo}:")
        for key in keys:
            local_vals = [p[key] for p in local_list]
            print(f"      {key}: {[f'{v:.4f}' if isinstance(v, float) else v for v in local_vals]} -> {final[key]}")

    # FASE 3: Freio de seguranca no Learning Rate
    print(f"\n>>> Aplicando freio de seguranca (LR max = {max_learning_rate})...")

    for name, params in [('XGBoost', final_xgb), ('LightGBM', final_lgbm), ('CatBoost', final_cat)]:
        if params['learning_rate'] > max_learning_rate:
            print(f"    {name}: {params['learning_rate']:.4f} -> {max_learning_rate}")
            params['learning_rate'] = max_learning_rate

    print("\n>>> Parametros finais agregados:")
    print(f"    XGBoost:  LR={final_xgb['learning_rate']:.4f}, depth={final_xgb['max_depth']}, n_est={final_xgb['n_estimators']}")
    print(f"    LightGBM: LR={final_lgbm['learning_rate']:.4f}, depth={final_lgbm['max_depth']}, n_est={final_lgbm['n_estimators']}")
    print(f"    CatBoost: LR={final_cat['learning_rate']:.4f}, depth={final_cat['depth']}, iter={final_cat['iterations']}")

    return {
        'xgboost': final_xgb,
        'lightgbm': final_lgbm,
        'catboost': final_cat,
    }
