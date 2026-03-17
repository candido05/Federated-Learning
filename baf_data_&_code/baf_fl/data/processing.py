"""
Pre-processamento de dados e particionamento para clientes federados.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.preprocessing import OneHotEncoder

from ..config import RANDOM_STATE


class DataPreprocessor:
    """Pre-processamento dos dados de fraude bancaria."""

    COLS_TO_DROP = ['prev_address_months_count', 'intended_balcon_amount', 'device_fraud_count']
    DROPNA_COLS = ['device_distinct_emails_8w']
    EXTRA_DROP = ['velocity_24h']

    CATEGORICAL_COLS = [
        'payment_type', 'employment_status', 'housing_status',
        'source', 'device_os', 'age_group', 'employment_income_cat',
        'credit_limit_bucket',
    ]

    EXCLUDE_COLS = ['fraud_bool', 'month', 'age_above_50']

    def __init__(self):
        self.ohe_encoder: OneHotEncoder = None
        self.categorical_cols: List[str] = self.CATEGORICAL_COLS
        self.numeric_cols: List[str] = None

    def load_and_clean(self, filepath: str) -> pd.DataFrame:
        print("=" * 60)
        print("CARREGAMENTO E LIMPEZA DOS DADOS")
        print("=" * 60)

        df = pd.read_csv(filepath)
        print(f"Dataset carregado: {df.shape[0]:,} linhas x {df.shape[1]} colunas")

        df_clean = df.replace(-1, np.nan)
        df_clean = df_clean.drop(columns=self.COLS_TO_DROP)

        rows_before = len(df_clean)
        df_clean = df_clean.dropna(subset=self.DROPNA_COLS)
        print(f"Linhas removidas: {rows_before - len(df_clean):,}")

        df_clean = df_clean.drop(columns=self.EXTRA_DROP)

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

        # Frequency encoding
        device_os_counts = df.groupby('device_os').size().to_dict()
        df['device_os_frequency'] = df['device_os'].map(device_os_counts)

        source_counts = df.groupby('source').size().to_dict()
        df['source_frequency'] = df['source'].map(source_counts)

        # Velocity ratio
        df['velocity_ratio_6h_4w'] = df['velocity_6h'] / (df['velocity_4w'] + 1)

        # Monthly z-scores
        for col in ['velocity_6h', 'velocity_4w', 'zip_count_4w', 'bank_branch_count_8w']:
            monthly_mean = df.groupby('month')[col].transform('mean')
            monthly_std = df.groupby('month')[col].transform('std').replace(0, 1)
            df[f'{col}_zscore_monthly'] = (df[col] - monthly_mean) / monthly_std

        # Age and income interactions
        df['income_x_age'] = df['income'] * df['customer_age']

        df['age_group'] = pd.cut(
            df['customer_age'],
            bins=[0, 30, 50, 100],
            labels=['young', 'middle', 'senior'],
        )
        income_by_age = df.groupby('age_group')['income'].transform('mean')
        df['income_vs_age_group_mean'] = df['income'] - income_by_age

        df['employment_income_cat'] = (
            df['employment_status'] + '_'
            + pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype(str)
        )

        df['age_x_credit_risk'] = df['customer_age'] * df['credit_risk_score']
        df['age_above_50'] = (df['customer_age'] > 50).astype(int)
        df['income_per_age'] = df['income'] / (df['customer_age'] + 1)

        # Credit risk vs employment
        emp_credit_mean = df.groupby('employment_status')['credit_risk_score'].transform('mean')
        emp_credit_std = df.groupby('employment_status')['credit_risk_score'].transform('std').replace(0, 1)
        df['credit_risk_vs_employment'] = (df['credit_risk_score'] - emp_credit_mean) / emp_credit_std

        # Phone validation
        df['phone_validation_score'] = df['phone_home_valid'] + df['phone_mobile_valid']

        # Session length vs source
        source_session_mean = df.groupby('source')['session_length_in_minutes'].transform('mean')
        df['session_length_vs_source'] = df['session_length_in_minutes'] - source_session_mean

        # Other derived features
        df['email_similarity_x_free'] = df['name_email_similarity'] * df['email_is_free']
        df['bank_vs_address_months'] = df['bank_months_count'] / (df['current_address_months_count'] + 1)
        df['days_since_request_log'] = np.log1p(df['days_since_request'])

        df['credit_limit_bucket'] = pd.cut(
            df['proposed_credit_limit'],
            bins=[0, 200, 500, 1000, 2000, np.inf],
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
        )

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

    def encode_features(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
    ) -> Tuple[np.ndarray, ...]:
        print("\n" + "=" * 60)
        print("ENCODING")
        print("=" * 60)

        for col in self.categorical_cols:
            for df in [df_train, df_val, df_test]:
                df[col] = df[col].astype(str)

        feature_cols = [c for c in df_train.columns if c not in self.EXCLUDE_COLS]

        X_train = df_train[feature_cols].copy()
        X_val = df_val[feature_cols].copy()
        X_test = df_test[feature_cols].copy()

        y_train = df_train['fraud_bool'].values
        y_val = df_val['fraud_bool'].values
        y_test = df_test['fraud_bool'].values

        age_train = df_train['age_above_50'].values
        age_val = df_val['age_above_50'].values
        age_test = df_test['age_above_50'].values

        self.numeric_cols = [c for c in X_train.columns if c not in self.categorical_cols]

        self.ohe_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown='ignore', drop='if_binary',
        )

        X_train_cat = self.ohe_encoder.fit_transform(X_train[self.categorical_cols])
        X_val_cat = self.ohe_encoder.transform(X_val[self.categorical_cols])
        X_test_cat = self.ohe_encoder.transform(X_test[self.categorical_cols])

        X_train_enc = np.hstack([X_train[self.numeric_cols].values, X_train_cat])
        X_val_enc = np.hstack([X_val[self.numeric_cols].values, X_val_cat])
        X_test_enc = np.hstack([X_test[self.numeric_cols].values, X_test_cat])

        print(f"Features encoded: {X_train_enc.shape[1]}")

        return (
            X_train_enc, y_train, age_train,
            X_val_enc, y_val, age_val,
            X_test_enc, y_test, age_test,
        )


class DataPartitioner:
    """Particiona dados entre clientes federados de forma balanceada."""

    @staticmethod
    def partition_balanced(
        X: np.ndarray, y: np.ndarray, age: np.ndarray, num_clients: int = 3
    ) -> List[Dict]:
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
