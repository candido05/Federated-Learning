"""Módulo de cálculo de métricas para experimentos de Aprendizado Federado.

Este módulo fornece funções para calcular, formatar e serializar métricas
de classificação incluindo acurácia, precisão, recall, F1-score, AUC e matriz de confusão.
"""

import json
from typing import Any, Dict, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsCalculator:
    """Calculadora de métricas para tarefas de classificação binária e multiclasse."""

    @staticmethod
    def calculate_comprehensive_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Union[float, int, Dict[str, int]]]:
        """Calcula métricas abrangentes para classificação.

        Args:
            y_true: Labels verdadeiros (array de shape [n_samples] ou [n_samples, n_classes]).
            y_pred_proba: Probabilidades preditas (array de shape [n_samples] ou [n_samples, n_classes]).
            threshold: Limiar para conversão de probabilidades em predições binárias (padrão: 0.5).

        Returns:
            Dicionário contendo:
                - accuracy: Acurácia de classificação
                - precision: Precisão (macro-averaged para multiclasse)
                - recall: Recall (macro-averaged para multiclasse)
                - f1_score: F1-score (macro-averaged para multiclasse)
                - auc: Area Under the ROC Curve
                - confusion_matrix: Dicionário com tn, fp, fn, tp (binário) ou matriz completa (multiclasse)
        """
        # Converte para numpy arrays se necessário
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        # Determina se é classificação binária ou multiclasse
        is_binary = len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1

        if is_binary:
            # Classificação binária
            if len(y_pred_proba.shape) == 2:
                y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba[:, 0]

            # Converte probabilidades em predições binárias
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Calcula métricas
            accuracy = float(accuracy_score(y_true, y_pred))
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))

            # Calcula AUC
            try:
                auc_score = float(roc_auc_score(y_true, y_pred_proba))
            except ValueError:
                # Se houver apenas uma classe nos dados, AUC não pode ser calculado
                auc_score = 0.0

            # Calcula matriz de confusão
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                confusion_matrix_dict = {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
            else:
                # Caso tenha apenas uma classe nas predições
                confusion_matrix_dict = {
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "tp": 0,
                    "note": "Apenas uma classe presente nas predições",
                }

        else:
            # Classificação multiclasse
            y_pred = np.argmax(y_pred_proba, axis=1)

            # Calcula métricas (macro-averaged)
            accuracy = float(accuracy_score(y_true, y_pred))
            precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

            # Calcula AUC (macro-averaged)
            try:
                auc_score = float(roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro"))
            except ValueError:
                auc_score = 0.0

            # Matriz de confusão completa
            cm = confusion_matrix(y_true, y_pred)
            confusion_matrix_dict = {"matrix": cm.tolist()}

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc_score,
            "confusion_matrix": confusion_matrix_dict,
        }

    @staticmethod
    def print_metrics_summary(metrics: Dict[str, Any], prefix: str = "") -> None:
        """Imprime um resumo formatado das métricas no console.

        Args:
            metrics: Dicionário de métricas retornado por calculate_comprehensive_metrics.
            prefix: String de prefixo para adicionar antes de cada linha (padrão: "").
        """
        print(f"{prefix}{'='*60}")
        print(f"{prefix}RESUMO DE MÉTRICAS")
        print(f"{prefix}{'='*60}")

        # Métricas principais
        if "accuracy" in metrics:
            print(f"{prefix}Acurácia:     {metrics['accuracy']:.4f}")
        if "precision" in metrics:
            print(f"{prefix}Precisão:     {metrics['precision']:.4f}")
        if "recall" in metrics:
            print(f"{prefix}Recall:       {metrics['recall']:.4f}")
        if "f1_score" in metrics:
            print(f"{prefix}F1-Score:     {metrics['f1_score']:.4f}")
        if "auc" in metrics:
            print(f"{prefix}AUC:          {metrics['auc']:.4f}")

        # Matriz de confusão
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            print(f"{prefix}{'-'*60}")
            print(f"{prefix}MATRIZ DE CONFUSÃO:")

            if isinstance(cm, dict) and "tn" in cm:
                # Classificação binária
                print(f"{prefix}  TN: {cm['tn']:>6}  |  FP: {cm['fp']:>6}")
                print(f"{prefix}  FN: {cm['fn']:>6}  |  TP: {cm['tp']:>6}")
            elif isinstance(cm, dict) and "matrix" in cm:
                # Classificação multiclasse
                matrix = cm["matrix"]
                print(f"{prefix}  {np.array(matrix)}")

        print(f"{prefix}{'='*60}")

    @staticmethod
    def metrics_to_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Converte métricas para formato JSON serializável.

        Converte tipos numpy (int64, float64, etc.) para tipos Python nativos
        para permitir serialização JSON.

        Args:
            metrics: Dicionário de métricas com potenciais tipos numpy.

        Returns:
            Dicionário com todos os valores convertidos para tipos Python nativos.
        """

        def convert_value(value: Any) -> Any:
            """Converte recursivamente valores numpy para tipos Python."""
            if isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            else:
                return value

        return convert_value(metrics)

    @staticmethod
    def save_metrics_to_file(metrics: Dict[str, Any], filepath: str) -> None:
        """Salva métricas em arquivo JSON.

        Args:
            metrics: Dicionário de métricas.
            filepath: Caminho para o arquivo de saída.
        """
        serializable_metrics = MetricsCalculator.metrics_to_json(metrics)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

    @staticmethod
    def calculate_aggregated_metrics(
        metrics_list: list[Dict[str, Any]]
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Calcula métricas agregadas (média, std, min, max) de uma lista de métricas.

        Args:
            metrics_list: Lista de dicionários de métricas.

        Returns:
            Dicionário com estatísticas agregadas para cada métrica.
        """
        if not metrics_list:
            return {}

        # Identifica métricas numéricas
        numeric_keys = ["accuracy", "precision", "recall", "f1_score", "auc"]
        aggregated = {}

        for key in numeric_keys:
            values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]

            if values:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        return aggregated
