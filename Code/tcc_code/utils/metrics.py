"""Módulo de cálculo de métricas para experimentos de Aprendizado Federado.

Este módulo fornece funções para calcular, formatar e serializar métricas
de classificação incluindo acurácia, precisão, recall, F1-score, AUC e matriz de confusão.
"""

import json
from typing import Any, Dict, List, Optional, Union

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
                - recall: Recall (sensibilidade, macro-averaged para multiclasse)
                - f1_score: F1-score (macro-averaged para multiclasse)
                - auc: Area Under the ROC Curve
                - specificity: Especificidade (apenas para binário)
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

            # Calcula matriz de confusão e especificidade
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()

                # Calcula especificidade (True Negative Rate)
                # Specificity = TN / (TN + FP)
                specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

                confusion_matrix_dict = {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
            else:
                # Caso tenha apenas uma classe nas predições
                specificity = 0.0
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

            # Specificity não é aplicável diretamente em multiclasse
            # Pode-se calcular macro-averaged specificity, mas fica para futura implementação
            specificity = None

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc_score,
            "specificity": specificity,  # None para multiclasse, float para binário
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
            print(f"{prefix}Acurácia:       {metrics['accuracy']:.4f}")
        if "precision" in metrics:
            print(f"{prefix}Precisão:       {metrics['precision']:.4f}")
        if "recall" in metrics:
            print(f"{prefix}Recall:         {metrics['recall']:.4f}")
        if "specificity" in metrics and metrics["specificity"] is not None:
            print(f"{prefix}Especificidade: {metrics['specificity']:.4f}")
        if "f1_score" in metrics:
            print(f"{prefix}F1-Score:       {metrics['f1_score']:.4f}")
        if "auc" in metrics:
            print(f"{prefix}AUC:            {metrics['auc']:.4f}")

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
    def save_history_to_json(
        history: Dict[str, Any],
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Salva histórico de métricas de FL em arquivo JSON formatado.

        Args:
            history: Histórico de métricas do Flower (ou dicionário customizado).
            filepath: Caminho para o arquivo de saída.
            metadata: Metadados adicionais do experimento (modelo, estratégia, etc.).

        Example:
            >>> save_history_to_json(
            ...     history={"losses": [...], "metrics": [...]},
            ...     filepath="results/experiment_001.json",
            ...     metadata={"model": "xgboost", "strategy": "bagging"}
            ... )
        """
        output = {}

        # Adiciona metadados se fornecidos
        if metadata:
            output["metadata"] = metadata

        # Adiciona histórico
        output["history"] = MetricsCalculator.metrics_to_json(history)

        # Salva em JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    @staticmethod
    def print_final_analysis(
        strategy_name: str,
        metrics_history: Dict[str, Any],
        num_rounds: Optional[int] = None,
    ) -> None:
        """Imprime análise final formatada do experimento FL.

        Args:
            strategy_name: Nome da estratégia usada (ex: "FedBagging", "FedCyclic").
            metrics_history: Histórico de métricas por round.
            num_rounds: Número total de rounds (opcional, inferido do histórico se None).

        Example:
            >>> metrics_history = {
            ...     "losses": [0.5, 0.4, 0.3],
            ...     "metrics": [
            ...         {"accuracy": 0.80, "f1_score": 0.78},
            ...         {"accuracy": 0.85, "f1_score": 0.83},
            ...         {"accuracy": 0.90, "f1_score": 0.88},
            ...     ]
            ... }
            >>> print_final_analysis("FedBagging", metrics_history)
        """
        print("\n" + "=" * 80)
        print(f"ANÁLISE FINAL - Estratégia: {strategy_name}")
        print("=" * 80)

        # Infere número de rounds se não fornecido
        if num_rounds is None:
            if "losses" in metrics_history:
                num_rounds = len(metrics_history["losses"])
            elif "metrics" in metrics_history:
                num_rounds = len(metrics_history["metrics"])
            else:
                num_rounds = 0

        print(f"\nNúmero de Rounds: {num_rounds}")

        # Análise de Loss (se disponível)
        if "losses" in metrics_history and metrics_history["losses"]:
            losses = metrics_history["losses"]
            print("\n" + "-" * 80)
            print("EVOLUÇÃO DO LOSS:")
            print(f"  Loss Inicial (Round 1):  {losses[0]:.6f}")
            print(f"  Loss Final (Round {len(losses)}):    {losses[-1]:.6f}")
            print(f"  Redução Total:           {losses[0] - losses[-1]:.6f} ({(1 - losses[-1]/losses[0])*100:.2f}%)")
            print(f"  Loss Mínimo:             {min(losses):.6f} (Round {losses.index(min(losses)) + 1})")

        # Análise de Métricas (se disponível)
        if "metrics" in metrics_history and metrics_history["metrics"]:
            metrics_list = metrics_history["metrics"]

            print("\n" + "-" * 80)
            print("EVOLUÇÃO DAS MÉTRICAS:")

            # Extrai métricas numéricas
            metric_names = ["accuracy", "precision", "recall", "f1_score", "auc", "specificity"]

            for metric_name in metric_names:
                # Coleta valores válidos
                values = []
                for m in metrics_list:
                    if isinstance(m, dict) and metric_name in m:
                        val = m[metric_name]
                        if val is not None:
                            values.append(val)

                if values:
                    print(f"\n  {metric_name.upper().replace('_', ' ')}:")
                    print(f"    Inicial: {values[0]:.4f}")
                    print(f"    Final:   {values[-1]:.4f}")
                    print(f"    Melhoria: {(values[-1] - values[0]):.4f} ({((values[-1] - values[0])/values[0])*100:+.2f}%)")
                    print(f"    Máximo:  {max(values):.4f} (Round {values.index(max(values)) + 1})")

        # Resumo Final
        print("\n" + "=" * 80)
        if "metrics" in metrics_history and metrics_history["metrics"]:
            final_metrics = metrics_history["metrics"][-1]
            print("MÉTRICAS FINAIS:")
            if isinstance(final_metrics, dict):
                for key, value in sorted(final_metrics.items()):
                    if isinstance(value, (int, float)) and value is not None:
                        print(f"  {key}: {value:.4f}")
        print("=" * 80 + "\n")

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

        # Identifica métricas numéricas (incluindo specificity)
        numeric_keys = ["accuracy", "precision", "recall", "f1_score", "auc", "specificity"]
        aggregated = {}

        for key in numeric_keys:
            # Filtra valores None para specificity (pode ser None em multiclasse)
            values = [
                m[key] for m in metrics_list
                if key in m and m[key] is not None and isinstance(m[key], (int, float))
            ]

            if values:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        return aggregated
