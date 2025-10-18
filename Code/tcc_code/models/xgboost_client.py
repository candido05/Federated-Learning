"""
Cliente Federated Learning para XGBoost.

Este módulo implementa o cliente FL específico para o framework XGBoost,
incluindo treinamento incremental, serialização e predição.
"""

import xgboost as xgb
import numpy as np
import tempfile
import os
from typing import Tuple, Optional
from datetime import datetime

from .base_client import BaseFLClient


class XGBoostClient(BaseFLClient):
    """
    Cliente Federated Learning para modelos XGBoost.

    Implementa treinamento incremental usando xgb.train com suporte a continuação
    de modelo global, serialização em formato JSON e predição com probabilidades.

    Attributes:
        Herda todos os atributos de BaseFLClient.
    """

    def train_local_model(self, global_model_bytes: Optional[bytes]) -> object:
        """
        Treina modelo XGBoost localmente, continuando do modelo global se existir.

        Args:
            global_model_bytes: Bytes do modelo global (None no primeiro round).

        Returns:
            xgb.Booster: Modelo XGBoost treinado.

        Raises:
            Exception: Se houver erro no treinamento.
        """
        try:
            # Desempacota dados de treino
            X_train, y_train = self.train_data

            # Cria DMatrix para XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)

            # Carrega modelo global se existir
            xgb_model = None
            if global_model_bytes is not None:
                try:
                    xgb_model = self.load_model_from_bytes(global_model_bytes)
                    if self.experiment_logger:
                        self.experiment_logger.logger.info(
                            f"Cliente {self.client_id}: Modelo global carregado para continuar treinamento"
                        )
                except Exception as e:
                    if self.experiment_logger:
                        self.experiment_logger.logger.warning(
                            f"Cliente {self.client_id}: Falha ao carregar modelo global: {e}. "
                            "Iniciando treinamento do zero."
                        )

            # Configura parâmetros com suporte a GPU
            params = self.params.copy()

            # Detecta e configura GPU se disponível
            try:
                if xgb.get_config().get('use_rmm', False) or 'tree_method' not in params:
                    # Tenta usar GPU
                    gpu_params = {
                        'tree_method': 'gpu_hist',
                        'predictor': 'gpu_predictor'
                    }
                    params.update(gpu_params)

                    if self.experiment_logger:
                        self.experiment_logger.logger.info(
                            f"Cliente {self.client_id}: Usando GPU para treinamento XGBoost"
                        )
            except Exception:
                # Fallback para CPU
                if 'tree_method' not in params:
                    params['tree_method'] = 'hist'

                if self.experiment_logger:
                    self.experiment_logger.logger.info(
                        f"Cliente {self.client_id}: Usando CPU para treinamento XGBoost"
                    )

            # Treina modelo (continua do global se existir)
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=self.num_local_round,
                xgb_model=xgb_model,  # Continua do modelo global
                verbose_eval=False
            )

            if self.experiment_logger:
                self.experiment_logger.logger.info(
                    f"Cliente {self.client_id}: Treinamento XGBoost concluído "
                    f"({self.num_local_round} rounds)"
                )

            return bst

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro no treinamento XGBoost: {e}"
            if self.experiment_logger:
                self.experiment_logger.logger.error(error_msg)
            raise Exception(error_msg)

    def save_model_bytes(self, model: xgb.Booster) -> bytes:
        """
        Serializa modelo XGBoost para bytes usando formato JSON.

        Args:
            model: Modelo XGBoost (Booster).

        Returns:
            bytes: Modelo serializado em formato JSON.

        Raises:
            Exception: Se houver erro na serialização.
        """
        temp_path = None
        try:
            # Cria arquivo temporário único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(
                temp_dir,
                f"xgboost_client_{self.client_id}_{timestamp}.json"
            )

            # Salva modelo em formato JSON
            model.save_model(temp_path)

            # Lê bytes do arquivo
            with open(temp_path, 'rb') as f:
                model_bytes = f.read()

            if self.experiment_logger:
                self.experiment_logger.logger.debug(
                    f"Cliente {self.client_id}: Modelo XGBoost serializado "
                    f"({len(model_bytes)} bytes)"
                )

            return model_bytes

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro ao serializar modelo XGBoost: {e}"
            if self.experiment_logger:
                self.experiment_logger.logger.error(error_msg)
            raise Exception(error_msg)

        finally:
            # Limpa arquivo temporário
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    if self.experiment_logger:
                        self.experiment_logger.logger.warning(
                            f"Cliente {self.client_id}: Falha ao remover arquivo temporário "
                            f"{temp_path}: {e}"
                        )

    def load_model_from_bytes(self, model_bytes: bytes) -> xgb.Booster:
        """
        Carrega modelo XGBoost a partir de bytes.

        Args:
            model_bytes: Bytes do modelo em formato JSON.

        Returns:
            xgb.Booster: Modelo XGBoost carregado.

        Raises:
            Exception: Se houver erro no carregamento.
        """
        temp_path = None
        try:
            # Cria arquivo temporário único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(
                temp_dir,
                f"xgboost_load_{self.client_id}_{timestamp}.json"
            )

            # Escreve bytes no arquivo temporário
            with open(temp_path, 'wb') as f:
                f.write(model_bytes)

            # Cria Booster e carrega modelo
            bst = xgb.Booster()
            bst.load_model(temp_path)

            if self.experiment_logger:
                self.experiment_logger.logger.debug(
                    f"Cliente {self.client_id}: Modelo XGBoost carregado "
                    f"({len(model_bytes)} bytes)"
                )

            return bst

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro ao carregar modelo XGBoost: {e}"
            if self.experiment_logger:
                self.experiment_logger.logger.error(error_msg)
            raise Exception(error_msg)

        finally:
            # Limpa arquivo temporário
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    if self.experiment_logger:
                        self.experiment_logger.logger.warning(
                            f"Cliente {self.client_id}: Falha ao remover arquivo temporário "
                            f"{temp_path}: {e}"
                        )

    def predict(self, model: xgb.Booster, X: np.ndarray) -> np.ndarray:
        """
        Realiza predição com modelo XGBoost.

        Args:
            model: Modelo XGBoost (Booster).
            X: Features para predição.

        Returns:
            np.ndarray: Probabilidades preditas (shape: [n_samples, n_classes]).

        Raises:
            Exception: Se houver erro na predição.
        """
        try:
            # Cria DMatrix para predição
            dtest = xgb.DMatrix(X)

            # Realiza predição (retorna probabilidades)
            predictions = model.predict(dtest)

            # Se for classificação binária, XGBoost retorna apenas probabilidade da classe 1
            # Precisamos criar array [prob_classe_0, prob_classe_1]
            if len(predictions.shape) == 1:
                prob_class_1 = predictions.reshape(-1, 1)
                prob_class_0 = 1 - prob_class_1
                predictions = np.hstack([prob_class_0, prob_class_1])

            if self.experiment_logger:
                self.experiment_logger.logger.debug(
                    f"Cliente {self.client_id}: Predição XGBoost realizada "
                    f"({X.shape[0]} amostras)"
                )

            return predictions

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro na predição XGBoost: {e}"
            if self.experiment_logger:
                self.experiment_logger.logger.error(error_msg)
            raise Exception(error_msg)
