"""
Cliente Federated Learning para LightGBM.

Este módulo implementa o cliente FL específico para o framework LightGBM,
incluindo treinamento incremental, serialização e predição.
"""

import lightgbm as lgb
import numpy as np
import tempfile
import os
from typing import Optional
from datetime import datetime

from .base_client import BaseFLClient


class LightGBMClient(BaseFLClient):
    """
    Cliente Federated Learning para modelos LightGBM.

    Implementa treinamento incremental usando lgb.train com suporte a continuação
    de modelo global via init_model, serialização em formato texto e predição.

    Attributes:
        Herda todos os atributos de BaseFLClient.
    """

    def train_local_model(self, global_model_bytes: Optional[bytes]) -> object:
        """
        Treina modelo LightGBM localmente, continuando do modelo global se existir.

        Args:
            global_model_bytes: Bytes do modelo global (None no primeiro round).

        Returns:
            lgb.Booster: Modelo LightGBM treinado.

        Raises:
            Exception: Se houver erro no treinamento.
        """
        try:
            # Desempacota dados de treino
            X_train, y_train = self.train_data

            # Cria Dataset para LightGBM
            train_data = lgb.Dataset(X_train, label=y_train)

            # Carrega modelo global se existir
            init_model = None
            if global_model_bytes is not None:
                try:
                    init_model = self.load_model_from_bytes(global_model_bytes)
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
                import subprocess
                # Tenta detectar GPU NVIDIA
                result = subprocess.run(
                    ['nvidia-smi'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                if result.returncode == 0 and 'device' not in params:
                    params['device'] = 'gpu'
                    if 'gpu_platform_id' not in params:
                        params['gpu_platform_id'] = 0
                    if 'gpu_device_id' not in params:
                        params['gpu_device_id'] = 0

                    if self.experiment_logger:
                        self.experiment_logger.logger.info(
                            f"Cliente {self.client_id}: Usando GPU para treinamento LightGBM"
                        )
            except Exception:
                # Fallback para CPU
                if 'device' not in params:
                    params['device'] = 'cpu'
                if self.experiment_logger:
                    self.experiment_logger.logger.info(
                        f"Cliente {self.client_id}: Usando CPU para treinamento LightGBM"
                    )

            # Define modo verboso
            if 'verbose' not in params:
                params['verbose'] = -1  # Silencioso

            # Treina modelo (continua do global se existir)
            bst = lgb.train(
                params,
                train_data,
                num_boost_round=self.num_local_round,
                init_model=init_model,  # Continua do modelo global
                keep_training_booster=True  # Permite continuar treinamento
            )

            if self.experiment_logger:
                self.experiment_logger.logger.info(
                    f"Cliente {self.client_id}: Treinamento LightGBM concluído "
                    f"({self.num_local_round} rounds)"
                )

            return bst

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro no treinamento LightGBM: {e}"
            if self.experiment_logger:
                self.experiment_logger.logger.error(error_msg)
            raise Exception(error_msg)

    def save_model_bytes(self, model: lgb.Booster) -> bytes:
        """
        Serializa modelo LightGBM para bytes usando formato texto.

        Args:
            model: Modelo LightGBM (Booster).

        Returns:
            bytes: Modelo serializado em formato texto.

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
                f"lightgbm_client_{self.client_id}_{timestamp}.txt"
            )

            # Salva modelo em formato texto
            model.save_model(temp_path)

            # Lê bytes do arquivo
            with open(temp_path, 'rb') as f:
                model_bytes = f.read()

            if self.experiment_logger:
                self.experiment_logger.logger.debug(
                    f"Cliente {self.client_id}: Modelo LightGBM serializado "
                    f"({len(model_bytes)} bytes)"
                )

            return model_bytes

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro ao serializar modelo LightGBM: {e}"
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

    def load_model_from_bytes(self, model_bytes: bytes) -> lgb.Booster:
        """
        Carrega modelo LightGBM a partir de bytes.

        Args:
            model_bytes: Bytes do modelo em formato texto.

        Returns:
            lgb.Booster: Modelo LightGBM carregado.

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
                f"lightgbm_load_{self.client_id}_{timestamp}.txt"
            )

            # Escreve bytes no arquivo temporário
            with open(temp_path, 'wb') as f:
                f.write(model_bytes)

            # Carrega modelo do arquivo
            bst = lgb.Booster(model_file=temp_path)

            if self.experiment_logger:
                self.experiment_logger.logger.debug(
                    f"Cliente {self.client_id}: Modelo LightGBM carregado "
                    f"({len(model_bytes)} bytes)"
                )

            return bst

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro ao carregar modelo LightGBM: {e}"
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

    def predict(self, model: lgb.Booster, X: np.ndarray) -> np.ndarray:
        """
        Realiza predição com modelo LightGBM.

        Args:
            model: Modelo LightGBM (Booster).
            X: Features para predição.

        Returns:
            np.ndarray: Probabilidades preditas (shape: [n_samples, n_classes]).

        Raises:
            Exception: Se houver erro na predição.
        """
        try:
            # Realiza predição (retorna probabilidades)
            predictions = model.predict(X)

            # Se for classificação binária, LightGBM pode retornar apenas probabilidade da classe 1
            # Precisamos criar array [prob_classe_0, prob_classe_1]
            if len(predictions.shape) == 1:
                prob_class_1 = predictions.reshape(-1, 1)
                prob_class_0 = 1 - prob_class_1
                predictions = np.hstack([prob_class_0, prob_class_1])

            if self.experiment_logger:
                self.experiment_logger.logger.debug(
                    f"Cliente {self.client_id}: Predição LightGBM realizada "
                    f"({X.shape[0]} amostras)"
                )

            return predictions

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro na predição LightGBM: {e}"
            if self.experiment_logger:
                self.experiment_logger.logger.error(error_msg)
            raise Exception(error_msg)
