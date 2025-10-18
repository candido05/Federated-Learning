"""
Cliente Federated Learning para CatBoost.

Este módulo implementa o cliente FL específico para o framework CatBoost,
incluindo treinamento incremental, serialização e predição.
"""

from catboost import CatBoost, Pool
import numpy as np
import tempfile
import os
from typing import Optional
from datetime import datetime

from .base_client import BaseFLClient


class CatBoostClient(BaseFLClient):
    """
    Cliente Federated Learning para modelos CatBoost.

    Implementa treinamento incremental usando CatBoost.fit com suporte a continuação
    de modelo global via init_model, serialização em formato CBM e predição de probabilidades.

    Attributes:
        Herda todos os atributos de BaseFLClient.
    """

    def train_local_model(self, global_model_bytes: Optional[bytes]) -> object:
        """
        Treina modelo CatBoost localmente, continuando do modelo global se existir.

        Args:
            global_model_bytes: Bytes do modelo global (None no primeiro round).

        Returns:
            CatBoost: Modelo CatBoost treinado.

        Raises:
            Exception: Se houver erro no treinamento.
        """
        try:
            # Desempacota dados de treino
            X_train, y_train = self.train_data

            # Cria Pool para CatBoost
            train_pool = Pool(X_train, label=y_train)

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
                if result.returncode == 0 and 'task_type' not in params:
                    params['task_type'] = 'GPU'
                    if self.experiment_logger:
                        self.experiment_logger.logger.info(
                            f"Cliente {self.client_id}: Usando GPU para treinamento CatBoost"
                        )
            except Exception:
                # Fallback para CPU
                if 'task_type' not in params:
                    params['task_type'] = 'CPU'
                if self.experiment_logger:
                    self.experiment_logger.logger.info(
                        f"Cliente {self.client_id}: Usando CPU para treinamento CatBoost"
                    )

            # Define iterações
            if 'iterations' not in params:
                params['iterations'] = self.num_local_round

            # Desabilita logging verboso
            if 'verbose' not in params:
                params['verbose'] = False

            # Cria modelo CatBoost
            model = CatBoost(params)

            # Treina modelo (continua do global se existir)
            if init_model is not None:
                # Continua treinamento do modelo global
                model.fit(
                    train_pool,
                    init_model=init_model,
                    verbose=False
                )
            else:
                # Treina do zero
                model.fit(
                    train_pool,
                    verbose=False
                )

            if self.experiment_logger:
                self.experiment_logger.logger.info(
                    f"Cliente {self.client_id}: Treinamento CatBoost concluído "
                    f"({params.get('iterations', self.num_local_round)} iterações)"
                )

            return model

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro no treinamento CatBoost: {e}"
            if self.experiment_logger:
                self.experiment_logger.logger.error(error_msg)
            raise Exception(error_msg)

    def save_model_bytes(self, model: CatBoost) -> bytes:
        """
        Serializa modelo CatBoost para bytes usando formato CBM.

        Args:
            model: Modelo CatBoost.

        Returns:
            bytes: Modelo serializado em formato CBM.

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
                f"catboost_client_{self.client_id}_{timestamp}.cbm"
            )

            # Salva modelo em formato CBM
            model.save_model(temp_path, format="cbm")

            # Lê bytes do arquivo
            with open(temp_path, 'rb') as f:
                model_bytes = f.read()

            if self.experiment_logger:
                self.experiment_logger.logger.debug(
                    f"Cliente {self.client_id}: Modelo CatBoost serializado "
                    f"({len(model_bytes)} bytes)"
                )

            return model_bytes

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro ao serializar modelo CatBoost: {e}"
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

    def load_model_from_bytes(self, model_bytes: bytes) -> CatBoost:
        """
        Carrega modelo CatBoost a partir de bytes.

        Args:
            model_bytes: Bytes do modelo em formato CBM.

        Returns:
            CatBoost: Modelo CatBoost carregado.

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
                f"catboost_load_{self.client_id}_{timestamp}.cbm"
            )

            # Escreve bytes no arquivo temporário
            with open(temp_path, 'wb') as f:
                f.write(model_bytes)

            # Carrega modelo do arquivo
            model = CatBoost()
            model.load_model(temp_path, format="cbm")

            if self.experiment_logger:
                self.experiment_logger.logger.debug(
                    f"Cliente {self.client_id}: Modelo CatBoost carregado "
                    f"({len(model_bytes)} bytes)"
                )

            return model

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro ao carregar modelo CatBoost: {e}"
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

    def predict(self, model: CatBoost, X: np.ndarray) -> np.ndarray:
        """
        Realiza predição com modelo CatBoost.

        Args:
            model: Modelo CatBoost.
            X: Features para predição.

        Returns:
            np.ndarray: Probabilidades preditas (shape: [n_samples, n_classes]).

        Raises:
            Exception: Se houver erro na predição.
        """
        try:
            # Realiza predição (retorna probabilidades)
            predictions = model.predict(X, prediction_type='Probability')

            # CatBoost retorna shape correto automaticamente para classificação
            # Binária: [n_samples, 2]
            # Multiclasse: [n_samples, n_classes]

            if self.experiment_logger:
                self.experiment_logger.logger.debug(
                    f"Cliente {self.client_id}: Predição CatBoost realizada "
                    f"({X.shape[0]} amostras)"
                )

            return predictions

        except Exception as e:
            error_msg = f"Cliente {self.client_id}: Erro na predição CatBoost: {e}"
            if self.experiment_logger:
                self.experiment_logger.logger.error(error_msg)
            raise Exception(error_msg)
