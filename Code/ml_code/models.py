"""
Modelos de Machine Learning para Federated Learning
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pickle


class MLModel:
    """Classe base para modelos de ML"""
    
    def __init__(self, model_name, model_params=None):
        self.model_name = model_name
        self.model = self._create_model(model_params)
        self.is_fitted = False
        
    def _create_model(self, params):
        """Cria o modelo baseado no nome"""
        if params is None:
            params = {}
            
        models = {
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42, **params),
            'svm': SVC(kernel='rbf', probability=True, random_state=42, **params),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, **params),
            'knn': KNeighborsClassifier(n_neighbors=5, **params),
            'naive_bayes': GaussianNB(**params)
        }
        
        if self.model_name not in models:
            raise ValueError(f"Modelo {self.model_name} não suportado")
            
        return models[self.model_name]
    
    def get_parameters(self):
        """Retorna os parâmetros do modelo como array numpy"""
        if not self.is_fitted:
            # Se o modelo não foi treinado, retorna parâmetros iniciais
            return self._get_initial_parameters()
        
        return self._serialize_parameters()
    
    def set_parameters(self, parameters):
        """Define os parâmetros do modelo"""
        self._deserialize_parameters(parameters)
        self.is_fitted = True
    
    def _get_initial_parameters(self):
        """Retorna parâmetros iniciais para o modelo"""
        # Treina rapidamente com dados fictícios para obter estrutura
        dummy_X = np.random.rand(10, 784)
        dummy_y = np.random.randint(0, 10, 10)
        self.model.fit(dummy_X, dummy_y)
        params = self._serialize_parameters()
        self.is_fitted = False
        return params
    
    def _serialize_parameters(self):
        """Serializa os parâmetros do modelo"""
        model_bytes = pickle.dumps(self.model)
        return [np.frombuffer(model_bytes, dtype=np.uint8)]
    
    def _deserialize_parameters(self, parameters):
        """Deserializa os parâmetros do modelo"""
        model_bytes = parameters[0].tobytes()
        self.model = pickle.loads(model_bytes)
    
    def fit(self, X, y):
        """Treina o modelo"""
        X_flat = X.reshape(X.shape[0], -1)  # Flatten das imagens
        self.model.fit(X_flat, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Faz predições"""
        X_flat = X.reshape(X.shape[0], -1)  # Flatten das imagens
        return self.model.predict(X_flat)
    
    def predict_proba(self, X):
        """Retorna probabilidades das predições"""
        X_flat = X.reshape(X.shape[0], -1)  # Flatten das imagens
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_flat)
        else:
            # Para modelos que não têm predict_proba, usa decision_function
            scores = self.model.decision_function(X_flat)
            # Converte scores para probabilidades usando softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def evaluate(self, X, y):
        """Avalia o modelo"""
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        try:
            probabilities = self.predict_proba(X)
            # Calcula log loss
            loss = log_loss(y, probabilities)
        except:
            # Se não conseguir calcular log_loss, usa erro de classificação
            loss = 1.0 - accuracy
            
        return loss, accuracy


def create_model(model_name, params=None):
    """Factory function para criar modelos"""
    return MLModel(model_name, params)


# Lista de modelos disponíveis
AVAILABLE_MODELS = [
    'random_forest',
    'svm', 
    'logistic_regression',
    'knn',
    'naive_bayes'
]