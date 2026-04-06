import pickle
import numpy as np
from xgboost import XGBClassifier
from config import RANDOM_STATE, LOCAL_EPOCHS

def train_model(model_type, X, y, num_classes, warm_start_model=None):
    """Treina XGBoost com warm start opcional via booster anterior."""
    m = XGBClassifier(
        n_estimators=LOCAL_EPOCHS, 
        max_depth=4, 
        learning_rate=0.1,
        random_state=RANDOM_STATE, 
        eval_metric="logloss"
    )
    
    # Se houver modelo anterior, extrai o booster para continuar o treino
    kw = {"xgb_model": warm_start_model.get_booster()} if warm_start_model else {}
    m.fit(X, y, **kw)
    return m

def serialize_model(model) -> bytes:
    """Transforma o modelo em bytes para envio via Flower."""
    return pickle.dumps(model)

def deserialize_model(data: bytes):
    """Reconstroi o modelo a partir dos bytes recebidos."""
    return pickle.loads(data)