import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import RANDOM_STATE, NUM_CLIENTS

def load_and_partition(dataset_name="breast_cancer", num_clients=NUM_CLIENTS):
    """Carrega, normaliza e particiona o dataset."""
    X, y = load_breast_cancer(return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.18, random_state=RANDOM_STATE, stratify=y_train
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Divide o treino igualmente entre os clientes
    idx = np.random.RandomState(RANDOM_STATE).permutation(len(X_train))
    splits = np.array_split(idx, num_clients)
    partitions = {cid: (X_train[s], y_train[s]) for cid, s in enumerate(splits)}

    info = {"num_classes": 2, "total_samples": len(X), "num_features": X.shape[1]}
    return partitions, (X_val, y_val), (X_test, y_test), info