import numpy as np


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes)[y]


def normalize(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))