import numpy as np


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Converts a categorical label array (y) into a one-hot encoded matrix. This is often used in classification tasks where each label is represented as a vector, where only the index corresponding to the class is set to 1, and all other indices are set to 0.

    Args:
        y (np.ndarray): An array of integer labels (e.g., [0, 1, 2] for 3 classes).
        n_classes (int): Number of unique target classes.

    Returns:
        np.ndarray: A one-hot encoded matrix, where each row corresponds to one input label, and each label is represented as a binary vector.
    """
    return np.eye(n_classes)[y]


def normalize(X: np.ndarray) -> np.ndarray:
    """Normalizes an input array (x) by scaling its values to a range between 0 and 1. This is a common preprocessing step in machine learning to ensure features are on a similar scale, which can improve model performance.

    Args:
        X (np.ndarray): An array of numerical data (e.g., features or values).

    Returns:
        np.ndarray: The normalized array where the minimum value is 0 and the maximum value is 1.
    """
    
    return (X - np.min(X)) / (np.max(X) - np.min(X))