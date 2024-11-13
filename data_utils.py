import numpy as np
from typing import Generator, Tuple


class DataLoader:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False):
        """Initializes data loader.

        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Target values (labels).
            batch_size (int): Number of samples per batch.
            shuffle (bool, optional): Whether to shuffle the data before each epoch. Defaults to False.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.indices = np.arange(self.n_samples)

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Creates an iterator that yields batches of data."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start in range(0, self.n_samples, self.batch_size):
            end = start + self.batch_size
            batch_indices = self.indices[start:end]
            yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size


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

    
def partition_data(X: np.ndarray, y: np.ndarray, test_size: float, random_seed=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """plits the dataset into training and test sets.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.
        test_size (float): The proportion of the dataset to include in the test split.
        random_seed (_type_, optional): The random seed for shuffling the data. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The split data arrays (X_train, X_test, y_train, y_test).
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get the number of samples
    n_samples = X.shape[0]
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    # Determine the split point
    test_set_size = int(n_samples * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    # Split the data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test