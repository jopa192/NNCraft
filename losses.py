import numpy as np


class Loss:
    """Base class for all loss functions."""
    
    def __init__(self, **params):
        self.params = params
    
    def calculate_loss(self, y_pred: np.ndarray, y_true: np.ndarray, training: bool = True) -> np.float64:
        """
        Calculating loss based on predictions and targets. Needs to be implemented by subclasses.
        
        Args:
        - y_pred: Predicted values (e.g., probabilities, outputs from the network).
        - y_true: True values.
        """
        raise NotImplementedError("The calculate loss method must be implemented by the subclass.")
    
    def backward(self) -> np.ndarray:
        """
        Backward pass for loss gradient calculation. Needs to be implemented by subclasses.
        
        Returns:
        - Gradient of the loss with respect to the input (same shape as y_pred).
        """
        raise NotImplementedError("The backward method must be implemented by the subclass.")
    
    def __str__(self):
        return f"[{self.__class__.__name__}]"


    
class BinaryCrossEntropyLoss(Loss):
    
    """Used for binary classification tasks, where each prediction is either 0 or 1.
    """
    
    def __init__(self):
        super().__init__()
    
    def calculate_loss(self, y_pred: np.ndarray, y_true: np.ndarray, training: bool = True) -> np.float64:
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        
        losses = -(y_true * np.log(y_pred) + (1.-y_true) * np.log(1.-y_pred))
        loss = np.mean(losses, axis=0).item()
        
        if training:
            self.y_pred = y_pred
            self.y_true = y_true      
            
        return loss
    
    def backward(self) -> np.ndarray:        
        n_outputs = self.y_pred.shape[0]
        
        self.d_output = -((self.y_true / self.y_pred) - ((1 - self.y_true) / (1 - self.y_pred))) / n_outputs
    
    
class CategoricalCrossEntropyLoss(Loss):
    
    """Used for multi-class classification tasks, where each prediction corresponds to one class out of multiple classes.
    """
    
    def __init__(self):
        super().__init__()
    
    def calculate_loss(self, y_pred: np.ndarray, y_true: np.ndarray, training: bool = True) -> np.float64:
        
        def softmax(x: np.ndarray) -> np.ndarray:
            exp_e = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
            return exp_e / np.sum(exp_e, axis=1, keepdims=True)
        
        # Apply softmax to logits
        y_pred = softmax(y_pred)
        
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        
        # Calculate cross-entropy loss
        N = y_pred.shape[0]
        log_likelihood = -np.log(y_pred[y_true.astype(bool)])
        loss = np.sum(log_likelihood) / N
        
        # Store for backward pass
        if training:
            self.y_pred = y_pred
            self.y_true = y_true

        return loss
    
    def backward(self) -> np.ndarray:
        # Number of samples
        N = self.y_pred.shape[0]
        
        # Copy the softmax output
        grad = self.y_pred.copy()
        
        # Subtract 1 from the true class probabilities
        grad[self.y_true.astype(bool)] -= 1
        
        # Average gradient over the batch
        self.d_output = grad / N


class MeanSquaredError(Loss):
    
    """Used for regression tasks, measuring the average squared difference between predicted and true values.
    """
    
    def __init__(self):
        super().__init__()
    
    def calculate_loss(self, y_pred: np.ndarray, y_true: np.ndarray, training: bool = True) -> np.float64:
        loss = np.mean((y_pred - y_true) ** 2)
        
        if training:
            self.y_pred = y_pred
            self.y_true = y_true
        
        return loss
    
    def backward(self) -> None:
        N = self.y_pred.shape[0]
        self.d_output = (2 / N) * (self.y_pred - self.y_true)
      
        
class MeanAbsoluteError(Loss):
    
    """Used for regression tasks, it calculates the average of the absolute differences between predicted and true values.
    """
    
    def __init__(self):
        super().__init__()
    
    def calculate_loss(self, y_pred: np.ndarray, y_true: np.ndarray, training: bool = True) -> None:
        loss = np.mean(np.abs(y_true - y_pred))
        
        if training:
            self.y_pred = y_pred
            self.y_true = y_true
            
        return loss
    
    def backward(self) -> None:
        N = self.y_true.shape[0]
    
        self.d_output = -((self.y_true - self.y_pred) / (abs(self.y_true - self.y_pred) + 10**-100)) / N
        

class HuberLoss(Loss):
    
    def __init__(self, delta):
        """A hybrid loss function that combines the best features of MSE and MAE, providing robustness to outliers while remaining differentiable at all points.

        Args:
            delta (float, optional): Transition point between MSE and MAE behavior. Defaults to 1.0.
        """
        super().__init__(delta=delta)
        
        self.delta = delta
        
    def calculate_loss(self, y_pred: np.ndarray, y_true: np.ndarray, training: bool = True) -> None:
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        # Compute Huber loss
        loss = np.where(abs_error <= self.delta,
                        0.5 * error**2,  
                        self.delta * (abs_error - 0.5 * self.delta))  
        loss = np.mean(loss)  
        
        if training:
            self.y_pred = y_pred
            self.y_true = y_true
            
        return loss
    
    def backward(self) -> None:
        error = self.y_pred - self.y_true
        abs_error = np.abs(error)
        
        grad = np.where(abs_error <= self.delta,
                        error,
                        self.delta * np.sign(error)) 
        
        self.d_output = grad / self.y_pred.shape[0]