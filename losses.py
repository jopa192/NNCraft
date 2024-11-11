import numpy as np

    
class BinaryCrossEntropyLoss:
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        
        losses = -(y_true * np.log(y_pred) + (1.-y_true) * np.log(1.-y_pred))
        loss = np.mean(losses, axis=-1)
        
        self.output = loss
        
        self.y_pred = y_pred
        self.y_true = y_true      
    
    def backward(self) -> np.ndarray:        
        n_outputs = self.y_pred.shape[0]
        
        self.d_output = -((self.y_true / self.y_pred) - ((1 - self.y_true) / (1 - self.y_pred))) / n_outputs
    
    def __str__(self):
        return "[Binary Cross-Entropy]"
    
    
class CategoricalCrossEntropyLoss:
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        
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
        self.y_pred = y_pred
        self.y_true = y_true

        self.output = loss
    
    def backward(self) -> np.ndarray:
        # Number of samples
        N = self.y_pred.shape[0]
        
        # Copy the softmax output
        grad = self.y_pred.copy()
        
        # Subtract 1 from the true class probabilities
        grad[self.y_true.astype(bool)] -= 1
        
        # Average gradient over the batch
        self.d_output = grad / N


class MeanSquaredError:
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        self.output = np.mean((y_pred - y_true) ** 2)
        
        self.y_pred = y_pred
        self.y_true = y_true
    
    def backward(self) -> None:
        N = self.y_pred.shape[0]
        self.d_output = (2 / N) * (self.y_pred - self.y_true)
      
        
class MeanAbsoluteError:
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        self.output = np.mean(np.abs(y_true - y_pred))
        
        self.y_pred = y_pred
        self.y_true = y_true
    
    def backward(self) -> None:
        N = self.y_true.shape[0]
    
        self.d_output = -((self.y_true - self.y_pred) / (abs(self.y_true - self.y_pred) + 10**-100)) / N
        

class HuberLoss:
    
    def __init__(self, delta: float = 1.0):
        self.delta = delta
        
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        # Compute Huber loss
        loss = np.where(abs_error <= self.delta,
                        0.5 * error**2,  
                        self.delta * (abs_error - 0.5 * self.delta))  
        self.output = np.mean(loss)  
        
        self.y_pred = y_pred
        self.y_true = y_true
    
    def backward(self) -> None:
        error = self.y_pred - self.y_true
        abs_error = np.abs(error)
        
        grad = np.where(abs_error <= self.delta,
                        error,
                        self.delta * np.sign(error)) 
        
        self.d_output = grad / self.y_pred.shape[0]