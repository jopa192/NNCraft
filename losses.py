import numpy as np
    
    
class BinaryCrossEntropyLoss:
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        
        self.y_pred = y_pred
        self.y_true = y_true      
        
        losses = -(y_true * np.log(y_pred) + (1.-y_true) * np.log(1.-y_pred))
        loss = np.mean(losses, axis=-1)
        
        return loss
    
    def backward(self) -> np.ndarray:        
        n_outputs = self.y_pred.shape[0]
        
        return -((self.y_true / self.y_pred) - ((1 - self.y_true) / (1 - self.y_pred))) / n_outputs