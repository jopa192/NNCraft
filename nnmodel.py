import numpy as np
import layers
import losses
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self) -> None:
        self.layers = []
        self.loss = None
        
    def add_layer(self, layer) -> None:
        if len(self.layers) == 0 and not hasattr(layer, "weights"):
            raise ValueError("First layer must be Linear")
        self.layers.append(layer)
        
    def config(self, loss_func) -> None:
        self.loss = loss_func
        
    def train(self, X: np.ndarray, y: np.ndarray, n_epochs: int, 
              lr: float=1e-3, print_every: int=1) -> None:
        
        for epoch in range(1, n_epochs+1):
            output = self.forward(X)
            
            self.loss.forward(output, y)
            
            self.backward()
            
            for layer in self.layers:
                if hasattr(layer, "d_weights"):
                    layer.weights -= lr * layer.d_weights
                    layer.biases  -= lr * layer.d_biases
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch} : loss {np.mean(self.loss.output)}")
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        output = inputs.copy()
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output
    
    def backward(self) -> None:
        self.loss.backward()
        d_output = self.loss.d_output
        for layer in reversed(self.layers):
            layer.backward(d_output)
            d_output = layer.d_output
            
    def __call__(self, X: np.ndarray):
        return self.forward(X)