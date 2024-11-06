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
        self.layers.append(layer)
        
    def config(self, loss_func) -> None:
        self.loss = loss_func
        
    def train(self, X: np.ndarray, y: np.ndarray, n_epochs: int, 
              lr: float=1e-3, print_every: int=1) -> None:
        
        for epoch in range(1, n_epochs+1):
            output = self.forward(X)
            
            loss = self.loss.forward(output, y)
            accuracy = np.mean((output > 0.5).astype(int) == y).item()
            
            self.backward()
            
            for layer in self.layers:
                if hasattr(layer, "d_weights"):
                    layer.weights -= lr * layer.d_weights
                    layer.biases  -= lr * layer.d_biases
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch} : loss {np.mean(loss)}, accuracy {accuracy*100:.2f}%")
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        output = inputs.copy()
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output
    
    def backward(self) -> None:
        d_output = self.loss.backward()
        for layer in reversed(self.layers):
            layer.backward(d_output)
            d_output = layer.d_output
            




X, y = make_circles(n_samples=500, noise=0.1)
y = y.reshape(-1, 1)

n_features = X.shape[1]
n_output = y.shape[1]

model = NeuralNetwork()

model.add_layer(layers.Linear(n_features, 128))
model.add_layer(layers.ReLU())
model.add_layer(layers.Linear(128, 128))
model.add_layer(layers.ReLU())
model.add_layer(layers.Linear(128, n_output))
model.add_layer(layers.Sigmoid())

model.config(losses.BinaryCrossEntropyLoss())

model.train(X, y, 20000, 0.05, 1000)

output = model.forward(X)
preds = (output > 0.5).astype(int)

plt.figure(1)
plt.scatter(X[np.where(preds==0), 0], X[np.where(preds==0), 1], c="r")
plt.scatter(X[np.where(preds==1), 0], X[np.where(preds==1), 1], c="g")
plt.figure(2)
plt.scatter(X[np.where(y==0), 0], X[np.where(y==0), 1], c="r")
plt.scatter(X[np.where(y==1), 0], X[np.where(y==1), 1], c="g")
plt.show()