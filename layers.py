import numpy as np    

class Linear:
    
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.randn(input_size, output_size) * .01
        self.biases = np.zeros((1, output_size)) 
        
    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs.copy()
        self.output = inputs @ self.weights + self.biases
    
    def backward(self, d_inputs: np.ndarray) -> None:
        self.d_weights = self.inputs.T @ d_inputs
        self.d_biases = np.sum(d_inputs, axis=0, keepdims=True)
        
        self.d_output = d_inputs @ self.weights.T
    
    def __str__(self) -> str:
        return f"[Dense Layer, input_size: {self.input_size}, n_neurons: {self.output_size}]"
    

class Sigmoid:
        
    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, d_inputs: np.ndarray) -> None:
        sigmoid_derivative = self.output * (1 - self.output)
        self.d_output = d_inputs * sigmoid_derivative 
    
    def __str__(self) -> str:
        return "[Sigmoid Activation]"
    
    
class ReLU:
            
    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.output = np.maximum(0., inputs)
        
    def backward(self, d_inputs: np.ndarray) -> None:
        relu_derivative = np.where(self.inputs > 0, 1, 0)
        self.d_output = d_inputs * relu_derivative
    
    def __str__(self) -> str:
        return "[ReLU Activation]"
    
    