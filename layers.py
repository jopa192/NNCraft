import numpy as np    

class Layer:
    """Base class for all layers in the neural network."""
    
    def forward(self, inputs: np.ndarray) -> None:
        """Forward pass, should be implemented by subclasses."""
        raise NotImplementedError

    def backward(self, d_inputs: np.ndarray) -> None:
        """Backward pass, should be implemented by subclasses."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """String representation of the layer."""
        raise NotImplementedError


class Linear(Layer):
    
    def __init__(self, input_size: int, output_size: int) -> None:
        """Initialize the Linear Layer with weights and biases

        Args:
            input_size (int): Number of inputs to the layer
            output_size (int): Number of outputs from the layer (i.e., number of neurons)
        """
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.randn(input_size, output_size) * .01
        self.biases = np.zeros((1, output_size)) 
        
    def forward(self, inputs: np.ndarray) -> None:
        """Forward pass for the linear layer

        Args:
            inputs (np.ndarray): Input data (shape: N x input_size)
        """
        
        self.inputs = inputs.copy()
        self.output = inputs @ self.weights + self.biases
    
    def backward(self, d_inputs: np.ndarray) -> None:
        """Backward pass for the linear layer to compute gradients

        Args:
            d_inputs (np.ndarray): Gradient of the loss with respect to the layer's output (shape: N x output_size)
        """
        
        self.d_weights = self.inputs.T @ d_inputs
        self.d_biases = np.sum(d_inputs, axis=0, keepdims=True)
        
        self.d_output = d_inputs @ self.weights.T
    
    def __repr__(self) -> str:
        return f"[Dense Layer, input_size: {self.input_size}, n_neurons: {self.output_size}]"
    

class Sigmoid(Layer):
    
    """Used for binary classification or output between 0 and 1.
    """
        
    def forward(self, inputs: np.ndarray) -> None:
        """f(x) = 1 / (1 + exp(-x))

        Args:
            inputs (np.ndarray): Input data (shape: N x input_size)
        """
        
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, d_inputs: np.ndarray) -> None:
        """Derivative of the sigmoid activation: f'(x) = sigmoid(x) * (1 - sigmoid(x))

        Args:
            d_inputs (np.ndarray): Gradient of the loss w.r.t the output (shape: N x input_size)
        """
        
        sigmoid_derivative = self.output * (1 - self.output)
        self.d_output = d_inputs * sigmoid_derivative 
    
    def __repr__(self) -> str:
        return "[Sigmoid Activation]"
    
    
class ReLU(Layer):
    
    """Sets negative values to zero and keeps positive values unchanged.
    """
            
    def forward(self, inputs: np.ndarray) -> None:
        """f(x) = max(0, x)

        Args:
            inputs (np.ndarray): Input data (shape: N x input_size)
        """
        
        self.inputs = inputs
        self.output = np.maximum(0., inputs)
        
    def backward(self, d_inputs: np.ndarray) -> None:
        """Derivative of the ReLU activation: f'(x) = 1 if x > 0 else 0

        Args:
            d_inputs (np.ndarray): Gradient of the loss w.r.t the output (shape: N x input_size)
        """
        
        relu_derivative = np.where(self.inputs > 0, 1, 0)
        self.d_output = d_inputs * relu_derivative
    
    def __repr__(self) -> str:
        return "[ReLU Activation]"
    
    