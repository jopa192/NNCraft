import numpy as np    

class Layer:
    """Base class for all layers in the neural network."""
    
    def forward(self, inputs: np.ndarray) -> None:
        """Forward pass, should be implemented by subclasses."""
        raise NotImplementedError("Forward pass should be implemented by subclasses.")

    def backward(self, d_inputs: np.ndarray) -> None:
        """Backward pass, should be implemented by subclasses."""
        raise NotImplementedError("Backward pass should be implemented by subclasses.")

    def __repr__(self) -> str:
        """String representation of the layer."""
        raise NotImplementedError


class Dense(Layer):
    
    def __init__(self, input_size: int, output_size: int, l1_lambda: float = 0.) -> None:
        """Initialize the Dense Layer with weights and biases

        Args:
            input_size (int): Number of inputs to the layer
            output_size (int): Number of outputs from the layer (i.e., number of neurons)
        """
                
        self.weights: np.ndarray = np.random.randn(input_size, output_size) * np.sqrt(1. / input_size)
        self.biases: np.ndarray = np.zeros((1, output_size)) 
        
        self.l1_lambda: float = l1_lambda
        
    def forward(self, inputs: np.ndarray) -> None:
        """Forward pass for the Dense layer (dot product of inputs  weights plus biases)

        Args:
            inputs (np.ndarray): Input data (shape: N x input_size)
        """
        
        self.inputs = inputs.copy()
        self.output = inputs @ self.weights + self.biases
    
    def backward(self, d_inputs: np.ndarray) -> None:
        """Calculating gradients of loss with respect to weights, biases and inputs separately

        Args:
            d_inputs (np.ndarray): Gradient of the loss with respect to the layer's output (shape: N x output_size)
        """
        
        self.d_weights = self.inputs.T @ d_inputs
        self.d_biases = np.sum(d_inputs, axis=0, keepdims=True)
        
        if self.l1_lambda > 0:
            self.d_weights += self.l1_lambda * np.sign(self.weights)
        
        self.d_output = d_inputs @ self.weights.T
        
    def l1_regularize(self):
        return self.l1_lambda * np.sum(np.abs(self.weights))
    
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
    

class Dropout(Layer):
    
    def __init__(self, dropout_rate: float) -> None:
        """A regularization technique used in neural networks to prevent overfitting. During training, it randomly "drops out" (sets to zero) a fraction of the neurons.

        Args:
            dropout_rate (float): The fraction of neurons to drop
        """
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> None:
        """Droping fraction of neurons.

        Args:
            inputs (np.ndarray): Input data (shape: N x input_size)
            training (bool, optional): Only drops neurons while training process, no dropout during inference. Defaults to True.
        """
        if not training:
            self.output = inputs
            return
        
        self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape)
        self.output = inputs * self.mask / (1 - self.dropout_rate)        
    
    def backward(self, d_inputs: np.ndarray) -> None:
        self.d_output = d_inputs * self.mask  