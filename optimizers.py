import numpy as np
from typing import Type, List, Tuple
from layers import Layer, Linear


class Optimizer:
    def __init__(self, learning_rate: float, layers: List[Type[Layer]]) -> None:
        """algorithm used to change the attributes of neural network (like weights and learning rate) to reduce the losses.

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            layers (List[Type[Layer]]): Layers of the model.
        """
        self.learning_rate: float = learning_rate
        
        self.momentum_parameters: List[Tuple[Linear, np.ndarray, np.ndarray]] = []
        for layer in layers:
            if hasattr(layer, "weights"):
                self.momentum_parameters.append((layer, 
                                          np.zeros_like(layer.weights),
                                          np.zeros_like(layer.biases)))
        
    def gradient_step(self) -> None:
        """Gradient step, should be implemented by subclasses."""
        raise NotImplementedError("Gradient step should be implemented by subclasses.")


class SGD(Optimizer):
    def __init__(self, learning_rate: float, layers: List[Type[Layer]], momentum: float = 0) -> None:
        """Stochastic Gradient Descent updates the model's weights by taking small steps proportional to the negative gradient of the loss function

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            layers (List[Type[Layer]]): Layers of the model.
            momentum (float, optional): Momentum helps SGD accelerate by using past gradients to smooth out the steps taken. Defaults to 0.
        """
        super().__init__(learning_rate, layers)
        self.momentum: float = momentum        
                
    def gradient_step(self) -> None:
        """Updates the parameters (weights and biases) of a model based on the gradients calculated during backpropagation.
        """
        for i, (layer, velocity_weights, velocity_biases) in enumerate(self.momentum_parameters):
            # Calculate velocities with momentum
            velocity_weights = self.momentum * velocity_weights - self.learning_rate * layer.d_weights
            velocity_biases = self.momentum * velocity_biases - self.learning_rate * layer.d_biases
            
            # Update weights and biases
            layer.weights += velocity_weights
            layer.biases += velocity_biases
            
            # Store updated velocities back in the momentum_parameters list
            self.momentum_parameters[i] = (layer, velocity_weights, velocity_biases)