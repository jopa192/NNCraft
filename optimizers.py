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
        self.trainable: List[Linear] = [layer for layer in layers if hasattr(layer, "weights")]
        
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
        
        self.momentum_parameters: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.zeros_like(layer.weights), np.zeros_like(layer.biases)) for layer in self.trainable
        ]
            
                
    def gradient_step(self) -> None:
        """Updates the parameters (weights and biases) of a model based on the gradients calculated during backpropagation.
        """
        for i, (layer, (velocity_weights, velocity_biases)) in enumerate(zip(self.trainable, self.momentum_parameters)):
            # Calculate velocities with momentum
            velocity_weights = self.momentum * velocity_weights - self.learning_rate * layer.d_weights
            velocity_biases = self.momentum * velocity_biases - self.learning_rate * layer.d_biases
            
            # Update weights and biases
            layer.weights += velocity_weights
            layer.biases += velocity_biases
            
            # Store updated velocities back in the momentum_parameters list
            self.momentum_parameters[i] = (velocity_weights, velocity_biases)
            
    
class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float, layers: List[Type[Layer]], epsilon: float=1e-9) -> None:
        """Optimizer that adapts the learning rate for each parameter by accumulating the square of gradients over time

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            layers (List[Type[Layer]]): Layers of the model.
            epsilon (float, optional): A small constant to prevent division by zero when adjusting gradients. Defaults to 1e-9.
        """
        super().__init__(learning_rate, layers)
    
        self.e: float = epsilon
        self.squared_gradients: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.zeros_like(layer.weights), np.zeros_like(layer.biases)) for layer in self.trainable
        ]
        
    def gradient_step(self) -> None:
        """Performs a single step of gradient descent using AdaGrad's adjusted learning rate. It updates each layer's weights and biases based on accumulated squared gradients.
        """
        for i, (layer, (sum_squared_weights, sum_squared_biases)) in enumerate(zip(self.trainable, self.squared_gradients)):
            # Accumulate the square of gradients for weights and biases
            sum_squared_weights += layer.d_weights ** 2
            sum_squared_biases += layer.d_biases ** 2

            # Update weights and biases with AdaGrad adjusted learning rate
            layer.weights -= self.learning_rate * layer.d_weights / (np.sqrt(sum_squared_weights) + self.e)
            layer.biases -= self.learning_rate * layer.d_biases / (np.sqrt(sum_squared_biases) + self.e)
            
            self.squared_gradients[i] = (sum_squared_weights, sum_squared_biases)