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
    
    def __init__(self, learning_rate: float, layers: List[Type[Layer]], epsilon: float = 1e-9) -> None:
        """Optimizer that adapts the learning rate for each parameter by accumulating the square of gradients over time

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            layers (List[Type[Layer]]): Layers of the model.
            epsilon (float, optional): A small constant to prevent division by zero when adjusting gradients. Defaults to 1e-9.
        """
        super().__init__(learning_rate, layers)
    
        self.epsilon: float = epsilon
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
            layer.weights -= self.learning_rate * layer.d_weights / (np.sqrt(sum_squared_weights) + self.epsilon)
            layer.biases -= self.learning_rate * layer.d_biases / (np.sqrt(sum_squared_biases) + self.epsilon)
            
            self.squared_gradients[i] = (sum_squared_weights, sum_squared_biases)
            

class AdaDelta(Optimizer):
    
    def __init__(self, learning_rate: float, layers: List[Type[Layer]], rho: float = 0.95, epsilon: float = 1e-9) -> None:
        """AdaDelta is an adaptive learning rate optimization algorithm that extends AdaGrad to overcome one of its key limitations: the rapid decrease in effective learning rate as gradients accumulate.

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            layers (List[Type[Layer]]): Layers of the model.
            rho (float, optional): Decay rate. Defaults to 0.95.
            epsilon (float, optional): A small constant to prevent division by zero when adjusting gradients. Defaults to 1e-9.
        """
        super().__init__(learning_rate, layers)
        self.rho = rho
        self.epsilon = epsilon
        
        # Initialize accumulated gradient and update arrays
        self.accumulated_gradients: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.zeros_like(layer.weights), np.zeros_like(layer.biases)) for layer in self.trainable
        ]
        
        self.accumulated_updates: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.zeros_like(layer.weights), np.zeros_like(layer.biases)) for layer in self.trainable
        ]
        
        
    def gradient_step(self) -> None:
        """Performs a single step of gradient descent using adjusted learning rate. Instead of directly scaling the learning rate by the gradient, AdaDelta scales it based on past updates, normalizing the step size according to an adaptive learning rate per parameter.
        """
        for i, (layer, (acc_grad_weights, acc_grad_biases), (acc_update_weights, acc_update_biases)) in \
            enumerate(zip(self.trainable, self.accumulated_gradients, self.accumulated_updates)):
            
            # Update accumulated gradients with the current gradients
            acc_grad_weights = self.rho * acc_grad_weights + (1 - self.rho) * layer.d_weights ** 2
            acc_grad_biases = self.rho * acc_grad_biases + (1 - self.rho) * layer.d_biases ** 2
            
            # Calculate parameter update values
            update_weights = -(np.sqrt(acc_update_weights + self.epsilon) / np.sqrt(acc_grad_weights + self.epsilon)) * layer.d_weights
            update_biases = -(np.sqrt(acc_update_biases + self.epsilon) / np.sqrt(acc_grad_biases + self.epsilon)) * layer.d_biases
            
            # Apply updates to weights and biases
            layer.weights += update_weights
            layer.biases += update_biases
            
            # Update accumulated updates with the calculated updates
            acc_update_weights = self.rho * acc_update_weights + (1 - self.rho) * update_weights ** 2
            acc_update_biases = self.rho * acc_update_biases + (1 - self.rho) * update_biases ** 2
            
            # Store the updated accumulated values
            self.accumulated_gradients[i] = (acc_grad_weights, acc_grad_biases)
            self.accumulated_updates[i] = (acc_update_weights, acc_update_biases)
            
            
class RMSprop(Optimizer):
    def __init__(self, learning_rate: float, layers: List[Type[Layer]], rho: float = 0.9, epsilon: float = 1e-8) -> None:
        """Adaptive learning rate optimization algorithm, similar to AdaGrad and AdaDelta, but designed specifically to address AdaGrad's rapid decay in learning rate.

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            layers (List[Type[Layer]]): Layers of the model.
            rho (float, optional): Decay rate. Defaults to 0.9.
            epsilon (float, optional): A small constant to prevent division by zero when adjusting gradients. Defaults to 1e-8.
        """
        super().__init__(learning_rate, layers)
        self.rho = rho
        self.epsilon = epsilon
        
        # Initialize moving average of squared gradients
        self.squared_gradients: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.zeros_like(layer.weights), np.zeros_like(layer.biases)) for layer in self.trainable
        ]
        
    def gradient_step(self) -> None:
        """Applies RMSprop update to each trainable layer."""
        for i, (layer, (accum_grad_weights, accum_grad_biases)) in enumerate(zip(self.trainable, self.squared_gradients)):
            # Accumulate the squared gradients
            accum_grad_weights = self.rho * accum_grad_weights + (1 - self.rho) * layer.d_weights ** 2
            accum_grad_biases = self.rho * accum_grad_biases + (1 - self.rho) * layer.d_biases ** 2
            
            # Update weights and biases using RMSprop rule
            layer.weights -= self.learning_rate * layer.d_weights / (np.sqrt(accum_grad_weights) + self.epsilon)
            layer.biases -= self.learning_rate * layer.d_biases / (np.sqrt(accum_grad_biases) + self.epsilon)
            
            # Store updated accumulated gradients
            self.squared_gradients[i] = (accum_grad_weights, accum_grad_biases)