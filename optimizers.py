import numpy as np
from typing import Type, List, Tuple
from layers import Dense

class Optimizer:
    
    def __init__(self, learning_rate: float, trainable: List[Type[Dense]]) -> None:
        """algorithm used to change the attributes of neural network (like weights and learning rate) to reduce the losses.

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            trainable (List[Type[Layer]]): Dense layers of the model.
        """
        self.learning_rate: float = learning_rate
        self.trainable: List[Dense] = trainable
        
    def gradient_step(self) -> None:
        """Gradient step, should be implemented by subclasses."""
        raise NotImplementedError("Gradient step should be implemented by subclasses.")


class SGD(Optimizer):
    
    def __init__(self, learning_rate: float, trainable: List[Type[Dense]], momentum: float = 0) -> None:
        """Stochastic Gradient Descent updates the model's weights by taking small steps proportional to the negative gradient of the loss function

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            layers (List[Type[Layer]]): Layers of the model.
            momentum (float, optional): Momentum helps SGD accelerate by using past gradients to smooth out the steps taken. Defaults to 0.
        """
        super().__init__(learning_rate, trainable)
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
    
    def __init__(self, learning_rate: float, trainable: List[Type[Dense]], epsilon: float = 1e-9) -> None:
        """Optimizer that adapts the learning rate for each parameter by accumulating the square of gradients over time

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            trainable (List[Type[Layer]]): Dense layers of the model.
            epsilon (float, optional): A small constant to prevent division by zero when adjusting gradients. Defaults to 1e-9.
        """
        super().__init__(learning_rate, trainable)
    
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
    
    def __init__(self, learning_rate: float, trainable: List[Type[Dense]], rho: float = 0.95, epsilon: float = 1e-9) -> None:
        """AdaDelta is an adaptive learning rate optimization algorithm that extends AdaGrad to overcome one of its key limitations: the rapid decrease in effective learning rate as gradients accumulate.

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            trainable (List[Type[Layer]]): Dense layers of the model.
            rho (float, optional): Decay rate. Defaults to 0.95.
            epsilon (float, optional): A small constant to prevent division by zero when adjusting gradients. Defaults to 1e-9.
        """
        super().__init__(learning_rate, trainable)
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
    def __init__(self, learning_rate: float, trainable: List[Type[Dense]], rho: float = 0.9, epsilon: float = 1e-9) -> None:
        """Adaptive learning rate optimization algorithm, similar to AdaGrad and AdaDelta, but designed specifically to address AdaGrad's rapid decay in learning rate.

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            trainable (List[Type[Layer]]): Dense layers of the model.
            rho (float, optional): Decay rate. Defaults to 0.9.
            epsilon (float, optional): A small constant to prevent division by zero when adjusting gradients. Defaults to 1e-9.
        """
        super().__init__(learning_rate, trainable)
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
            

class Adam(Optimizer):
    def __init__(self, learning_rate: float, trainable: List[Type[Dense]], beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-9) -> None:
        """Adam (Adaptive Moment Estimation) is an optimization algorithm that combines the advantages of two other popular optimization algorithms: Momentum and RMSprop.

        Args:
            learning_rate (float): Tuning parameter determines the step size at each iteration while moving toward a minimum of a loss.
            trainable (List[Type[Layer]]): Dense layers of the model.
            beta1 (float, optional): Decay rate for the moving averages of the first moment . Defaults to 0.9.
            beta2 (float, optional): Decay rate for the moving averages of the second moment. Defaults to 0.999.
            epsilon (float, optional): A small constant to prevent division by zero when adjusting gradients. Defaults to 1e-9.
        """
        super().__init__(learning_rate, trainable)
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize moment estimates
        self.m = [(np.zeros_like(layer.weights), np.zeros_like(layer.biases)) for layer in self.trainable]
        self.v = [(np.zeros_like(layer.weights), np.zeros_like(layer.biases)) for layer in self.trainable]
        self.t = 0  # Time step (used for bias correction)

    def gradient_step(self) -> None:
        self.t += 1  # Increment time step

        for i, (layer, (m_w, m_b), (v_w, v_b)) in enumerate(zip(self.trainable, self.m, self.v)):
            # Update biased first moment estimate (m)
            m_w = self.beta1 * m_w + (1 - self.beta1) * layer.d_weights
            m_b = self.beta1 * m_b + (1 - self.beta1) * layer.d_biases

            # Update biased second moment estimate (v)
            v_w = self.beta2 * v_w + (1 - self.beta2) * (layer.d_weights ** 2)
            v_b = self.beta2 * v_b + (1 - self.beta2) * (layer.d_biases ** 2)

            # Correct bias for first moment estimate (m_hat)
            m_hat_w = m_w / (1 - self.beta1 ** self.t)
            m_hat_b = m_b / (1 - self.beta1 ** self.t)

            # Correct bias for second moment estimate (v_hat)
            v_hat_w = v_w / (1 - self.beta2 ** self.t)
            v_hat_b = v_b / (1 - self.beta2 ** self.t)

            # Reshape m_hat and v_hat to match layer weights and biases shapes
            m_hat_w = np.reshape(m_hat_w, layer.weights.shape)
            m_hat_b = np.reshape(m_hat_b, layer.biases.shape)
            v_hat_w = np.reshape(v_hat_w, layer.weights.shape)
            v_hat_b = np.reshape(v_hat_b, layer.biases.shape)

            # Update the weights and biases
            layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

            # Update moment estimates
            self.m[i] = (m_w, m_b)
            self.v[i] = (v_w, v_b)