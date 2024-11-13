import numpy as np
from typing import Type, List
from layers import Layer
from losses import Loss
from optimizers import Optimizer
from data_utils import DataLoader


class NeuralNetwork:
    def __init__(self) -> None:        
        """Initializes the neural network by setting up an empty list for layers and a placeholder for the loss function.
        """
        
        self.layers: List[Type[Layer]] = []
        self.loss: Type[Loss] = None
        self.optimizer: Type[Optimizer] = None
        
    def add_layer(self, layer: Type[Layer]) -> None:
        """Adds a layer to the neural network. The first layer must be a linear layer.

        Args:
            layer (Type[Layer]): The layer to be added. If it is first layer, it should be Linear.

        Raises:
            ValueError: If the first layer is not a linear layer.
        """
        
        if len(self.layers) == 0 and not hasattr(layer, "weights"):
            raise ValueError("First layer must be Linear")
        self.layers.append(layer)
        
    def config(self, loss_func: Type[Loss], optimizer: Type[Optimizer]) -> None:
        """Configures the neural network with a loss function for training

        Args:
            loss_func (Type[Loss]): The loss function class to be used for training.
        """
        
        self.loss: Type[Loss] = loss_func
        self.optimizer: Type[Optimizer] = optimizer
        
    def train(self, data: DataLoader, n_epochs: int, print_every: int=1) -> None:
        """Trains the neural network using the provided dataset for a specified number of epochs with backpropagation.

        Args:
            data (DataLoader): Data loader containig feature data and target values.
            n_epochs (int): Number of epochs to train the network.
            print_every (int, optional): How often the training info is printed. Defaults to 1.
        """
        
        for epoch in range(1, n_epochs+1):
            running_loss = 0.
            for sample_batch, target in data:
                output = self.forward(sample_batch)
                
                self.loss.forward(output, target)
                
                self.backward()
                
                self.optimizer.gradient_step()
                        
                running_loss += self.loss.output
                
            if epoch % print_every == 0:
                print(f"Epoch {epoch} : loss {running_loss / len(data)}")
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Performs a forward pass through the network by propagating inputs through each layer.

        Args:
            inputs (np.ndarray): Input data for the network.

        Returns:
            np.ndarray: The final output after passing through all layers.
        """
        
        output = inputs.copy()
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output
    
    def backward(self) -> None:
        """Performs backpropagation through the network, starting with the loss function and propagating the gradients backward through each layer.
        """
        
        self.loss.backward()
        d_output = self.loss.d_output
        for layer in reversed(self.layers):
            layer.backward(d_output)
            d_output = layer.d_output