import numpy as np
from typing import Type, List, Tuple
from layers import Layer, Dense
from losses import Loss
from optimizers import Optimizer
from data_utils import DataLoader
from lr_schedulers import LRScheduler


class NeuralNetwork:
    def __init__(self) -> None:        
        """Initializes the neural network by setting up an empty list for layers and a placeholder for the loss function.
        """
        
        self.layers: List[Type[Layer]] = []
        self.trainable: List[Type[Dense]] = []
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
        
        if hasattr(layer, "weights"):
            self.trainable.append(layer)
        
    def config(self, loss_func: Type[Loss], optimizer: Type[Optimizer]) -> None:
        """Configures the neural network with a loss function for training

        Args:
            loss_func (Type[Loss]): The loss function class to be used for training.
        """
        
        self.loss: Type[Loss] = loss_func
        self.optimizer: Type[Optimizer] = optimizer
    """

    Args:
        train_data (DataLoader): 
        n_epochs (int): 
        val_data (DataLoader, optional): 
        lr_scheduler (Type[LRScheduler] | None, optional):  . Defaults to None.
        print_every (int, optional): . Defaults to 1.
        early_stop (int | None, optional): Stops . Defaults to None.
    """
        
    def train(self, train_data: DataLoader, n_epochs: int, val_data: DataLoader | None = None, 
              lr_scheduler: Type[LRScheduler] | None = None, print_every: int=1, return_best: bool = False) -> None:
        """Trains the neural network using the provided dataset for a specified number of epochs with backpropagation.

        Args:
            train_data (DataLoader): Data loader containig training feature data and target values.
            n_epochs (int): Number of epochs to train the network.
            val_data (DataLoader | None, optional): Data loader containig validation feature data and target values. Defaults to None.
            lr_scheduler (Type[LRScheduler] | None, optional): Determines the learning rate based on the current epoch. Defaults to None.
            print_every (int, optional): How often the training info is printed. Defaults to 1.
            return_best (bool, optional): If True, the model applies the parameters recorded at the epoch with the smallest loss (val loss if val data is provided, otherwise train loss). Defaults to False.
        """
        
        best_params = None
        for epoch in range(1, n_epochs+1):
            # Training and validation loss
            running_loss: np.float64 = 0.
            running_val_loss: np.float64 = 0. # Calculated only if validation dataset is provided
            
            # Iterating through batches
            for sample_batch, target in train_data:
                output, params = self.forward(sample_batch, training=True)
                
                running_loss += self.loss.calculate_loss(output, target)
                
                self.backward()
                
                self.optimizer.gradient_step()
                
                # Training validation
                if val_data is not None:
                    for val_sample_batch, val_target in val_data:
                        val_output = self.forward(val_sample_batch, training=False)
                        
                        running_val_loss += self.loss.calculate_loss(val_output, val_target, training=False)
                
                # Adjusting learning rate if learning rate scheduler is provided
                if lr_scheduler is not None:
                    lr_scheduler.schedule(epoch)
                
            train_loss = running_loss / len(train_data)
            val_loss = running_val_loss / len(val_data) if val_data is not None else None
            
            # Comparing losses for best model parameters
            # If validation loss does not exist (validation data is not provided),
            # training loss serves as model efficiency metric, otherwise validation loss is efficiency metric
            if return_best:
                if best_params is None:
                    best_params = {"params": params, 
                                   "loss": val_loss if val_loss is not None else train_loss}
                else:
                    if val_loss is not None:
                        best_params = {"params": params, "loss": val_loss} if val_loss < best_params["loss"] \
                            else best_params
                    else:
                        best_params = {"params": params, "loss": train_loss} if train_loss < best_params["loss"] \
                            else best_params
            
            # Printing efficiency of model for current epoch
            if epoch % print_every == 0:
                self.monitor_progress(epoch, train_loss, val_loss)
                
        if best_params:
            self.apply_best_params(best_params)
            print(f"Best loss: {best_params["loss"]}")
        
    def forward(self, inputs: np.ndarray, training: bool = True) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]] | np.ndarray:
        """Performs a forward pass through the neural network.

        Args:
            inputs (np.ndarray): 
            training (bool, optional): Indicates whether the network is in training mode. Defaults to True.
            
        Returns: 
            Tuple[np.ndarray, List[Dict[str, np.ndarray]]] | np.ndarray: 
            - During training (training=True):
                A tuple containing:
                  - Final output of the forward pass (np.ndarray).
                  - List of dictionaries with trainable parameters (weights, biases).
            - During inference (training=False):
                Final output of the forward pass (np.ndarray).
        """
        
        output = inputs.copy()
        params = []
        
        for layer in self.layers:
            if hasattr(layer, "dropout_rate"):
                layer.forward(output, training=training)
            else:
                layer.forward(output)
                if training and hasattr(layer, "weights"):
                    params.append({"weights": layer.weights, "biases": layer.biases})
            output = layer.output
        
        if training:
            return output, params
        
        return output
    
    def backward(self) -> None:
        """Performs backpropagation through the network, starting with the loss function and propagating the gradients backward through each layer.
        """
        
        self.loss.backward()
        d_output = self.loss.d_output
        for layer in reversed(self.layers):
            layer.backward(d_output)
            d_output = layer.d_output
            
    def apply_best_params(self, best_params):
        for dense, params in zip(self.trainable, best_params["params"]):
            dense.weights = params["weights"]
            dense.biases = params["biases"]            
            
    def predict(self, inputs: np.ndarray) -> np.ndarray | float:
        return self.forward(inputs, training=False) 
    
    @staticmethod
    def monitor_progress(epoch: int, loss: float, val_loss: float | None) -> None:
        print_str = f"Epoch {epoch} : loss {loss} "
        
        if val_loss is not None:
            print_str += f"val loss {val_loss}"
            
        print(print_str)